import os
import sys
import math
import random
import pickle
import skimage
import numpy as np
import tensorflow as tf
import skimage.io as imgio
from datetime import datetime
import skimage.transform as imgtf

class dataset_dsrg():
    def __init__(self,config={}):
        self.config = config
        self.w,self.h = self.config.get("input_size",(321,321))
        self.categorys = self.config.get("categorys",["train"])
        self.category_num = self.config.get("category_num",21)
        self.main_path = self.config.get("main_path",os.path.join("data","VOCdevkit","VOC2012"))
        self.ignore_label = self.config.get("ignore_label",255)
        self.default_category = self.config.get("default_category",self.categorys[0])
        self.img_mean = np.ones((self.w,self.h,3))
        self.img_mean[:,:,0] *= 104.00698793
        self.img_mean[:,:,1] *= 116.66876762
        self.img_mean[:,:,2] *= 122.67891434

        self.data_f,self.data_len = self.get_data_f()

    def get_data_f(self):
        self.cues_data = pickle.load(open("data/localization_cues-sal.pickle","rb"),encoding="iso-8859-1")
        data_f = {}
        data_len = {}
        for category in self.categorys:
            data_f[category] = {"img":[],"gt":[],"label":[],"id":[],"id_for_slice":[]}
            data_len[category] = 0
        for one in self.categorys:
           assert one == "train", "extra category found!"
           with open(os.path.join("data","input_list.txt"),"r") as f:
               for line in f.readlines():
                   line = line.rstrip("\n")
                   id_name,id_identy = line.split(" ")
                   id_name = id_name[:-4] # then id_name is like '2007_007028'
                   data_f[one]["id"].append(id_name)
                   data_f[one]["id_for_slice"].append(id_identy)
                   data_f[one]["img"].append(os.path.join(self.main_path,"JPEGImages","%s.jpg" % id_name))
                   data_f[one]["gt"].append(os.path.join(self.main_path,"SegmentationClassAug","%s.png" % id_name))
               
               if "length" in self.config:
                   length = self.config["length"]
                   data_f[one]["id"] = data_f[one]["id"][:length]
                   data_f[one]["id_for_slice"] = data_f[one]["id_for_slice"][:length]
                   data_f[one]["img"] = data_f[one]["img"][:length]
                   data_f[one]["gt"] = data_f[one]["gt"][:length]
                   print("id:%s" % str(data_f[one]["id"]))
                   print("img:%s" % str(data_f[one]["img"]))
                   print("id_for_slice:%s" % str(data_f[one]["id_for_slice"]))

           data_len[one] = len(data_f[one]["id"])

        print("len:%s" % str(data_len))
        return data_f,data_len

    def next_batch(self,category=None,batch_size=None,epoches=-1):
        if category is None: category = self.default_category
        if batch_size is None:
            batch_size = self.config.get("batch_size",1)
        dataset = tf.data.Dataset.from_tensor_slices({
            "id":self.data_f[category]["id"],
            "id_for_slice":self.data_f[category]["id_for_slice"],
            "img_f":self.data_f[category]["img"],
            "gt_f":self.data_f[category]["gt"],
            })
        def m(x):
            id_ = x["id"]

            img_f = x["img_f"]
            img_raw = tf.read_file(img_f)
            img = tf.image.decode_image(img_raw)
            gt_f = x["gt_f"]
            gt_raw = tf.read_file(gt_f)
            gt = tf.image.decode_image(gt_raw)

            id_for_slice = x["id_for_slice"]
            def get_data(identy):
                identy = identy.decode()
                tag = np.zeros([self.category_num])
                tag[self.cues_data["%s_labels" % identy]] = 1.0
                tag[0] = 1.0
                cues = np.zeros([41,41,21])
                cues_i = self.cues_data["%s_cues" % identy]
                cues[cues_i[1],cues_i[2],cues_i[0]] = 1.0
                return tag.astype(np.float32),cues.astype(np.float32)

            tag,cues = tf.py_func(get_data,[id_for_slice],[tf.float32,tf.float32])
            img,gt,cues = self.image_preprocess(img,gt,cues,flip=True)

            img = tf.reshape(img,[self.h,self.w,3])
            gt = tf.reshape(gt,[self.h,self.w,1])
            tag.set_shape([21])
            cues.set_shape([41,41,21])

            return img,gt,tag,cues,id_

        dataset = dataset.repeat(epoches)
        dataset = dataset.shuffle(self.data_len[category])
        dataset = dataset.map(m)
        dataset = dataset.batch(batch_size)
        iterator = dataset.make_initializable_iterator()
        img,gt,tag,cues,id_ = iterator.get_next()
            
        return img,gt,tag,cues,id_,iterator

    def image_preprocess(self,img,gt,cues,flip=False):
        #img = tf.image.resize_image_with_crop_or_pad(img, self.h, self.w)
        #gt = tf.image.resize_image_with_crop_or_pad(gt, self.h, self.w)
        img = tf.expand_dims(img,axis=0)
        img = tf.image.resize_bilinear(img,(self.h,self.w))
        img = tf.squeeze(img,axis=0)
        gt = tf.expand_dims(gt,axis=0)
        gt = tf.image.resize_nearest_neighbor(gt,(self.h,self.w))
        gt = tf.squeeze(gt,axis=0)

        r,g,b = tf.split(axis=2,num_or_size_splits=3,value=img)
        img = tf.cast(tf.concat([b,g,r],2),dtype=tf.float32)
        img -= self.img_mean

        if flip is True:
            r = tf.random_uniform([1])
            r = tf.reduce_sum(r)
            img = tf.cond(r < 0.5, lambda:tf.image.flip_left_right(img),lambda:img)
            gt = tf.cond(r < 0.5, lambda:tf.image.flip_left_right(gt),lambda:gt)
            cues = tf.cond(r < 0.5, lambda:tf.image.flip_left_right(cues),lambda:cues)

        return img,gt,cues
