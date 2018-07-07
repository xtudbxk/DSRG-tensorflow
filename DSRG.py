import os
import sys
import time
import copy
from multiprocessing import Pool
import numpy as np
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]

from pythonlib.crf import crf_inference
from pythonlib.dataset_DSRG import dataset_dsrg as dataset
from pythonlib.CC_labeling_8 import CC_lab

def single_generate_seed_step(params):
    #th_f,th_b = 0.85,0.99
    th_f,th_b = 0.5,0.7
    tag,cue,prob = params
    existing_prob = prob*tag
    existing_prob_argmax = np.argmax( existing_prob, axis=2) + 1 # to tell the background pixel and the not-satisfy-condition pixel
    tell_where_is_foreground_mask = (existing_prob_argmax > 1).astype(np.uint8)

    existing_prob_fg_th_mask = (np.sum((existing_prob[:,:,1:] > th_f).astype(np.uint8),axis=2) > 0.5).astype(np.uint8) # if there is one existing category's score is bigger than th_f, the the mask is 1 for this pixel
    existing_prob_bg_th_mask = (np.sum((existing_prob[:,:,0:1] > th_b).astype(np.uint8),axis=2) > 0.5).astype(np.uint8)

    label_map = (existing_prob_fg_th_mask*tell_where_is_foreground_mask + existing_prob_bg_th_mask*(1-tell_where_is_foreground_mask))*existing_prob_argmax
    # the label map is a two-dimensional map to show which category satisify the following three conditions for each pixel
    # 1. the category is in the tags of the image
    # 2. the category has a max probs among the tags
    # 3. the prob of the category is bigger that the threshold
    # and those three conditions is the similarity criteria
    # for the value in label_map, 0 is for no category satisifies the conditions, n is for the category n-1 satisifies the conditions

    cls_index = np.where(tag>0.5)[2] # the existing labels index
    for c in cls_index:
        mat = (label_map == (c+1))
        mat = mat.astype(int)
        cclab = CC_lab(mat)
        cclab.connectedComponentLabel() # this divide each connected region into a group, and update the value of cclab.labels which is a two-dimensional list to show the group index of each pixel
        high_confidence_set_label = set() # this variable colloects the connected region index
        for (x,y), value in np.ndenumerate(mat):
            if value == 1 and cue[x,y,c] == 1:
                high_confidence_set_label.add(cclab.labels[x][y])
            elif value == 1 and np.sum(cue[x,y,:]) == 1:
                cclab.labels[x][y] = -1
        for (x,y),value in np.ndenumerate(np.array(cclab.labels)):
            if value in high_confidence_set_label:
                cue[x,y,c] = 1
    return np.expand_dims(cue,axis=0)


class DSRG():
    def __init__(self,config):
        self.config = config
        self.h,self.w = self.config.get("input_size",(25,25))
        self.category_num = self.config.get("category_num",21)
        self.accum_num = self.config.get("accum_num",1)
        self.data = self.config.get("data",None)
        self.net = {}
        self.weights = {}
        self.trainable_list = []
        self.loss = {}

        self.variables={"total":[]}
        self.l2loss = {"total":0}

        self.min_prob = 0.0001
        self.stride = {}
        self.stride["input"] = 1

        # different lr for different variable
        self.lr_1_list = []
        self.lr_2_list = []
        self.lr_10_list = []
        self.lr_20_list = []

        # multiprocess to accelerate the region growing
        self.pool = Pool(10)

    def build(self,net_input,net_label,net_tag,net_cues):
        if "output" not in self.net:
            with tf.name_scope("placeholder"):
                self.net["input"] = net_input
                self.net["label"] = net_label # [None, self.h,self.w,1], int32
                self.net["tag"] = net_tag # [None, self.category_num], int32
                self.net["cues"] = net_cues # [None,41,41,self.category_num])
                self.net["drop_prob"] = tf.Variable(0.5)

            self.net["output"] = self.create_network()
        return self.net["output"]

    def create_network(self):
        if "init_model_path" in self.config:
            self.load_init_model()
        with tf.name_scope("vgg") as scope:
            # build block
            block = self.build_block("input",["conv1_1","relu1_1","conv1_2","relu1_2","pool1"])
            block = self.build_block(block,["conv2_1","relu2_1","conv2_2","relu2_2","pool2"])
            block = self.build_block(block,["conv3_1","relu3_1","conv3_2","relu3_2","conv3_3","relu3_3","pool3"])
            block = self.build_block(block,["conv4_1","relu4_1","conv4_2","relu4_2","conv4_3","relu4_3","pool4"])
            block = self.build_block(block,["conv5_1","relu5_1","conv5_2","relu5_2","conv5_3","relu5_3","pool5","pool5a"])
            fc1 = self.build_fc(block,["fc6_1","relu6_1","drop6_1","fc7_1","relu7_1","drop7_1","fc8_1"], dilate_rate=6)
            fc2 = self.build_fc(block,["fc6_2","relu6_2","drop6_2","fc7_2","relu7_2","drop7_2","fc8_2"], dilate_rate=12)
            fc3 = self.build_fc(block,["fc6_3","relu6_3","drop6_3","fc7_3","relu7_3","drop7_3","fc8_3"], dilate_rate=18)
            fc4 = self.build_fc(block,["fc6_4","relu6_4","drop6_4","fc7_4","relu7_4","drop7_4","fc8_4"], dilate_rate=24)
            self.net["fc8"] = self.net[fc1]+self.net[fc2]+self.net[fc3]+self.net[fc4]

            # SEC
            softmax = self.build_sp_softmax("fc8","fc8-softmax")
            new_seed = self.build_dsrg_layer("cues","fc8-softmax","new_cues")
            crf = self.build_crf("fc8","crf")

            return self.net[crf] # NOTE: crf is log-probability

    def build_block(self,last_layer,layer_lists):
        for layer in layer_lists:
            if layer.startswith("conv"):
                if layer[4] != "5":
                    with tf.name_scope(layer) as scope:
                        self.stride[layer] = self.stride[last_layer]
                        weights,bias = self.get_weights_and_bias(layer)
                        self.net[layer] = tf.nn.conv2d( self.net[last_layer], weights, strides = [1,1,1,1], padding="SAME", name="conv")
                        self.net[layer] = tf.nn.bias_add( self.net[layer], bias, name="bias")
                        last_layer = layer
                if layer[4] == "5":
                    with tf.name_scope(layer) as scope:
                        self.stride[layer] = self.stride[last_layer]
                        weights,bias = self.get_weights_and_bias(layer)
                        self.net[layer] = tf.nn.atrous_conv2d( self.net[last_layer], weights, rate=2, padding="SAME", name="conv")
                        self.net[layer] = tf.nn.bias_add( self.net[layer], bias, name="bias")
                        last_layer = layer
            if layer.startswith("relu"):
                with tf.name_scope(layer) as scope:
                    self.stride[layer] = self.stride[last_layer]
                    self.net[layer] = tf.nn.relu( self.net[last_layer],name="relu")
                    last_layer = layer
            elif layer.startswith("pool5a"):
                with tf.name_scope(layer) as scope:
                    self.stride[layer] = self.stride[last_layer]
                    self.net[layer] = tf.nn.avg_pool( self.net[last_layer], ksize=[1,3,3,1], strides=[1,1,1,1],padding="SAME",name="pool")
                    last_layer = layer
            elif layer.startswith("pool"):
                if layer[4] not in ["4","5"]:
                    with tf.name_scope(layer) as scope:
                        self.stride[layer] = 2 * self.stride[last_layer]
                        self.net[layer] = tf.nn.max_pool( self.net[last_layer], ksize=[1,3,3,1], strides=[1,2,2,1],padding="SAME",name="pool")
                        last_layer = layer
                if layer[4] in ["4","5"]:
                    with tf.name_scope(layer) as scope:
                        self.stride[layer] = self.stride[last_layer]
                        self.net[layer] = tf.nn.max_pool( self.net[last_layer], ksize=[1,3,3,1], strides=[1,1,1,1],padding="SAME",name="pool")
                        last_layer = layer
        return last_layer

    def build_fc(self,last_layer, layer_lists, dilate_rate=12):
        for layer in layer_lists:
            if layer.startswith("fc"):
                with tf.name_scope(layer) as scope:
                    weights,bias = self.get_weights_and_bias(layer)
                    if layer.startswith("fc6"):
                        self.net[layer] = tf.nn.atrous_conv2d( self.net[last_layer], weights, rate=dilate_rate, padding="SAME", name="conv")
                    else:
                        self.net[layer] = tf.nn.conv2d( self.net[last_layer], weights, strides = [1,1,1,1], padding="SAME", name="conv")
                    self.net[layer] = tf.nn.bias_add( self.net[layer], bias, name="bias")
                    last_layer = layer
            if layer.startswith("batch_norm"):
                with tf.name_scope(layer) as scope:
                    self.net[layer] = tf.contrib.layers.batch_norm(self.net[last_layer])
                    last_layer = layer
            if layer.startswith("relu"):
                with tf.name_scope(layer) as scope:
                    self.net[layer] = tf.nn.relu( self.net[last_layer])
                    last_layer = layer
            if layer.startswith("drop"):
                with tf.name_scope(layer) as scope:
                    self.net[layer] = tf.nn.dropout( self.net[last_layer],self.net["drop_prob"])
                    last_layer = layer

        return last_layer

    def build_sp_softmax(self,last_layer,layer):
        preds_max = tf.reduce_max(self.net[last_layer],axis=3,keepdims=True)
        preds_exp = tf.exp(self.net[last_layer] - preds_max)
        self.net[layer] = preds_exp / tf.reduce_sum(preds_exp,axis=3,keepdims=True) + self.min_prob
        self.net[layer] = self.net[layer] / tf.reduce_sum(self.net[layer],axis=3,keepdims=True)
        return layer

    def build_crf(self,featmap_layer,layer):
        origin_image = self.net["input"] + self.data.img_mean
        origin_image_zoomed = tf.image.resize_bilinear(origin_image,(41,41))
        featemap = self.net[featmap_layer]
        featemap_zoomed = tf.image.resize_bilinear(featemap,(41,41))
    
        def crf(featemap,image):
            #crf_config = {"g_sxy":3,"g_compat":3,"bi_sxy":80,"bi_srgb":13,"bi_compat":10,"iterations":5} # for test
            crf_config = {"g_sxy":3/12,"g_compat":3,"bi_sxy":80/12,"bi_srgb":13,"bi_compat":10,"iterations":5} # for train, in previous iter, probability is not accurate, so we should weaken the weight of position

            batch_size = featemap.shape[0]
            image = image.astype(np.uint8)
            ret = np.zeros(featemap.shape,dtype=np.float32)
            for i in range(batch_size):
                ret[i,:,:,:] = crf_inference(image[i],crf_config,self.category_num,featemap[i],use_log=True)

            ret[ret < self.min_prob] = self.min_prob
            ret /= np.sum(ret,axis=3,keepdims=True)
            ret = np.log(ret)
            return ret.astype(np.float32)
        
        crf = tf.py_func(crf,[featemap_zoomed,origin_image_zoomed],tf.float32) # shape [N, h, w, C], RGB or BGR doesn't matter
        self.net[layer] = crf

        return layer

    def build_dsrg_layer(self,seed_layer,prob_layer,layer):
        def generate_seed_step(tags,cues,probs):
            ''' tags shape: [-1,21]
                cues shape: [-1,41,41,21]
                probs shape: [-1,41,41,21]
            '''
            tags = np.reshape(tags,[-1,1,1,self.category_num])

            params_list = []
            for i in range(batch_size):
                params_list.append([tags[i],cues[i],probs[i]])

            ret = self.pool.map(single_generate_seed_step,params_list)
            
            new_cues = ret[0]
            for i in range(1,batch_size):
                new_cues = np.concatenate([new_cues,ret[i]],axis=0)

            return new_cues

        self.net[layer] = tf.py_func(generate_seed_step,[self.net["tag"],self.net[seed_layer],self.net[prob_layer]],tf.float32)
        return layer

    def load_init_model(self):
        model_path = self.config["init_model_path"]
        self.init_model = np.load(model_path,encoding="latin1").item()
        print("load init model success: %s" % model_path)

    def get_weights_and_bias(self,layer,shape=None):
        print("layer: %s" % layer)
        if layer in self.weights:
            return self.weights[layer]
        if shape is not None:
            pass
        elif layer.startswith("conv"):
            shape = [3,3,0,0]
            if layer == "conv1_1":
                shape[2] = 3
            else:
                shape[2] = 64 * self.stride[layer]
                if shape[2] > 512: shape[2] = 512
                if layer in ["conv2_1","conv3_1","conv4_1"]: shape[2] = int(shape[2]/2)
            shape[3] = 64 * self.stride[layer]
            if shape[3] > 512: shape[3] = 512
        elif layer.startswith("fc"):
            if layer.startswith("fc6"):
                shape = [3,3,512,1024]
            if layer.startswith("fc7"):
                shape = [1,1,1024,1024]
            if layer.startswith("fc8"):
                shape = [1,1,1024,self.category_num]
        if "init_model_path" not in self.config:
            init = tf.random_normal_initializer(stddev=0.01)
            weights = tf.get_variable(name="%s_weights" % layer,initializer=init, shape = shape)
            init = tf.constant_initializer(0)
            bias = tf.get_variable(name="%s_bias" % layer,initializer=init, shape = [shape[-1]])
        else:
            if layer.startswith("fc8"):
                init = tf.contrib.layers.xavier_initializer(uniform=True)
            else:
                init = tf.constant_initializer(self.init_model[layer]["w"])
            weights = tf.get_variable(name="%s_weights" % layer,initializer=init,shape = shape)
            if layer.startswith("fc8"):
                init = tf.constant_initializer(0)
            else:
                init = tf.constant_initializer(self.init_model[layer]["b"])
            bias = tf.get_variable(name="%s_bias" % layer,initializer=init,shape = [shape[-1]])
        self.weights[layer] = (weights,bias)
        if layer.startswith("fc8"):
            self.lr_10_list.append(weights)
            self.lr_20_list.append(bias)
        else:
            self.lr_1_list.append(weights)
            self.lr_2_list.append(bias)
        self.trainable_list.append(weights)
        self.trainable_list.append(bias)
        self.variables["total"].append(weights)
        self.variables["total"].append(bias)

        return weights,bias

    def getloss(self):
        loss = 0
        
        # for DSRG
        seed_loss = self.get_balanced_seed_loss(self.net["fc8-softmax"],self.net["new_cues"])
        constrain_loss = self.get_constrain_loss(self.net["fc8-softmax"],self.net["crf"])
        self.loss["seed"] = seed_loss
        self.loss["constrain"] = constrain_loss

        loss += seed_loss + constrain_loss

        return loss

    def get_balanced_seed_loss(self,softmax,cues):
        count_bg = tf.reduce_sum(cues[:,:,:,0:1],axis=(1,2,3),keepdims=True)
        loss_bg = -tf.reduce_mean(tf.reduce_sum(cues[:,:,:,0:1]*tf.log(softmax[:,:,:,0:1]),axis=(1,2,3),keepdims=True)/(count_bg+1e-8))

        count_fg = tf.reduce_sum(cues[:,:,:,1:],axis=(1,2,3),keepdims=True)
        loss_fg = -tf.reduce_mean(tf.reduce_sum(cues[:,:,:,1:]*tf.log(softmax[:,:,:,1:]),axis=(1,2,3),keepdims=True)/(count_fg+1e-8))
        return loss_bg+loss_fg

    def get_constrain_loss(self,softmax,crf):
        probs_smooth = tf.exp(crf)
        loss = tf.reduce_mean(tf.reduce_sum(probs_smooth * tf.log(probs_smooth/(softmax+1e-8)+1e-8), axis=3))
        return loss

    def optimize(self,base_lr,momentum,weight_decay):
        self.loss["norm"] = self.getloss()
        self.loss["l2"] = sum([tf.nn.l2_loss(self.weights[layer][0]) for layer in self.weights])
        self.loss["total"] = self.loss["norm"]+ weight_decay*self.loss["l2"]
        self.net["lr"] = tf.Variable(base_lr, trainable=False)
        opt = tf.train.MomentumOptimizer(self.net["lr"],momentum)
        gradients = opt.compute_gradients(self.loss["total"])
        self.net["accum_gradient"] = []
        self.net["accum_gradient_accum"] = []
        new_gradients = []
        for (g,v) in gradients:
            if g is None: continue
            if v in self.lr_2_list:
                g = 2*g
            if v in self.lr_10_list:
                g = 10*g
            if v in self.lr_20_list:
                g = 20*g
            self.net["accum_gradient"].append(tf.Variable(tf.zeros_like(g),trainable=False))
            self.net["accum_gradient_accum"].append(self.net["accum_gradient"][-1].assign_add( g/self.accum_num, use_locking=True))
            new_gradients.append((self.net["accum_gradient"][-1],v))

        self.net["accum_gradient_clean"] = [g.assign(tf.zeros_like(g)) for g in self.net["accum_gradient"]]
        self.net["accum_gradient_update"]  = opt.apply_gradients(new_gradients)

    def train(self,base_lr,weight_decay,momentum,batch_size,epoches):
        self.sess = tf.Session()
        data_x,data_y,data_tag,data_cues,id_of_image,iterator_train = self.data.next_batch(category="train",batch_size=batch_size,epoches=-1)
        self.build(net_input=data_x,net_label=data_y,net_tag=data_tag,net_cues=data_cues)
        self.optimize(base_lr,momentum,weight_decay)
        saver = tf.train.Saver(max_to_keep=2,var_list=self.trainable_list)

        with self.sess.as_default():
            self.sess.run(tf.global_variables_initializer())
            self.sess.run(tf.local_variables_initializer())
            self.sess.run(iterator_train.initializer)

            start_time = time.time()
            print("start_time: %f" % start_time)
            print("config -- lr:%f weight_decay:%f momentum:%f batch_size:%f epoches:%f" % (base_lr,weight_decay,momentum,batch_size,epoches))

            epoch,i = 0.0,0
            iterations_per_epoch_train = 10582 // batch_size
            lowest_loss = 2.0
            while epoch < epoches:
                if i == 0: # to protect restore
                    self.sess.run(tf.assign(self.net["lr"],base_lr))
                    self.sess.run(self.net["accum_gradient_clean"])
                if i == 8*iterations_per_epoch_train:
                    new_lr = 1e-4
                    print("save model before new_lr:%f" % new_lr)
                    self.saver["lr"].save(self.sess,os.path.join(self.config.get("saver_path","saver"),"lr-%f" % base_lr),global_step=i)
                    self.sess.run(tf.assign(self.net["lr"],new_lr))
                    base_lr = new_lr
                if i == 16*iterations_per_epoch_train:
                    new_lr = 1e-5
                    print("save model before new_lr:%f" % new_lr)
                    self.saver["lr"].save(self.sess,os.path.join(self.config.get("saver_path","saver"),"lr-%f" % base_lr),global_step=i)
                    self.sess.run(tf.assign(self.net["lr"],new_lr))
                    base_lr = new_lr

                self.sess.run(self.net["accum_gradient_accum"])
                if i % self.accum_num == self.accum_num - 1:
                    _ = self.sess.run(self.net["accum_gradient_update"])
                    _ = self.sess.run(self.net["accum_gradient_clean"])

                if i%500 == 0:
                    seed_l,constrain_l,loss,lr = self.sess.run([self.loss["seed"],self.loss["constrain"],self.loss["total"],self.net["lr"]])
                    print("epoch:%f, iteration:%f, lr:%f, loss:%f" % (epoch,i,lr,loss))
                    print("seed_loss:%f,constrain_loss:%f" % (seed_l,constrain_l))

                i+=1
                epoch = i / iterations_per_epoch_train
                epoch += 300


            end_time = time.time()
            self.pool.close()
            self.pool.join()
            print("end_time:%f" % end_time)
            print("duration time:%f" %  (end_time-start_time))
            saver.save(self.sess,os.path.join(self.config.get("saver_path","saver"),"final"),global_step=0)

if __name__ == "__main__":
    batch_size = 4 # the actual batch size is  batch_size * accum_num
    input_size = (321,321)
    category_num = 21
    epoches = 32
    data = dataset({"batch_size":batch_size,"input_size":input_size,"epoches":epoches,"category_num":category_num,"categorys":["train"]})
    dsrg = DSRG({"data":data,"batch_size":batch_size,"input_size":input_size,"epoches":epoches,"category_num":category_num,"init_model_path":"./model/vgg16_deeplab_aspp.npy","accum_num":16})

    lr = 1e-3
    dsrg.train(base_lr=lr,weight_decay=5e-4,momentum=0.9,batch_size=batch_size,epoches=epoches)
