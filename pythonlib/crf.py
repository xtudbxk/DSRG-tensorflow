import os
import re
import sys
import glob
import json
import time
import numpy as np 
import skimage
import skimage.io as imgio
import matplotlib.pyplot as plt
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels


def crf_inference(img,crf_config,category_num,feat=None,pred=None,gt_prob=0.7,use_log=False):
    '''
    feat: the feature map of cnn, shape [h,w,c] or pred, shape [h,w], float32
    img: the origin img, shape [h,w,3], uint8
    crf_config: {"g_sxy":3,"g_compat":3,"bi_sxy":5,"bi_srgb":5,"bi_compat":10,"iterations":5}
    '''
    img = img.astype(np.uint8)
    h,w = img.shape[0:2]
    crf = dcrf.DenseCRF2D(w,h,category_num)

    if feat is not None:
        feat = feat.astype(np.float32)
        if use_log is True:
            feat = np.exp(feat -np.max(feat,axis=2,keepdims=True))
            feat /= np.sum(feat,axis=2,keepdims=True)
            unary = -np.log(feat)
        else:
            unary = -feat
        unary = np.reshape(unary,(-1,category_num))
        unary = np.swapaxes(unary,0,1)
        unary = np.copy(unary,order="C")
        crf.setUnaryEnergy(unary)
    else:
        pred = pred.astype(np.float32)
        # unary energy
        unary = unary_from_labels(pred,category_num,gt_prob,zero_unsure=False)
        crf.setUnaryEnergy(unary)

    # pairwise energy
    crf.addPairwiseGaussian( sxy=crf_config["g_sxy"], compat=crf_config["g_compat"] )
    crf.addPairwiseBilateral( sxy=crf_config["bi_sxy"], srgb=crf_config["bi_srgb"], rgbim=img, compat=crf_config["bi_compat"] )
    Q = crf.inference( crf_config["iterations"] )
    Q = np.array(Q)
    Q = np.reshape(Q,[category_num,h,w])
    Q = np.transpose(Q,axes=[1,2,0]) # new shape: [h,w,c]
    return Q
