from adaptive_hist import bhattacharyya
import cv2
import numpy as np
import sys


import os

# Suppress Caffe output
os.environ['GLOG_minloglevel'] = '2'

import caffe

# Test class to extract CNN features from two sample input images
# and calculates the bhattacharyya given two arbitrary vector inputs and
# the detected input object/vehicle
class googlenet:
    
    def __init__(self, prototxt, model):
        MEAN_PATH = "ilsvrc_2012_mean.npy"
        mean = np.load(MEAN_PATH).mean(1).mean(1)
        channel_swap = (2, 0, 1)
        raw_scale = 255.0
        im_dims = (224, 224)

        self.net = caffe.Net(prototxt, model, caffe.TEST)
        self.transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
        self.transformer.set_mean('data', mean)
        self.transformer.set_transpose('data', channel_swap)
        self.transformer.set_raw_scale('data', raw_scale)

        caffe.set_mode_cpu()

    # Returns vector of detected object within image
    def vectorize(self, image_path):
        
        # Selected layer to extract feature from as defined in prototxt
        LAYER = 'pool5'
        
        image = caffe.io.load_image(image_path)
        self.net.blobs['data'].data[0] = self.transformer.preprocess('data', image)
        self.net.forward()

        # Feature vector extracted from layer
        return self.net.blobs[LAYER].data[0].copy()

    def normalize(self, vec):
        return vec / sum(vec)

    def analyze(self, img_A_path, img_B_path):
        img_A_vec = self.normalize(self.vectorize(img_A_path))
        img_B_vec = self.normalize(self.vectorize(img_B_path))
        return bhattacharyya(img_A_vec, img_B_vec)
