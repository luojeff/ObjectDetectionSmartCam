import cv2
import os, sys

from darkflow.defaults import argHandler
from darkflow.net.build import TFNet
from darkflow.net import flow

"""
Darkflow implementation of image recognition ported into a class
using OpenCV
"""
class opencv_darkflow:

    def __init__(self, dir = '.', confidence = 0.5, gpu = 0.0):
        self.GPU = gpu
        self.IMG_DIR = dir
        self.FG = argHandler()
        
        self.construct_tfnet(confidence)

    def construct_tfnet(self, confidence):
        self.FG.setDefaults()

        args = ['flow', '--imgdir', self.IMG_DIR, '--threshold', str(confidence), '--model', 'cfg/yolo.cfg'
        , '--load', 'bin/yolo.weights', '--gpu', str(self.GPU)]

        self.FG.parseArgs(args)
    
        requiredDirectories = [self.FG.imgdir, self.FG.binary, self.FG.backup, os.path.join(self.FG.imgdir,'out')]
        if self.FG.summary: 
            requiredDirectories.append(self.FG.summary)

        for d in requiredDirectories:
            this = os.path.abspath(os.path.join(os.path.curdir, d))
            if not os.path.exists(this): os.makedirs(this)
        
        try: self.FG.load = int(self.FG.load) 
        except: pass

        # print("Creating layered network...")
        sys.stdout = open(os.devnull, 'w') # Block stdout print
        self.tfnet = TFNet(self.FG)
        sys.stdout = sys.__stdout__ # Enable stdout print
        # print("Done")

    # Reformat bounding box array to only return coordinates
    def bb_reformat(self, arr):
        ret_arr = []

        for bb in arr:
            (x1, y1, x2, y2) = (bb['topleft']['x'], bb['topleft']['y'], 
            bb['bottomright']['x'], bb['bottomright']['y'])

            ret_arr.append((x1, y1, x2, y2))

        return ret_arr

    def detect(self, img_file):
        frame = cv2.imread(self.IMG_DIR + '/' + img_file)
        return self.bb_reformat(self.tfnet.return_predict(frame))