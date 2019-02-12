import cv2
import os, sys

from darkflow.net.build import TFNet

DARKFLOW_HOME="/home/device/darkflow"

"""
Darkflow implementation of image recognition ported into a class
using OpenCV
"""
class opencv_darkflow:

    def __init__(self, model = 'yolo.cfg', load = 'yolo.weights', confidence = 0.1):
        
        basecfg = os.path.join(DARKFLOW_HOME, 'cfg')
        basebin = os.path.join(DARKFLOW_HOME, 'bin')
        
        options = {'config' : basecfg,
                   'model' : os.path.join(basecfg, model),
                   'load' : os.path.join(basebin, load),
                   'threshold' : confidence}
        
        #sys.stdout = open(os.devnull, 'w') # Block stdout print
        self.tfnet = TFNet(options)
        #sys.stdout = sys.__stdout__ # Enable stdout print

    # Reformat bounding box array to only return coordinates
    def bb_reformat(self, arr, labels):

        ret_arr = []

        for bb in arr:
            
            if bb['label'] in labels:
                
                (x1, y1) = (bb['topleft']['x'], bb['topleft']['y'])
                (x2, y2) = (bb['bottomright']['x'], bb['bottomright']['y'])
            
                ret_arr.append((x1, y1, x2, y2))
			
        return ret_arr

    def detect(self, frame, labels = {'bus', 'car', 'truck'}):
        return self.bb_reformat(self.tfnet.return_predict(frame), labels)
