from adaptive_hist import bhattacharyya
import cv2
import numpy as np
import caffe

# Test class to extract CNN features from two sample input images
# and calculates the bhattacharyya given two arbitrary vector inputs and
# the detected input object/vehicle


class googlenet:
    
    def __init__(self, prototxt, model):
        # self.net = cv2.dnn.readNetFromCaffe(prototxt, model)
        # Set mean path to respective file
        
        MEAN_PATH = "ilsvrc_2012_mean.npy"
        mean = np.load(MEAN_PATH).mean(1).mean(1)
        channel_swap = (2, 1, 0)
        raw_scale = 255

        # Google Net prototxt specified dimension
        im_dims = (224, 224) 
        
        self.classifier = caffe.Classifier(prototxt, model, mean, channel_swap, raw_scale, im_dims);
        caffe.set_mode_cpu()

    # Returns bhattacharya distance between two detected object images
    def bhattacharyya(self, det_A, det_B):
        return bhattacharyya(det_A, det_B)

    # Returns vector of detected object within image
    def vectorize(self, image_path):
        """
        (h, w) = (224, 224)
        scaleFactor = 0.007843
        mean = 255.0 / 2
        swapRB = True
        blob = cv2.dnn.blobFromImage(frame, scaleFactor, (w, h), mean, swapRB)
        
        self.net.setInput(blob)
        pred = self.net.forward()        
        """

        # Selected layer to extract feature from as defined in prototxt
        LAYER = 'pool5'

        # Feature vector extracted from layer        
        image = caffe.io.load_image(image_path)
        classifier.predict([image], oversample=False)
        feature_vec = classifier.blobs[LAYER].data(0).reshape(1, -1)

        return feature_vec;

    def analyze(self, img_A_path, img_B_path):
        # frame_A = cv2.imread(img_A_path)
        # frame_B = cv2.imread(img_B_path)

        return self.bhattacharyya(self.vectorize(img_A_path), self.vectorize(img_B_path))
