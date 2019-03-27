# Extract adaptive histogram for feature description.
# See http://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w3/Tang_Single-Camera_and_Inter-Camera_CVPR_2018_paper.pdf for detail.

from imutils.video import FPS
from skimage.feature import hog
import numpy as np
import math
import cv2
import argparse
import imutils

def bhattacharyya(a, b):
    if not len(a) == len(b):
        raise ValueError("a and b must be of the same size")
    return -math.log(sum((math.sqrt(u * w) for u, w in zip(a, b))))

def adaptive_hist(image):
    """
    image is opencv image format. I.e. image = cv2.imread(path).
    """
    mask = np.zeros(image.shape[:2], np.uint8)
    # spatially weighted by Gaussian distribtuion?
    mask = cv2.ellipse(mask, (image.shape[1] // 2,image.shape[0] // 2),
                       (image.shape[1] // 2,image.shape[0] // 2), 0, 0, 360, 255, -1)

    # RGB color histogram
    hist1 = cv2.calcHist([image], [0], mask, [16], [0, 256]).reshape(1, -1)
    hist2 = cv2.calcHist([image], [1], mask, [16], [0, 256]).reshape(1, -1)
    hist3 = cv2.calcHist([image], [2], mask, [16], [0, 256]).reshape(1, -1)
    rgb_hist = np.concatenate((hist1, hist2, hist3), axis=1)
    cv2.normalize(rgb_hist, rgb_hist)

    # HSV color histogram
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist1 = cv2.calcHist([img_hsv], [0], mask, [16], [0, 256]).reshape(1, -1)
    hist2 = cv2.calcHist([img_hsv], [1], mask, [16], [0, 256]).reshape(1, -1)
    hsv_hist = np.concatenate((hist1, hist2), axis=1)
    cv2.normalize(hsv_hist, hsv_hist)

    # YCrCb color histogram
    img_YCrCb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    hist1 = cv2.calcHist([img_YCrCb], [1], mask, [16], [0, 256]).reshape(1, -1)
    hist2 = cv2.calcHist([img_YCrCb], [2], mask, [16], [0, 256]).reshape(1, -1)
    YCrCb_hist = np.concatenate((hist1, hist2), axis=1)
    cv2.normalize(YCrCb_hist, YCrCb_hist)

    # Lab color histogram
    img_lab = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
    hist1 = cv2.calcHist([img_lab], [1], mask, [16], [0, 256]).reshape(1, -1)
    hist2 = cv2.calcHist([img_lab], [2], mask, [16], [0, 256]).reshape(1, -1)
    lab_hist = np.concatenate((hist1, hist2), axis=1)
    cv2.normalize(lab_hist, lab_hist)

    # Hog
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_gray = cv2.resize(image_gray, (200,200))
    hog_hist = hog(image_gray, orientations=8, pixels_per_cell=(50,50), cells_per_block=(1,1), visualise=False).reshape(1, -1)
    cv2.normalize(hog_hist, hog_hist)

    # type?
    #type_hist = np.zeros(8).reshape(1,8) + 0.5
    #type_hist[0, int(image_path[-5])] = 1
    #cv2.normalize(type_hist, type_hist)

    return np.concatenate((3 * rgb_hist, hsv_hist, YCrCb_hist, lab_hist), axis=1)[0]

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--prototxt", required=True,
                    help="path to Caffe 'deploy' prototxt file")
    ap.add_argument("-m", "--model", required=True,
                    help="path to Caffe pre-trained model")
    ap.add_argument("-v", "--video", required=True,
                    help="path to input video file")
    ap.add_argument("-l", "--label", required=True,
                    help="class label we are interested in detecting + tracking")
    ap.add_argument("-o", "--output", type=str,
                    help="path to optional output video file")
    ap.add_argument("-c", "--confidence", type=float, default=0.2,
                    help="minimum probability to filter weak detections")
    ap.add_argument("-s", "--skip-frames", type=int, default=30,
                    help="# of skip frames between detections")
    args = vars(ap.parse_args())

    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
               "sofa", "train", "tvmonitor"]

    print("[INFO] loading model...")
    net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

    print("[INFO] starting video stream...")
    vs = cv2.VideoCapture(args["video"])

    writer = None
    totalFrames = 0
    lasthist = None

    fps = FPS().start()

    while True:
        (grabbed, frame) = vs.read()

        if frame is None:
            break

        frame = imutils.resize(frame, width=600)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if args["output"] is not None and writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(args["output"], fourcc, 30,
                                     (frame.shape[1], frame.shape[0]), True)

        if totalFrames % args["skip_frames"] == 0:
            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(frame, 0.007843, (w, h), 127.5)

            net.setInput(blob)
            detections = net.forward()

            for i in np.arange(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > args["confidence"]:
                    label = CLASSES[int(detections[0, 0, i, 1])]

                    if label != args["label"]:
                        continue

                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    crop_image = rgb[startY:endY, startX:endX]
                    cv2.imshow("vehicle", cv2.resize(crop_image, (0,0), fx=5, fy=5))

                    adhist = adaptive_hist(crop_image)
                    if lasthist is not None:
                        print(bhattacharyya(lasthist / sum(lasthist), adhist / sum(adhist)))
                        lasthist = adhist
                        cv2.waitKey(0)

        if writer is not None:
            writer.write(frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        totalFrames += 1
        fps.update()

    fps.stop()
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    if writer is not None:
        writer.release()

    cv2.destroyAllWindows()
    vs.release()
