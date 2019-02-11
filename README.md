## Object Detection in Video Testing
This repository stores documentation and files necessary for running/testing the Tensorflow (Darkflow) implementation of YOLOv2's object detection software. The testing shown is run on a specific hardware setup as described in the section below. 

### Steps to Obtain Bounding Boxes
Contained in this repository is `opencv_darkflow.py`. This file contains the class that allows for interaction with Darkflow's bounding box creation software using Python. An example instantiation of an object is shown:
```python
obj_det = opencv_darkflow(dir = 'sample_img_2', confidence = 0.6, gpu = 0.0)
print(obj_det.detect('sample.jpg'))
```
This prints the respective bounding boxes for the objects detected in the file `sample_img_2/sample.jpg` with confidence of greater than or equal to 60%, while utilizing only the CPU for performance. 

### Steps to Test
*Windows 10*
1. Clone the Darkflow repository as provided in the image/video classifier section
2. Install OpenCV for Python
3. Install TensorFlow (much preferably a CPU or GPU-optimized build)
4. Create a batch file `run_video.bat`with the following command:
```
python flow --model cfg/yolo.cfg --threshold 0.4 --load bin/yolo.weights --demo <VideoName>.avi --gpu 0.0 --saveVideo
```
Replace <VideoName> with the designated video file on which you want to run the object classifier. The threshold 0.4 specifies that only detected objects with a confidence of >= 40% will appear in bounding boxes. If you are using a GPU, then it is recommended to change the GPU parameter to ~0.8. 

5. Run the batch file using `$ run_video.bat`. This will create a resulting video `video.avi` (this may take a while when running only on the CPU). 

6. To only allow for certain object labels to appear (like "car" or "bus"), access the file `darkflow\darkflow\net\yolov2\predict.py` and add the following lines of code: 
```python
...
boxResults = self.process_box(b, h, w, threshold)
if boxResults is None:
	continue
left, right, top, bot, mess, max_indx, confidence = boxResults

### START OF INSERTION
if max_indx not in [2, 3, 5, 7]: 
	continue
### END OF INSERTION

thick = int((h + w) // 300)
if self.FLAGS.json:
...
```
Then re-run the batch file. 

### Personal Hardware Setup
CPU: Intel i7-8550U CPU (1.80 GHz) \
GPU (Integrated): Intel UHD Graphics 620 \
Operating System: **Windows 10**

### Image/Video Classifier
Darkflow implementation of YOLOv2. This can be accessed at [https://github.com/thtrieu/darkflow](https://github.com/thtrieu/darkflow).

### Necessary Software Installation
Wheel files for my hardware configuration are provided for the following libraries:
- Python OpenCV
- TensorFlow

For my personal hardware setup, I am running TensorFlow completely off the CPU. As such, it is recommended to build a CPU-optimized version of TensorFlow that support specific processor architecture operations (AVX, AVX2) to attain better performance. A list of pre-built binaries can be found at [https://github.com/fo40225/tensorflow-windows-wheel](https://github.com/fo40225/tensorflow-windows-wheel).

### Results
*NOTE*: There results are specific to my hardware and software configurations.

### Helpful Resources
Visit [http://130.207.122.57:8000/](http://130.207.122.57:8000/) to obtain lab resources. \
To conduct training on custom data sets, check out DarkFlow's [documentation]([https://github.com/thtrieu/darkflow](https://github.com/thtrieu/darkflow)).

### Authors
Jeffrey Luo