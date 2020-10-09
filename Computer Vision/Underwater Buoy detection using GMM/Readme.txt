ENPM 673 - Project 3 - Group 2

-Programming Language -Python

- The folders “green_train”, “yellow_train”, “orange_train” and frames used to create the training images are included in the google drive and the folder also.
- Change the path of training data according to the folder. 
- The video link is given in the report. 
- Report is included in the folder

1D Gaussian
- Segmentation of each buoy is asked from the user. There are three videos uploaded. Each video shows segmentation of  diferent buoy.

3D Gaussian
- Each buoy is segmented in a single video.

Importing Modules: The following modules were used.
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import os
from imutils import contours

Python Files Included :-

1. takeimage.py :- Contains code to generate Data set.

2. 1D_final.py :-   Code to detect green buoy using 1D gaussian on green channel.
	             Code to detect yellow buoy using 1D gaussian on green and red channels.
	             Code to detect orange buoy using 1D gaussian on red channel.

3. 3D_final.py :-  Code to detect all buoys using 3D gaussian on all RGB channels.


Videos Included :-

1) 1d_orange.avi :- 1D Gauss-Orange
2)1D_yellow.avi :- 1D Gauss-Yellow
3)1D_green3.avi:- 1D Gauss-Green
4) 3D_Gauss :- 3D Gauss-all buoy