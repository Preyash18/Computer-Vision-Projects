- libraries used in the project:
	import cv2
	import numpy as np
	import matplotlib.pyplot as plt
	import os
	import math
	import random
	import numpy as np
	from ReadCameraModel import ReadCameraModel
	from UndistortImage import UndistortImage

- we have used sift detector in our code to extract and match keypoints in adjacent images.
  in order for sift to work, an older version of cv2 should be installed as sift doesn't seem to work in the latest version of cv2.

- the python file for original expected pipeline (in which opencv inbuilt function are not used) is "user_defined_final.py"

- the python file which uses inbuilt open cv functions is "inbuilt_final.py" (for extra credits).

- In order for the code to run properly, both the files "ReadCameraModel.py" and "UndistortImage.py" are to kept in the same folder as the main python files.

- Also, in the file "user_defined_final.py", change the path of the dataset in line 11 and change the path for model folder in the line 17.

- Similarly, for the file "inbuilt_final.py", change the path of the dataset in line 17 and change the path for model folder in the line 23.

- It takes around 2 hours for the file "user_defined_final.py" to produce output and it takes more than 3 hours for the file "inbuilt_final.py" to produce output.

- While the code is running, the index of the image being processed is printed in the output terminal.