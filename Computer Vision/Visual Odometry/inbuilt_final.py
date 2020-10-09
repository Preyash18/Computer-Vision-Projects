import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from ReadCameraModel import ReadCameraModel
from UndistortImage import UndistortImage


def image_process(img):
    color_image = cv2.cvtColor(img, cv2.COLOR_BayerGR2BGR)
    color_image_undistorted = UndistortImage(color_image, LUT)
    gray_image = cv2.cvtColor(color_image_undistorted, cv2.COLOR_BGR2GRAY)
    return gray_image


sift = cv2.xfeatures2d.SIFT_create()
path = "stereo/centre"

images = []
for image in os.listdir(path):
    images.append(image)

fx, fy, cx, cy, G_camera_image, LUT = ReadCameraModel('model/')
K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

init = np.identity(4)
origin = np.array([[0, 0, 0, 1]]).T

trajectory_points = []

for ind in range(19, len(images) - 1):
    first_image = cv2.imread("%s\\%s" % (path, images[ind]), 0)
    first_image_processed = image_process(first_image)

    next_image = cv2.imread("%s\\%s" % (path, images[ind + 1]), 0)
    next_image_processed = image_process(next_image)

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(first_image_processed, None)
    kp2, des2 = sift.detectAndCompute(next_image_processed, None)

    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    features_first = []
    features_next = []
    good_features = []

    for i, (m, n) in enumerate(matches):
        if m.distance < 1 * n.distance:
            features_first.append(kp1[m.queryIdx].pt)
            features_next.append(kp2[m.trainIdx].pt)
            good_features.append(m)

    features_first = np.int32(features_first)
    features_next = np.int32(features_next)
    F, mask = cv2.findFundamentalMat(features_first, features_next, cv2.FM_RANSAC)

    features_first = features_first[mask.ravel() == 1]
    features_next = features_next[mask.ravel() == 1]

    E = K.T @ F @ K
    value, rot_final, trans_final, mask = cv2.recoverPose(E, features_first, features_next, K)

    z = np.column_stack((rot_final, trans_final))
    a = np.array([0, 0, 0, 1])
    h2 = np.vstack((z, a))
    init = init @ h2
    p = init @ origin

    plt.scatter(p[0][0], -p[2][0], color='r')
    trajectory_points.append([p[0][0], -p[2][0]])
    print(ind)

plt.show()