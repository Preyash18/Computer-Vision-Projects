import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
import os
import math
from ReadCameraModel import ReadCameraModel
from UndistortImage import UndistortImage

images = []
path = "stereo/centre"

for image in os.listdir(path):
    images.append(image)
    images.sort()

fx, fy, cx, cy, G_camera_image, LUT = ReadCameraModel('model/')
K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

init = np.identity(4)
origin = np.array([[0, 0, 0, 1]]).T

trajectory_points = []

for ind in range(19, len(images) - 1):
    print(ind)
    img1 = cv2.imread(path + str(images[ind]), 0)
    first_image = cv2.cvtColor(img1, cv2.COLOR_BayerGR2BGR)
    first_image_undistorted = UndistortImage(first_image, LUT)
    first_image_gray = cv2.cvtColor(first_image_undistorted, cv2.COLOR_BGR2GRAY)

    # Reading next image
    img2 = cv2.imread(path + str(images[ind + 1]), 0)
    next_image = cv2.cvtColor(img2, cv2.COLOR_BayerGR2BGR)
    next_image_undistorted = UndistortImage(next_image, LUT)
    next_image_gray = cv2.cvtColor(next_image_undistorted, cv2.COLOR_BGR2GRAY)

    # Cropping the area of interest from the current image
    first_image_final = first_image_gray[200:650, 0:1280]
    # Cropping the area of interest from the next frame
    next_image_final = next_image_gray[200:650, 0:1280]

    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT in current as well as next frame
    kp1, des1 = sift.detectAndCompute(first_image_final, None)
    kp2, des2 = sift.detectAndCompute(next_image_final, None)
    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    features_first = []
    features_next = []

    # Ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.5 * n.distance:
            features_first.append(kp1[m.queryIdx].pt)
            features_next.append(kp2[m.trainIdx].pt)

    number_of_inliers = 0
    final_Fund_Matrix = np.zeros((3, 3))

    inliers_first = []
    inliers_next = []
    # RANSAC Algorithm
    for i in range(0, 50):
        count = 0

        eight_points = []

        while (True):
            num = random.randint(0, len(features_first) - 1)
            if num not in eight_points:
                eight_points.append(num)
            if len(eight_points) == 8:
                break

        good_features_first = []
        good_features_next = []
        for point in eight_points:
            good_features_first.append([features_first[point][0], features_first[point][1]])
            good_features_next.append([features_next[point][0], features_next[point][1]])

        # Computing Fundamental Matrix from current frame to next frame
        A = np.empty((8, 9))
        for i in range(0, len(good_features_first)):
            x1 = good_features_first[i][0]
            y1 = good_features_first[i][1]
            x2 = good_features_next[i][0]
            y2 = good_features_next[i][1]
            A[i] = np.array([x1 * x2, x2 * y1, x2, y2 * x1, y2 * y1, y2, x1, y1, 1])

        u, s, v = np.linalg.svd(A, full_matrices=True)
        f = v[-1].reshape(3, 3)

        u1, s1, v1 = np.linalg.svd(f)
        # Constraining Fundamental Matrix to Rank 2
        s2 = np.array([[s1[0], 0, 0], [0, s1[1], 0], [0, 0, 0]])
        FundMatrix = u1 @ s2 @ v1

        temp_features_first = []
        temp_features_next = []
        for number in range(0, len(features_first)):
            x11 = np.array([features_first[number][0], features_first[number][1], 1]).T
            x22 = np.array([features_next[number][0], features_next[number][1], 1])
            dist = abs(np.squeeze(np.matmul((np.matmul(x22, FundMatrix)), x11)))
            # defining threshold as 0.01
            if dist < 0.01:
                count += 1
                temp_features_first.append(features_first[number])
                temp_features_next.append(features_next[number])
        if count > number_of_inliers:
            number_of_inliers = count
            final_Fund_Matrix = FundMatrix
            inliers_first = temp_features_first
            inliers_next = temp_features_next

    # Computing Essential Matrix from current frame to next frame
    tempMatrix = np.matmul(np.matmul(K.T, final_Fund_Matrix), K)
    u, s_, v = np.linalg.svd(tempMatrix, full_matrices=True)
    sigmaF = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])  # Constraining Eigenvalues to 1, 1, 0
    temp = np.matmul(u, sigmaF)
    E_matrix = np.matmul(temp, v)
    essentialMatrix = E_matrix

    # Computing all four solutions of rotation matrix and translation vector
    u, s, v = np.linalg.svd(essentialMatrix, full_matrices=True)
    w = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])

    # Solution 1
    c1 = u[:, 2]
    r1 = u @ w @ v
    if np.linalg.det(r1) < 0:
        c1 = -c1
        r1 = -r1
    c1 = c1.reshape((3, 1))

    # Solution 2
    c2 = -u[:, 2]
    r2 = u @ w @ v
    if np.linalg.det(r2) < 0:
        c2 = -c2
        r2 = -r2
    c2 = c2.reshape((3, 1))

    # Solution 3
    c3 = u[:, 2]
    r3 = u @ w.T @ v
    if np.linalg.det(r3) < 0:
        c3 = -c3
        r3 = -r3
    c3 = c3.reshape((3, 1))

    # Solution 4
    c4 = -u[:, 2]
    r4 = u @ w.T @ v
    if np.linalg.det(r4) < 0:
        c4 = -c4
        r4 = -r4
    c4 = c4.reshape((3, 1))

    rot_list, trans_list = [r1, r2, r3, r4], [c1, c2, c3, c4]
    # Finalising one solution from four
    check = 0
    init_ = np.identity(4)
    for index in range(0, len(rot_list)):
        sy = math.sqrt(rot_list[index][0, 0] * rot_list[index][0, 0] + rot_list[index][1, 0] * rot_list[index][1, 0])
        singular = sy < 1e-6
        if not singular:
            x = math.atan2(rot_list[index][2, 1], rot_list[index][2, 2])
            y = math.atan2(-rot_list[index][2, 0], sy)
            z = math.atan2(rot_list[index][1, 0], rot_list[index][0, 0])
        else:
            x = math.atan2(-rot_list[index][1, 2], rot_list[index][1, 1])
            y = math.atan2(-rot_list[index][2, 0], sy)
            z = 0
        deg_x = x * 180 / math.pi
        deg_y = y * 180 / math.pi
        deg_z = z * 180 / math.pi
        angles = np.array([deg_x, deg_y, deg_z])

        if -50 < angles[0] < 50 and -50 < angles[2] < 50:
            count = 0
            new_pose = np.hstack((rot_list[index], trans_list[index]))  # New camera Pose
            for i in range(0, len(inliers_first)):
                x_o = np.array([[0, -1, inliers_first[i][1]], [1, 0, -inliers_first[i][0]], [-inliers_first[i][1], inliers_first[i][0], 0]])
                x_o_d = np.array([[0, -1, inliers_next[i][1]], [1, 0, -inliers_next[i][0]], [-inliers_next[i][1], inliers_next[i][0], 0]])

                A1 = x_o @ init_[0:3, :][0:3, :]
                A2 = x_o_d @ new_pose
                A = np.vstack((A1, A2))

                u, s, v = np.linalg.svd(A)
                new1X = v[-1]
                new1X = new1X / new1X[3]
                new1X = new1X.reshape((4, 1))

                temp1x = new1X[0:3].reshape((3, 1))
                third_row = rot_list[index][2, :].reshape((1, 3))
                if np.squeeze(third_row @ (temp1x - trans_list[index])) > 0:
                    count = count + 1

            if count > check:
                check = count
                trans = trans_list[index]
                rot = rot_list[index]

    if trans[2] > 0:
        trans = -trans

    rot_final, trans_final = rot, trans
    z = np.column_stack((rot_final, trans_final))
    a = np.array([0, 0, 0, 1])
    z = np.vstack((z, a))
    # Transforming from current frame to next frame
    init = init @ z
    # Determining the transformation of the origin from current frame to next frame
    p = init @ origin

    trajectory_points.append([p[0][0], -p[2][0]])
    plt.scatter(p[0][0], -p[2][0], color='r')

plt.show()
