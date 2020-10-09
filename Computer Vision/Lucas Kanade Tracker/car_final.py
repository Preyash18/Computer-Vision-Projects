import numpy as np
import cv2
import copy
from scipy.interpolate import RectBivariateSpline
import os

images = []
dataset_images = []
path = 'Car4/img/'

for image in os.listdir(path):
    images.append(image)
images.sort()

for image in images:
    img = cv2.imread("%s%s" % (path, image))
    dataset_images.append(img)


def image_process(y1, x1, y2, x2, itr):
    frame_coordinates = [y1, x1, y2, x2]
    height = frame_coordinates[3] - frame_coordinates[1]
    length = frame_coordinates[2] - frame_coordinates[0]
    frame_coordinates_copy = copy.deepcopy(frame_coordinates)
    initial_frame = dataset_images[itr]
    initial_frame_gray = cv2.cvtColor(initial_frame, cv2.COLOR_BGR2GRAY)
    processed_init_frame = cv2.equalizeHist(initial_frame_gray)

    return height, length, frame_coordinates, frame_coordinates_copy, processed_init_frame


img_list = []

width, length, frame_points, frame_points_copy, hist_init_frame = image_process(71, 50, 175, 139, 0)


def LucasKanade(init_fr, next_fr, frame_points_copy, p_list=np.zeros(2)):
    threshold = 0.001
    x1, y1, x2, y2 = frame_points_copy[0], frame_points_copy[1], frame_points_copy[2], frame_points_copy[3]
    initial_y, initial_x = np.gradient(next_fr)
    # variable for storing error in the parameters
    err = 1

    while np.square(err).sum() > threshold:
        # Initial Parameters
        list_x, list_y = p_list[0], p_list[1]
        # Warped Parameters
        x1_warp = x1 + list_x
        y1_warp = y1 + list_y
        x2_warp = x2 + list_x
        y2_warp = y2 + list_y

        x = np.arange(0, init_fr.shape[0], 1)
        y = np.arange(0, init_fr.shape[1], 1)

        a = np.linspace(x1, x2, 87)
        b = np.linspace(y1, y2, 36)
        mesh_a, mesh_b = np.meshgrid(a, b)

        a_warp = np.linspace(x1_warp, x2_warp, 87)
        b_warp = np.linspace(y1_warp, y2_warp, 36)
        mesh_a_warp, mesh_b_warp = np.meshgrid(a_warp, b_warp)

        spline = RectBivariateSpline(x, y, init_fr)
        T = spline.ev(mesh_b, mesh_a)

        spline1 = RectBivariateSpline(x, y, next_fr)
        warpImg = spline1.ev(mesh_b_warp, mesh_a_warp)

        error = T - warpImg
        error_img = error.reshape(-1, 1)

        spline_gx = RectBivariateSpline(x, y, initial_x)
        initial_x_warp = spline_gx.ev(mesh_b_warp, mesh_a_warp)

        spline_gy = RectBivariateSpline(x, y, initial_y)
        initial_y_warp = spline_gy.ev(mesh_b_warp, mesh_a_warp)

        I = np.vstack((initial_x_warp.ravel(), initial_y_warp.ravel())).T

        jacobian_matrix = np.array([[1, 0], [0, 1]])

        hessian = I @ jacobian_matrix
        H = hessian.T @ hessian

        err = np.linalg.inv(H) @ hessian.T @ error_img
        p_list[0] += err[0, 0]
        p_list[1] += err[1, 0]

    final = p_list
    return final


for i in range(0, len(dataset_images) - 1):

    if i == 100:
        width, length, frame_points, frame_points_copy, hist_init_frame = image_process(69, 58, 169, 135, 100)

    if i == 150:
        width, length, frame_points, frame_points_copy, hist_init_frame = image_process(94, 57, 186, 123, 150)

    if i == 200:
        width, length, frame_points, frame_points_copy, hist_init_frame = image_process(134, 69, 215, 125, 200)

    frame = dataset_images[i]
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hist_frame = cv2.equalizeHist(gray_frame)
    # Drawing rectangle over image
    cv2.rectangle(frame, (int(frame_points[0]), int(frame_points[1])),
                  (int(frame_points[0]) + length, int(frame_points[1]) + width), (0, 255, 0), 3)
    img_list.append(frame)
    # displaying output
    cv2.imshow('Car_Tracking', frame)

    next_frame = dataset_images[i + 1]
    gray_next_frame = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
    hist_next_frame = cv2.equalizeHist(gray_next_frame)

    init_fr = hist_init_frame / 255.
    next_fr = hist_next_frame / 255.

    p = LucasKanade(init_fr, next_fr, frame_points_copy)

    frame_points[0] = frame_points_copy[0] + p[0]
    frame_points[1] = frame_points_copy[1] + p[1]
    frame_points[2] = frame_points_copy[2] + p[0]
    frame_points[3] = frame_points_copy[3] + p[1]

    if cv2.waitKey(1) == 27:
        cv2.destroyAllWindows()
        break

# cv2.imwrite('first_frame_car.jpg', img_list[0])

out = cv2.VideoWriter('car_final.avi', cv2.VideoWriter_fourcc(*'XVID'), 5.0, (360, 240))
for image in img_list:
    out.write(image)
    cv2.waitKey(10)

out.release()
