import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import math
from imutils import contours

# taking input from user for buoy selection
print("Choose the buoy color: \n Press 1 for Orange, Press 2 for Yellow, Press 3 for Green \n")
ent = int(input("Your input: "))
if ent == 1:
    path = 'C:/UMD/Spring2020/673-Perception for autonomous/Project3_me/orange_train_trial'
elif ent == 2:
    path = 'C:/UMD/Spring2020/673-Perception for autonomous/Project3_me/yellow_train_trial'
elif ent == 3:
    path = 'C:/Users/p1889/OneDrive/Desktop/try'
else:
    print("Invalid Selection. Try again!")
    exit(0)

blue_histogram = np.zeros((256, 1))
green_histogram = np.zeros((256, 1))
red_histogram = np.zeros((256, 1))

for every_image in os.listdir(path):
    # taking images from training data folder
    image = cv2.imread(os.path.join(path, every_image))
    if ent == 1 or ent == 2:
        # resizing image for faster processing
        resized_image = cv2.resize(image, (40, 40), interpolation=cv2.INTER_LINEAR)
        # cropping image for best results
        cropped_image = resized_image[13:27, 13:27]
        # smoothing image using gaussian blur
        ready_image = cv2.GaussianBlur(cropped_image, (3, 3), 0)
    elif ent == 3:
        ready_image = cv2.GaussianBlur(image, (5, 5), 0)

    color = ("b", "g", "r")
    for i, colour in enumerate(color):

        # calculating histogram for blue channel
        if colour == 'b':
            blue_hist = cv2.calcHist([ready_image], [i], None, [256], [0, 256])
            blue_histogram = np.column_stack((blue_histogram, blue_hist))

        # calculating histogram for green channel
        if colour == 'g':
            green_hist = cv2.calcHist([ready_image], [i], None, [256], [0, 256])
            green_histogram = np.column_stack((green_histogram, green_hist))

        # calculating histogram for red channel
        if colour == 'r':
            red_hist = cv2.calcHist([ready_image], [i], None, [256], [0, 256])
            red_histogram = np.column_stack((red_histogram, red_hist))

total_r = red_histogram.shape[1] - 1
total_g = green_histogram.shape[1] - 1
total_b = blue_histogram.shape[1] - 1

# calculating average histogram for all 3 sets
average_hist_red = np.sum(red_histogram, axis=1) / total_r
average_hist_green = np.sum(green_histogram, axis=1) / total_g
average_hist_blue = np.sum(blue_histogram, axis=1) / total_b

#plt.plot(average_hist_green, color='g')
#plt.plot(average_hist_red, color='r')
#plt.show()
#plt.plot(average_hist_green, color='g')
#plt.show()
plt.plot(average_hist_blue, color='b')
plt.show()

# calculating mean and standard deviation for the final image
mean, std_dev = cv2.meanStdDev(ready_image)
print("Mean", mean)
print("Standard Deviation", std_dev)

temp_list = list(range(0, 256))

# gaussian for all three sets
gaussian_blue = (1 / (std_dev[0] * math.sqrt(2 * math.pi))) * np.exp(-np.power(temp_list - mean[0], 2.) /
                                                                     (2 * (std_dev[0] ** 2)))
gaussian_green = (1 / (std_dev[1] * math.sqrt(2 * math.pi))) * np.exp(-np.power(temp_list - mean[1], 2.) /
                                                                      (2 * (std_dev[1] ** 2)))
gaussian_red = (1 / (std_dev[2] * math.sqrt(2 * math.pi))) * np.exp(-np.power(temp_list - mean[2], 2.) /
                                                                    (2 * (std_dev[2] ** 2)))

plt.plot(gaussian_blue, 'b')
# print(max(gaussian_blue))
# plt.show()

plt.plot(gaussian_green, 'g')
# print(max(gaussian_green))
# plt.show()

plt.plot(gaussian_red, 'r')
# print(max(gaussian_red))
# plt.show()

c = cv2.VideoCapture("C:/UMD/Spring2020/673-Perception for autonomous/Project3_me/detectbuoy.avi")
size = (640, 480)
# creating output video file
if ent ==1:
    out = cv2.VideoWriter('FinalOutputOneDOrange.avi', cv2.VideoWriter_fourcc(*'DIVX'), 5, size)
elif ent ==2:
    out = cv2.VideoWriter('FinalOutputOneDYellow.avi', cv2.VideoWriter_fourcc(*'DIVX'), 5, size)
elif ent ==3:
    out = cv2.VideoWriter('FinalOutputOneDGreen.avi', cv2.VideoWriter_fourcc(*'DIVX'), 5, size)

while True:
    ret, every_image = c.read()

    # slicing every frame into 3 channels
    red_channel = every_image[:, :, 2]
    green_channel = every_image[:, :, 1]
    blue_channel = every_image[:, :, 0]
    check = 0
    if ret:
        # initialising output image as blank image of zeroes
        output_frame1 = np.zeros(red_channel.shape, dtype=np.uint8)
        output_frame2 = np.zeros(green_channel.shape, dtype=np.uint8)

        # for red buoy
        if ent == 1:
            for i in range(0, red_channel.shape[0]):
                for j in range(0, red_channel.shape[1]):
                    r_element = red_channel[i][j]

                    if gaussian_red[r_element] > 0.010 and blue_channel[i][j] < 150:
                        output_frame1[i][j] = 255

                    if gaussian_green[r_element] > 0.0070 and blue_channel[i][j] < 150:
                        output_frame1[i][j] = 0

                    if gaussian_blue[r_element] > 0.004 and blue_channel[i][j] < 150:
                        output_frame1[i][j] = 0
            blur = cv2.medianBlur(output_frame1, 3)
            edged = cv2.Canny(blur, 50, 255)
            cv2.imshow("edge", edged)
            contours, h = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            (contours_sorted) = sorted(contours, key=cv2.contourArea, reverse=True)
            try:
                (x, y), radius = cv2.minEnclosingCircle(contours_sorted[0])
                center = (int(x), int(y))
                radius = int(radius)
                if 7 < radius < 25:
                    cv2.circle(every_image, (int(x), int(y)), int(radius), (0, 170, 255), 2)
                    cv2.imshow("Final output", every_image)
                    out.write(every_image)
                else:
                    cv2.imshow("Final output", every_image)
                    out.write(every_image)
            except:
                pass
            cv2.waitKey(30)

        # for yellow buoy
        elif ent == 2:
            for i in range(0, red_channel.shape[0]):
                for j in range(0, red_channel.shape[1]):
                    r_element = red_channel[i][j]
                    g_element = green_channel[i][j]
                    if (gaussian_red[r_element]) > 0.017 and gaussian_red[g_element] > 0.017 and blue_channel[i][j] < 150:
                        output_frame1[i][j] = 255
                    if gaussian_blue[r_element] > 0.001 and gaussian_blue[g_element] > 0.017 and blue_channel[i][j] < 150:
                        output_frame1[i][j] = 255
                    if gaussian_green[r_element] > 0.009 and blue_channel[i][j] < 150:
                        output_frame1[i][j] = 255
            blur = cv2.medianBlur(output_frame1, 3)
            edged = cv2.Canny(blur, 50, 255)
            cv2.imshow("edge", edged)
            contours, h = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            (contours_sorted) = sorted(contours, key=cv2.contourArea, reverse=True)
            try:
                (x, y), radius = cv2.minEnclosingCircle(contours_sorted[0])
                center = (int(x), int(y))
                radius = int(radius)
                if 10 < radius < 35:
                    cv2.circle(every_image, (int(x), int(y)), int(radius), (0, 230, 255), 2)
                    cv2.imshow("Final output", every_image)
                    out.write(every_image)
                else:
                    cv2.imshow("Final output", every_image)
                    out.write(every_image)
            except:
                pass
            cv2.waitKey(30)

        # for green buoy
        elif ent == 3:
            for i in range(0, green_channel.shape[0]):
                for j in range(0, green_channel.shape[1]):
                    g_element = green_channel[i][j]

                    if gaussian_red[g_element] > 0.010 and gaussian_green[g_element] < 0.02 and gaussian_blue[g_element] < 0.0099 and \
                            red_channel[i][j] < 200:
                        output_frame2[i][j] = 255
                    else:
                        output_frame2[i][j] = 0

            ret, threshold = cv2.threshold(output_frame2, 254, 255, cv2.THRESH_BINARY)
            kernel1 = np.ones((2, 2), np.uint8)

            dilation1 = cv2.dilate(threshold, kernel1, iterations=9)
            contours1, _ = cv2.findContours(dilation1, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours1:
                if cv2.contourArea(contour) > 30:
                    (x, y), radius = cv2.minEnclosingCircle(contour)
                    center = (int(x), int(y))
                    radius = int(radius)
                    if 13 < radius < 15.5:
                        cv2.circle(every_image, center, radius, (0, 255, 0), 2)
                        cv2.imshow('Output Image', every_image)
                        cv2.imshow("Threshold", dilation1)
                        out.write(every_image)
                        #out.write(dilation1)

            k = cv2.waitKey(45) & 0xff
            if k == 27:
                break

out.release()
c.release()
cv2.destroyAllWindows()