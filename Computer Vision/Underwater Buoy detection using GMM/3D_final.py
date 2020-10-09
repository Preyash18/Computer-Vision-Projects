import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
import os


def calculateProbability(x_co, mean, std):
    return (1 / (std * math.sqrt(2 * math.pi))) * (math.exp(-((x_co - mean) * 2) / (2 * std * 2)))


def em_gmm(path):
    pixels = []
    for filename in os.listdir(path):
        image = cv2.imread(os.path.join(path, filename))
        resized_image = cv2.resize(image, (40, 40), interpolation=cv2.INTER_LINEAR)
        # cropping image for best results
        cropped_image = resized_image[19:21, 19:21]
        # smoothing image using gaussian blur
        # ready_image = cv2.GaussianBlur(cropped_image, (3, 3), 0)
        image = cropped_image[:, :, 2]
        r, c = image.shape
        for i in range(0, r):
            for j in range(0, c):
                px = image[i][j]
                pixels.append(px)

    mean1 = 190
    mean2 = 150
    mean3 = 250

    std_dev1 = 10
    std_dev2 = 10
    std_dev3 = 10

    n = 0
    while n != 50:

        baye1 = []
        baye2 = []
        baye3 = []

        for px in pixels:
            p1 = calculateProbability(px, mean1, std_dev1)
            p2 = calculateProbability(px, mean2, std_dev2)
            p3 = calculateProbability(px, mean3, std_dev3)

            total = (p1 / 3) + (p2 / 3) + (p3 / 3)
            baye1.append((p1 / 3) / total)
            baye2.append((p2 / 3) / total)
            baye3.append((p3 / 3) / total)

        mean1 = np.sum(np.array(baye1) * np.array(pixels)) / np.sum(np.array(baye1))
        mean2 = np.sum(np.array(baye2) * np.array(pixels)) / np.sum(np.array(baye2))
        mean3 = np.sum(np.array(baye3) * np.array(pixels)) / np.sum(np.array(baye3))

        std_dev1 = math.sqrt(np.sum(np.array(baye1) * ((np.array(pixels)) - mean1) ** 2) / np.sum(np.array(baye1)))
        std_dev2 = math.sqrt(np.sum(np.array(baye2) * ((np.array(pixels)) - mean2) ** 2) / np.sum(np.array(baye2)))
        std_dev3 = math.sqrt(np.sum(np.array(baye3) * ((np.array(pixels)) - mean3) ** 2) / np.sum(np.array(baye3)))

        n += 1
    # print('final mean- ', mean1, mean2, mean3)
    # print('final std- ', std_dev1, std_dev2, std_dev3)
    return mean1, std_dev1, mean2, std_dev2, mean3, std_dev3


mb1, sb1, mg1, sg1, mr1, sr1 = em_gmm("C:/UMD/Spring2020/673-Perception for autonomous/Project3_me/orange_train_trial")
mb2, sb2, mg2, sg2, mr2, sr2 = em_gmm("C:/UMD/Spring2020/673-Perception for autonomous/Project3_me/yellow_train_trial")
mb3, sb3, mg3, sg3, mr3, sr3 = em_gmm("C:/Users/p1889/OneDrive/Desktop/try")


def gaussian(mu, sig, x):
    return ((1 / (sig * math.sqrt(2 * math.pi))) * np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.))))


temp_list = list(range(0, 256))

gaussian_blue_1 = gaussian(mb1, sb1, temp_list)      #orange
gaussian_green_1 = gaussian(mg1, sg1, temp_list)
gaussian_red_1 = gaussian(mr1, sr1, temp_list)

gaussian_blue_2 = gaussian(mb2, sb2, temp_list)         #yellow
gaussian_green_2 = gaussian(mg2, sg2, temp_list)
gaussian_red_2 = gaussian(mr2, sr2, temp_list)

gaussian_blue_3 = gaussian(mb3, sb3, temp_list)         #green
gaussian_green_3 = gaussian(mg3, sg3, temp_list)
gaussian_red_3 = gaussian(mr3, sr3, temp_list)

plt.plot(gaussian_blue_1, 'b')
# print(max(gaussian_blue_1))
#plt.show()

plt.plot(gaussian_green_1, 'g')
# print(max(gaussian_green_1))
#plt.show()


plt.plot(gaussian_red_1, 'r')
# print(max(gaussian_red_1))
#plt.show()


plt.plot(gaussian_blue_2, 'b')
# print(max(gaussian_blue_2))
#plt.show()

plt.plot(gaussian_green_2, 'g')
# print(max(gaussian_green_2))
#plt.show()


plt.plot(gaussian_red_2, 'r')
# print(max(gaussian_red_2))
#plt.show()


plt.plot(gaussian_blue_3, 'b')
# print(max(gaussian_blue_3))
plt.show()

plt.plot(gaussian_green_3, 'g')
# print(max(gaussian_green_3))
plt.show()


plt.plot(gaussian_red_3, 'r')
# print(max(gaussian_red_3))
plt.show()


c = cv2.VideoCapture("C:/Users/p1889/OneDrive/Desktop/detectbuoy.avi")

# defining size using dimensions of the given frame
size = (640, 480)

# creating output video file
out = cv2.VideoWriter('FinalOutput3d.avi', cv2.VideoWriter_fourcc(*'DIVX'), 5, size)

while True:
    ret, every_image = c.read()

    # slicing every frame into 3 channels
    red_channel = every_image[:, :, 2]
    green_channel = every_image[:, :, 1]
    blue_channel = every_image[:, :, 0]
    check =0
    if ret:
        output_frame_1 = np.zeros(red_channel.shape, dtype=np.uint8)
        output_frame_2 = np.zeros(red_channel.shape, dtype=np.uint8)
        output_frame_3 = np.zeros(green_channel.shape, dtype=np.uint8)
        check = check + 1

        for i in range(0, red_channel.shape[0]):
            for j in range(0, red_channel.shape[1]):
                r_element = red_channel[i][j]
                g_element = green_channel[i][j]
                #Orange
                if gaussian_red_1[r_element] > 0.001 and blue_channel[i][j] < 150:
                    output_frame_1[i][j] = 255

                if gaussian_green_1[r_element] > 0.02 and blue_channel[i][j] < 150:
                    output_frame_1[i][j] = 255

                if gaussian_blue_1[r_element] > 0.019 and blue_channel[i][j] < 150:
                    output_frame_1[i][j] = 255

                ###GREEN



                if gaussian_red_3[g_element] > 0.08 and gaussian_green_3[g_element] < 0.017  and gaussian_blue_3[g_element] < 0.015 and red_channel[i][j]<150 :
                    output_frame_3[i][j] = 255
                #else:
                 #   output_frame_3[i][j] = 0

                ######YELLOW

                if (gaussian_red_2[r_element]) > 0.083 and gaussian_red_2[g_element] > 0.01 and blue_channel[i][j] < 150:
                    output_frame_2[i][j] = 255
                if gaussian_blue_2[r_element] > 0.02 and gaussian_blue_2[g_element] > 0.01 and blue_channel[i][j] < 150:
                    output_frame_2[i][j] = 255
                if gaussian_green_2[r_element] > 0.001 and blue_channel[i][j] < 150:
                    output_frame_2[i][j] = 255


        blur_red = cv2.medianBlur(output_frame_1, 3)
        edged_red = cv2.Canny(blur_red, 50, 255)
        cv2.imshow("edge", edged_red)
        contours_red, h_red = cv2.findContours(edged_red, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_sorted_red = sorted(contours_red, key=cv2.contourArea, reverse=True)
        try:
            (x, y), radius = cv2.minEnclosingCircle(contours_sorted_red[0])
            center = (int(x), int(y))
            radius = int(radius)
            if 10 < radius < 25:
                cv2.circle(every_image, (int(x), int(y)), int(radius), (255, 0, 0), 2)
        except:
            pass

        ret, threshold_green = cv2.threshold(output_frame_3, 254, 255, cv2.THRESH_BINARY)
        kernel_for_green = np.ones((2, 2), np.uint8)

        dilation_green = cv2.dilate(threshold_green, kernel_for_green, iterations=9)
        contours1, _ = cv2.findContours(dilation_green, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours1:
            if cv2.contourArea(contour) > 20:
                (x, y), radius = cv2.minEnclosingCircle(contour)
                center = (int(x), int(y))
                radius = int(radius)
                if 13.5 < radius < 25:
                    cv2.circle(every_image, center, radius, (255, 0, 0), 2)

        blur_yellow = cv2.medianBlur(output_frame_2, 3)
        edged_yellow = cv2.Canny(blur_yellow, 50, 255)
        # cv2.imshow("edge", edged_yellow)
        contours_yellow, h_yellow = cv2.findContours(edged_yellow, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours_sorted_yellow = sorted(contours_yellow, key=cv2.contourArea, reverse=True)
        try:
            (x, y), radius = cv2.minEnclosingCircle(contours_sorted_yellow[0])
            center = (int(x), int(y))
            radius = int(radius)
            if 10 < radius < 35:
                cv2.circle(every_image, (int(x), int(y)), int(radius), (255,0, 0), 2)
        except:
            pass

        cv2.imshow("Threshold", dilation_green)
        cv2.imshow('Final Output', every_image)
        out.write(every_image)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break  # wait for ESC key to exit


out.release()
c.release()
cv2.destroyAllWindows()