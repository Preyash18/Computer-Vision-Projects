import cv2
import numpy as np
import copy
import math
############ Asking user ######################
print("Choose Tag videos")
print("press 0 for Tag0")
print("press 1 for Tag1")
print("press 2 for Tag2")
print("press 3 for Multiple_tags")
print("")
ent = int(input("Your input: "))
if ent == 0:
    vid = cv2.VideoCapture('Tag0.mp4')
elif ent == 1:
    vid = cv2.VideoCapture('Tag1.mp4')
elif ent == 2:
    vid = cv2.VideoCapture('Tag2.mp4')
elif ent == 3:
    vid = cv2.VideoCapture('multipleTags.mp4')
else:
    print("No Tags! Try again")
    exit(0)
image1 = cv2.imread('Lena.png')

def TagMatrix(img):
    dimension_tag = img.shape  # Calculate the shape of the image
    height_img = dimension_tag[0]
    width_img = dimension_tag[1]
    bit_height = int((height_img / 8))
    bit_width = int(width_img / 8)
    a = 0
    b = 0
    ar_tag = np.empty((8, 8))  # Initialising the 8X8 matrix
    for i in range(0, height_img, bit_height):
        b = 0
        for j in range(0, width_img, bit_width):
            count_black_boxes = 0
            count_white_boxes = 0
            for x in range(0, bit_height - 1):
                for y in range(0, bit_width - 1):
                    if (img[i + x][j + y] == 0):
                        count_black_boxes = count_black_boxes + 1
                    else:
                        count_white_boxes = count_white_boxes + 1

            if (count_white_boxes >= count_black_boxes):  # Checking whether that block has more white or black pixel and corresponding assigning it in the tag matrix
                ar_tag[a][b] = 1
            else:
                ar_tag[a][b] = 0
            b = b + 1
        a = a + 1
    return ar_tag


# Comparing the inner 4x4 grid to check for the orientation of the tag
def Tag_chara(ar_tag):
    ar_tag_created = TagMatrix(ar_tag)
    # Checking the location of white block in the inner 4X4 matrix of the AR tag to detect the orientation of the tag in camera frame
    if (ar_tag_created[2][2] == 0 and ar_tag_created[2][5] == 0 and ar_tag_created[5][2] == 0 and ar_tag_created[5][5] == 1):
        rotation_by_angle = 0
    elif (ar_tag_created[2][2] == 1 and ar_tag_created[2][5] == 0 and ar_tag_created[5][2] == 0 and ar_tag_created[5][5] == 0):
        rotation_by_angle = 180
    elif (ar_tag_created[2][2] == 0 and ar_tag_created[2][5] == 1 and ar_tag_created[5][2] == 0 and ar_tag_created[5][5] == 0):
        rotation_by_angle = 90
    elif (ar_tag_created[2][2] == 0 and ar_tag_created[2][5] == 0 and ar_tag_created[5][2] == 1 and ar_tag_created[5][5] == 0):
        rotation_by_angle = -90
    else:
        rotation_by_angle = None

    if (rotation_by_angle == None):
        flag = False
        return flag, rotation_by_angle
    else:
        flag = True
        return flag, rotation_by_angle


# Finding the Homography of the AR tag from World Coordinate frame to Image Coordinate frame
def find_homography(frame1, frame2):
    A = []

    for i in range(0, len(frame2)):
        x, y = frame1[i][0], frame1[i][1]
        u, v = frame2[i][0], frame2[i][1]
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
        A.append([0, 0, 0, x, y, 1, -v * x, -v * y, -v])
    A = np.array(A)
    U, S, Vh = np.linalg.svd(A)
    l = Vh[-1, :] / Vh[-1, -1]
    h = np.reshape(l, (3, 3))
    return h


def projection_Matrix(h, K):  # h is the homographic matrix and k is the camera calibration matrix
    h1 = h[:, 0]
    h2 = h[:, 1]

    lamda = 2 / (np.linalg.norm(np.matmul(np.linalg.inv(K), h1)) + np.linalg.norm(np.matmul(np.linalg.inv(K), h2)))
    b_tilda = lamda * np.matmul(np.linalg.inv(K), h)

    d = np.linalg.det(b_tilda)
    if d > 0:
        b = b_tilda
    else:
        b = -1 * b_tilda
    row1 = b[:, 0]
    row2 = b[:, 1]
    row3 = np.cross(row1, row2)
    l = b[:, 2]
    R = np.column_stack((row1, row2, row3, l))
    P_matrix = np.matmul(K, R)  # projection matrix
    return P_matrix

def orientation(list):
    count = 0
    top_left, top_right, bottom_left, bottom_right = 0, 0, 0, 0
    x = list[0][1]
    y = list[0][0]
    corner = ''
    # Condition to extract AR Tag corners and the index of the corners to be fed into 'corner_four' list
    if thresh[x - 10][y - 10] == 255:
        count = count + 1
        top_left = 1
    if thresh[x + 10][y + 10] == 255:
        count = count + 1
        bottom_right = 1
    if thresh[x - 10][y + 10] == 255:
        count = count + 1
        top_right = 1
    if thresh[x + 10][y - 10] == 255:
        count = count + 1
        bottom_left = 1
    if count == 3:
        if bottom_right == 0:
            corner = 'TL'
        elif bottom_left == 0:
            corner = 'TR'
        elif top_right == 0:
            corner = 'BL'
        elif top_left == 0:
            corner = 'BR'
    return y, x, corner

def reorient(angle, list):
    corner_actual = [[], [], [], []]
    if angle == 0:
        corner_actual = list
    elif angle == 90:
        corner_actual = [list[2], list[0], list[3], list[1]]
    elif angle == -90:
        corner_actual = [list[1], list[3], list[0], list[2]]
    elif angle == 180:
        corner_actual = [list[3], list[2], list[1], list[0]]
    return corner_actual


K = np.array(
    [[1406.08415449821, 0, 0], [2.20679787308599, 1417.99930662800, 0], [1014.13643417416, 566.347754321696, 1]]).T

if (vid.isOpened() == False):
    print("Error opening video file")

if ent == 0 or ent == 1 or ent ==2:
    while (vid.isOpened()):
        ret, frame = vid.read()    # Capture frame-by-frame
        if ret == True:
            smooth = cv2.GaussianBlur(frame, (7, 7), 0)    # Smoothing # 5,5 kernel # border= 0
            gray = cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY)  # cv2.cvtColor() method is used to convert an image from one color space to another.

            ret, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
            # find contours present in the video
            contours, hie = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            corners_list = []
            blank = []
            #blank2 = []
            for i in hie[0]:
                four_contour_list = []
                if i[3] == 1:
                    blank.append(i[3])

                for c in blank:
                    if cv2.contourArea(contours[c]) > 750:

                        epsilon = 0.1 * cv2.arcLength(contours[c], True)
                        contour_corners = cv2.approxPolyDP(contours[c], epsilon, True)

                        for j in contour_corners:
                            y, x, corner = orientation(j)
                            four_contour_list.append([y, x, corner])

                    if len(four_contour_list) == 4:
                        corners_list.append(four_contour_list)

            if corners_list != []:  # if listforcorners is not empty then go forward
                for i in range(0, len(corners_list)):
                    list = [0, 0, 0, 0]  # to put x and y corner values in a list
                    for value in corners_list[i]:
                        if value[-1] == 'TL':
                            list[0] = value[0:2]
                        elif value[-1] == 'TR':
                            list[1] = value[0:2]
                        elif value[-1] == 'BL':
                            list[2] = value[0:2]
                        elif value[-1] == 'BR':
                            list[3] = value[0:2]

                    if 0 not in list:
                        H = find_homography(list, [[0, 0], [199, 0], [0, 199], [199, 199]])
                        H_inv = np.linalg.inv(H)
                        black = np.zeros((200, 200))
                        for a in range(0, 200):
                            for b in range(0, 200):
                                x, y, z = np.matmul(H_inv, [a, b, 1])
                                y_dash = int(y / z)
                                x_dash = int(x /z)
                                if (1080 > y_dash > 0) and (1920 > x_dash > 0):
                                    black[a][b] = thresh[y_dash][x_dash]
                        flag, angle = Tag_chara(black)

                        if flag:
                            actual_corners = reorient(angle, list)

                            H_new = find_homography([[0, 0], [199, 0], [0, 199], [199, 199]], actual_corners)
                            P = projection_Matrix(H_new, K)

                            x1, y1, z1 = np.matmul(P, [0, 0, 0, 1])
                            x2, y2, z2 = np.matmul(P, [0, 199, 0, 1])
                            x3, y3, z3 = np.matmul(P, [199, 0, 0, 1])
                            x4, y4, z4 = np.matmul(P, [199, 199, 0, 1])
                            x5, y5, z5 = np.matmul(P, [0, 0, -199, 1])
                            x6, y6, z6 = np.matmul(P, [0, 199, -199, 1])
                            x7, y7, z7 = np.matmul(P, [199, 0, -199, 1])
                            x8, y8, z8 = np.matmul(P, [199, 199, -199, 1])

                            # four vertical
                            cv2.line(frame, (int(x1 / z1), int(y1 / z1)), (int(x5 / z5), int(y5 / z5)), (0, 255, 0), 5)
                            cv2.line(frame, (int(x2 / z2), int(y2 / z2)), (int(x6 / z6), int(y6 / z6)), (0, 255, 0), 5)
                            cv2.line(frame, (int(x3 / z3), int(y3 / z3)), (int(x7 / z7), int(y7 / z7)), (0, 255, 0), 5)
                            cv2.line(frame, (int(x4 / z4), int(y4 / z4)), (int(x8 / z8), int(y8 / z8)), (0, 255, 0), 5)
                            # down /
                            cv2.line(frame, (int(x1 / z1), int(y1 / z1)), (int(x2 / z2), int(y2 / z2)), (0, 255, 255), 5)
                            cv2.line(frame, (int(x1 / z1), int(y1 / z1)), (int(x3 / z3), int(y3 / z3)), (0, 255, 255), 5)
                            cv2.line(frame, (int(x2 / z2), int(y2 / z2)), (int(x4 / z4), int(y4 / z4)), (0, 255, 255), 5)
                            cv2.line(frame, (int(x3 / z3), int(y3 / z3)), (int(x4 / z4), int(y4 / z4)), (0, 255, 255), 5)
                            # four/
                            cv2.line(frame, (int(x5 / z5), int(y5 / z5)), (int(x6 / z6), int(y6 / z6)), (255, 0, 0), 5)
                            cv2.line(frame, (int(x5 / z5), int(y5 / z5)), (int(x7 / z7), int(y7 / z7)), (255, 0, 0), 5)
                            cv2.line(frame, (int(x6 / z6), int(y6 / z6)), (int(x8 / z8), int(y8 / z8)), (255, 0, 0), 5)
                            cv2.line(frame, (int(x7 / z7), int(y7 / z7)), (int(x8 / z8), int(y8 / z8)), (255, 0, 0), 5)

        cv2.drawContours(frame, contour_corners, -1, (0, 0, 255), 3)
        cv2.imshow('Frame', frame)
        # print("Frame", len(frame[0]))

        # Press a on keyboard to  exit
        if cv2.waitKey(2) & 0xFF == ord('a'):
            break

elif ent == 3:
    while (vid.isOpened()):
        ret, frame = vid.read()  # Capture frame-by-frame
        if ret == True:
            smooth = cv2.GaussianBlur(frame, (7, 7), 0)  # Smoothing # 5,5 kernel # border= 0
            gray = cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY)
            # cv2.cvtColor() method is used to convert an image from one color space to another.

            ret, thresh = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
            # find contours present in the video
            contours, hie = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            corners_list = []
            blank = []
            for i in hie[0]:
                four_contour_list = []
                if i[3] != 0 and i[3] != -1 and i[3] != 2:
                    blank.append(i[3])

                for c in blank:
                    if 1480 < cv2.contourArea(contours[c]) < 13000:

                        epsilon = 0.1 * cv2.arcLength(contours[c], True)
                        contour_corners = cv2.approxPolyDP(contours[c], epsilon, True)

                        for j in contour_corners:
                            y, x, corner1 = orientation(j)
                            four_contour_list.append([y, x, corner1])

                    if len(four_contour_list) == 4:
                        corners_list.append(four_contour_list)

            if corners_list != []:  # if listforcorners is not empty then go forward
                for i in range(0, len(corners_list)):
                    list = [0, 0, 0, 0]  # to put x and y corner values in a list
                    for value in corners_list[i]:
                        if value[-1] == 'TL':
                            list[0] = value[0:2]
                        elif value[-1] == 'TR':
                            list[1] = value[0:2]
                        elif value[-1] == 'BL':
                            list[2] = value[0:2]
                        elif value[-1] == 'BR':
                            list[3] = value[0:2]

                    if 0 not in list:
                        H = find_homography(list, [[0, 0], [199, 0], [0, 199], [199, 199]])
                        H_inv = np.linalg.inv(H)
                        black = np.zeros((200, 200))
                        for a in range(0, 200):
                            for b in range(0, 200):
                                x, y, z = np.matmul(H_inv, [a, b, 1])
                                y_dash = int(y / z)
                                x_dash = int(x / z)
                                if (1080 > y_dash > 0) and (1920 > x_dash > 0):
                                    black[a][b] = thresh[y_dash][x_dash]
                        flag, angle = Tag_chara(black)

                        if flag:
                            actual_corners = reorient(angle, list)

                            H_new = find_homography([[0, 0], [199, 0], [0, 199], [199, 199]], actual_corners)
                            P = projection_Matrix(H_new, K)

                            x1, y1, z1 = np.matmul(P, [0, 0, 0, 1])
                            x2, y2, z2 = np.matmul(P, [0, 199, 0, 1])
                            x3, y3, z3 = np.matmul(P, [199, 0, 0, 1])
                            x4, y4, z4 = np.matmul(P, [199, 199, 0, 1])
                            x5, y5, z5 = np.matmul(P, [0, 0, -199, 1])
                            x6, y6, z6 = np.matmul(P, [0, 199, -199, 1])
                            x7, y7, z7 = np.matmul(P, [199, 0, -199, 1])
                            x8, y8, z8 = np.matmul(P, [199, 199, -199, 1])

                            # four vertical
                            cv2.line(frame, (int(x1 / z1), int(y1 / z1)), (int(x5 / z5), int(y5 / z5)), (0, 255, 0), 5)
                            cv2.line(frame, (int(x2 / z2), int(y2 / z2)), (int(x6 / z6), int(y6 / z6)), (0, 255, 0), 5)
                            cv2.line(frame, (int(x3 / z3), int(y3 / z3)), (int(x7 / z7), int(y7 / z7)), (0, 255, 0), 5)
                            cv2.line(frame, (int(x4 / z4), int(y4 / z4)), (int(x8 / z8), int(y8 / z8)), (0, 255, 0), 5)
                            # down /
                            cv2.line(frame, (int(x1 / z1), int(y1 / z1)), (int(x2 / z2), int(y2 / z2)), (0, 255, 255),
                                     5)
                            cv2.line(frame, (int(x1 / z1), int(y1 / z1)), (int(x3 / z3), int(y3 / z3)), (0, 255, 255),
                                     5)
                            cv2.line(frame, (int(x2 / z2), int(y2 / z2)), (int(x4 / z4), int(y4 / z4)), (0, 255, 255),
                                     5)
                            cv2.line(frame, (int(x3 / z3), int(y3 / z3)), (int(x4 / z4), int(y4 / z4)), (0, 255, 255),
                                     5)
                            # four/
                            cv2.line(frame, (int(x5 / z5), int(y5 / z5)), (int(x6 / z6), int(y6 / z6)), (255, 0, 0), 5)
                            cv2.line(frame, (int(x5 / z5), int(y5 / z5)), (int(x7 / z7), int(y7 / z7)), (255, 0, 0), 5)
                            cv2.line(frame, (int(x6 / z6), int(y6 / z6)), (int(x8 / z8), int(y8 / z8)), (255, 0, 0), 5)
                            cv2.line(frame, (int(x7 / z7), int(y7 / z7)), (int(x8 / z8), int(y8 / z8)), (255, 0, 0), 5)

        cv2.imshow('Frame', frame)
        # print("Frame", len(frame[0]))

        # Press a on keyboard to  exit
        if cv2.waitKey(2) & 0xFF == ord('a'):
            break

vid.release()
# Closes all the frames
cv2.destroyAllWindows()