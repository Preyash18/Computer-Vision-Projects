import cv2
import numpy as np
from matplotlib import pyplot as plt


def plot_image(image, title):
    # plt.imshow(image, cmap=plt.cm.gray)
    plt.title(title)
    plt.show()


flag = False


# function to smooth out the image
def smooth(image):
    blur_image = cv2.GaussianBlur(image, (5, 5), 0)
    if flag:
        plot_image(blur_image, "blur")
    return blur_image


# function to apply canny edge detection on image
def canny(image):
    canny_image = cv2.Canny(image, 150, 250)
    if flag:
        plot_image(canny_image, "canny")
    return canny_image


# function for defining region of interest
def roi(image):
    height = image.shape[0]
    width = image.shape[1]
    # only cropping upper half of the frame
    bottom_left = [0, height]
    bottom_right = [width, height]
    top_right = [0, height * 17 / 30]
    top_left = [width, height * 17 / 30]
    vertices = [np.array([bottom_left, bottom_right, top_left, top_right], dtype=np.int32)]
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, vertices, [255, 255, 255])
    if flag:
        plot_image(mask, "mask")
    masked_image = cv2.bitwise_and(image, mask)
    if flag:
        plot_image(masked_image, "roi")
    return masked_image


def filtering_lines(image, lines):
    right_lines = []
    left_lines = []
    for x1, y1, x2, y2 in lines[:, 0]:
        parameters = np.polyfit((x1, x2), (y1, y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        if slope >= 0.5:
            right_lines.append([slope, intercept])
        elif slope <= -0.5:
            left_lines.append([slope, intercept])
        else:
            pass

    def combine_lines(image, lines):
        if len(lines) > 0:
            slope, intercept = np.average(lines, axis=0)
            y1 = int(image.shape[0] + 90)
            y2 = int(y1 * (1 / 2) + 90)
            x1 = int(((y1 - intercept) / slope))
            x2 = int(((y2 - intercept) / slope))
            return np.array([x1, y1, x2, y2])

    left = combine_lines(image, left_lines)
    #print("ll", left_lines)
    right = combine_lines(image, right_lines)
    #print("rr", right_lines)
    return left, right


def Hough_line(image, rho, theta, threshold, min_line_len, max_line_gap):
    lines_image = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    lines = cv2.HoughLinesP(image, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    # print("lines", lines)
    if lines is not None:
        lines = filtering_lines(image, lines)
        # print("lines", lines)
        for line in lines:
            if line is not None:
                x1, y1, x2, y2 = line
                cv2.line(lines_image, (x1, y1), (x2, y2), (0, 0, 255), 20)
        if flag:
            plot_image(lines_image, "lines")
        if len(lines) == 2:
            if lines[1] is None or lines[0] is None:
                return lines_image
            # code for plotting mesh on detected lane
            if lines[1] is not None and lines[0] is not None:
                x1, y1, x2, y2 = lines[0]
                x3, y3, x4, y4 = lines[1]
                #print("lines[0]", lines[0])
                #print("lines[1]", lines[1])
                position = (600, 500)
                m = (y4 - y3) / (x4 - x3)
                mask = np.zeros_like(lines_image)
                vertices = [np.array([[x1, y1], [x2, y2], [x4, y4], [x3, y3]], dtype=np.int32)]
                cv2.fillPoly(mask, vertices, [0, 255, 255])

                # turn prediction
                if m > 0.8:
                    cv2.putText(mask, 'Turning Right', position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)
                elif 0 <= m <= 0.8:
                    cv2.putText(mask, 'Going Straight', position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)
                elif m <= 0:
                    cv2.putText(mask, 'Turning Left', position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)
                if flag:
                    plot_image(mask, "mask")
                return mask


def combine_images(image, initial_image, alpha=0.8, beta=2, gamma=0.0):
    combined_image = cv2.addWeighted(initial_image, alpha, image, beta, gamma)
    if flag:
        plot_image(combined_image, "combined")
    return combined_image


# defining size using dimensions of the given video
size = (1280, 720)

# creating output video file
out = cv2.VideoWriter('FinalOutput22.avi', cv2.VideoWriter_fourcc(*'DIVX'), 5, size)

#Camera Matrix
K = np.array([[1.15422732e+03, 0.00000000e+00, 6.71627794e+02], [0.00000000e+00, 1.14818221e+03, 3.86046312e+02], [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

#Distortion Coefficients
D = np.array([[-2.42565104e-01, -4.77893070e-02, -1.31388084e-03, -8.79107779e-05, 2.20573263e-02]])

capture = cv2.VideoCapture('challenge_video.mp4')

# main code run
# applying operations on each frame from the given dataset
while True:
    ret, frame = capture.read()
    if frame is not None:
        # undistorting the image
        undistorted_image = cv2.undistort(frame, K, D, None, K)

        # smoothing out the frames
        smooth_image = smooth(undistorted_image)

        # applying roi mask on the frame
        roi_defined_image = roi(smooth_image)

        # converting frame into hsl image
        hsl_img = cv2.cvtColor(roi_defined_image, cv2.COLOR_BGR2HLS)

        # defining yellow mask to find yellow lanes
        yellow_lower_range = np.array([16, 120, 80], dtype='uint8')
        yellow_upper_range = np.array([45, 200, 255], dtype='uint8')
        yellow_mask = cv2.inRange(hsl_img, yellow_lower_range, yellow_upper_range)
        yellow_mask_final = cv2.bitwise_and(hsl_img, hsl_img, mask=yellow_mask).astype(np.uint8)

        # defining white mask to detect white lanes
        white_lower_range = np.array([0, 175, 0], dtype='uint8')
        white_upper_range = np.array([255, 255, 255], dtype='uint8')
        white_mask = cv2.inRange(hsl_img, white_lower_range, white_upper_range)
        white_mask_final = cv2.bitwise_and(hsl_img, hsl_img, mask=white_mask).astype(np.uint8)

        # combining both masks
        masks_combined = cv2.bitwise_or(yellow_mask_final, white_mask_final)
        lanes_using_masks = cv2.cvtColor(masks_combined, cv2.COLOR_HLS2BGR)
        lanes_using_masks_colored = cv2.cvtColor(lanes_using_masks, cv2.COLOR_BGR2GRAY)

        # applying canny edge detector to get edges of the lanes
        detected_edges_canny = canny(lanes_using_masks_colored)

        # applying hough lines function to get lines surrounding the lanes
        Hough_lines_image = Hough_line(detected_edges_canny, 0.9, np.pi / 180, 17, 20, 2500)
        final_image = combine_images(Hough_lines_image, frame)
        cv2.imshow("video", final_image)
        out.write(final_image)
        cv2.waitKey(27)
    else:
        break

out.release()
capture.release()
cv2.destroyAllWindows()
exit()
