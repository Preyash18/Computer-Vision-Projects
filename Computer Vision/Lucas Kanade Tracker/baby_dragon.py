import numpy as np
import cv2
import glob
from scipy.ndimage import affine_transform

img_array = []
for filename in glob.glob('DragonBaby/img/*.jpg'):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width, height)
    img_array.append(img)

first_frame = img_array[0]
ref_point = []
crop = False

mul_fact = 0.03125
rect_points = [(int(mul_fact * 128), int(mul_fact * 65)), (int(mul_fact * 251), int(mul_fact * 175))]

rect = np.array([rect_points[0][0], rect_points[0][1], rect_points[1][0], rect_points[1][1]])
rect1 = np.reshape(np.array([rect_points[0][0], rect_points[0][1], 1]), (3, 1))
rect2 = np.reshape(np.array([rect_points[1][0], rect_points[1][1], 1]), (3, 1))
first_frame_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

# reducing resolution
img_pyrdown1 = cv2.pyrDown(first_frame_gray)
img_pyrdown2 = cv2.pyrDown(img_pyrdown1)
img_pyrdown3 = cv2.pyrDown(img_pyrdown2)
img_pyrdown4 = cv2.pyrDown(img_pyrdown3)
final_image = cv2.pyrDown(img_pyrdown4)

img_list = []

for next_frame in img_array:
    next_img_copy = next_frame.copy()
    next_frame_gray = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)

    # reducing resolution
    next_img_pyrdown1 = cv2.pyrDown(next_frame_gray)
    next_img_pyrdown2 = cv2.pyrDown(next_img_pyrdown1)
    next_img_pyrdown3 = cv2.pyrDown(next_img_pyrdown2)
    next_img_pyrdown4 = cv2.pyrDown(next_img_pyrdown3)
    final_next_image = cv2.pyrDown(next_img_pyrdown4)

    M = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    p = np.asarray([0.0] * 6)
    I = M
    threshold = 0.005

    for i in range(100):
        img_warp = affine_transform(final_next_image, np.flip(M)[..., [1, 2, 0]])

        mask = affine_transform(np.ones(final_next_image.shape), np.flip(M)[..., [1, 2, 0]])
        error_img = (mask * final_image) - (mask * img_warp)
        grad = np.dstack(np.gradient(final_next_image)[::-1])
        grad[:, :, 0] = affine_transform(grad[:, :, 0], np.flip(M)[..., [1, 2, 0]])
        grad[:, :, 1] = affine_transform(grad[:, :, 1], np.flip(M)[..., [1, 2, 0]])
        warp_grad = grad.reshape(grad.shape[0] * grad.shape[1], 2).T
        h, w = final_image.shape
        T_x = np.tile(np.linspace(0, w - 1, w), (h, 1)).flatten()
        T_y = np.tile(np.linspace(0, h - 1, h), (w, 1)).T.flatten()

        final_stack_array = np.vstack([warp_grad[0] * T_x, warp_grad[0] * T_y,
                                       warp_grad[0], warp_grad[1] * T_x, warp_grad[1] * T_y,
                                       warp_grad[1]]).T

        hessian = np.matmul(final_stack_array.T, final_stack_array)

        delta_p = np.matmul(np.linalg.inv(hessian), np.matmul(final_stack_array.T, error_img.flatten()))

        p = p + delta_p
        M = p.reshape(2, 3) + I

        if np.linalg.norm(delta_p) <= threshold:
            break

    new_rect1 = np.matmul(M, rect1)
    new_rect2 = np.matmul(M, rect2)
    # drawing rectangle on frame
    cv2.rectangle(next_img_copy, (int(32 * new_rect1[0]), int(32 * new_rect1[1])),
                  (int(32 * new_rect2[0]), int(32 * new_rect2[1])), (0, 0, 255), 2)
    cv2.imshow('baby', next_img_copy)
    img_list.append(next_img_copy)
    if cv2.waitKey(1) == 27:
        cv2.destroyAllWindows()
        break

# cv2.imwrite('first_frame_baby_dragon.jpg', img_list[0])
out = cv2.VideoWriter('baby_dragon_final.avi', cv2.VideoWriter_fourcc(*'XVID'), 5.0, (640, 360))
for image in img_list:
    out.write(image)
    cv2.waitKey(10)

out.release()
