import cv2
import numpy as np

# Opens the Video file
cap = cv2.VideoCapture('NightDrive.mp4')

# defining size using dimensions of the given frame
size = (1920, 1080)

i = 0

#breaking video into frames
while (cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break
    cv2.imwrite('fr' + str(i) + '.jpg', frame)
    i += 1

cap.release()
cv2.destroyAllWindows()
gamma = 0.4

# changing gamma values for all frames
for j in range(i):
    img = cv2.imread('fr' + str(j) + '.jpg')
    #    imgYCC = cv2.cvtColor(img, cv2.COLOR_RGB2YCR_CB)
    #    channels = cv2.split(imgYCC)
    #    channels[2]=cv2.equalizeHist(channels[2])
    #    fimg=cv2.merge(channels)
    #    imgBGR=cv2.cvtColor(fimg, cv2.COLOR_YCrCb2RGB)
    lookUpTable = np.empty((1, 256), np.uint8)
    for i in range(256):
        lookUpTable[0, i] = np.clip(pow(i / 150.0, gamma) * 150.0, 0, 150)
    res = cv2.LUT(img, lookUpTable)
    filename = 'rfr' + str(j) + '.jpg'
    cv2.imwrite(filename, res)

# compiling all frames to create video file
img_array = []
for j in range(i):
    filename = 'rfr' + str(j) + '.jpg'
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width, height)
    img_array.append(img)
out = cv2.VideoWriter('FinalOutput.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()
