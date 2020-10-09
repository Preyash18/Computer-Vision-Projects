import cv2
import numpy as np
import copy
name = "C:/UMD/Spring2020/673-Perception for autonomous/Project3_me/detectbuoy.avi"
cap = cv2.VideoCapture(name)

points = []


def getImage():
    
    image = cv2.imread('C:/UMD/Spring2020/673-Perception for autonomous/Project3_me/frame 3.6 sec.jpg')
    return image
    
def crop_contour(points):
    print("In crop contour")
    image = getImage()

    new_list = []
    for elem in points:
        new_list.append(np.array(elem))
    print(new_list)    
    new_list = np.array(new_list)
    mask = np.zeros_like(image)
    cv2.drawContours(mask,[new_list],-1,(255,255,255),-1)
    mask = cv2.bitwise_not(mask)
    final = cv2.add(image,mask)
       
    # cv2.imshow("new",final)
    # cv2.waitKey(100)
    x,y,c = np.where(final != 255)
    TL_x,TL_y = np.min(x),np.min(y)
    BR_x,BR_y = np.max(x),np.max(y)
    count = 6
    cropped = final[TL_x-20:BR_x+20,TL_y-20:BR_y+20]
    #cv2.imwrite("yellow"+str(count)+"buoy.jpg",cropped)
    cv2.imwrite("ornge"+str(count)+"buoy.jpg",cropped)
    #cv2.imwrite("orange"+str(count)+"buoy.jpg",cropped)
    cv2.imshow("cropped",cropped)
    #cv2.waitKey(100)
    
    
def click_and_crop(event, x, y, flag, params):
    #count = 0      
    if event==cv2.EVENT_LBUTTONDOWN:
        points.append((x,y))        
        print(points)
        
        #image = getImage()
        if len(points) >=2 :
            cv2.line(image,points[-1],points[-2],(0,0,0),1)
            cv2.imshow("name",image)
        if len(points) == 10:
            crop_contour(points)
            cv2.imshow("name",image)   



image = getImage()
cv2.imshow("name",image)
cv2.setMouseCallback("name",click_and_crop)
cv2.waitKey(0)