import numpy as np
import cv2
import math

def RGBtoHSVconverter(blue, green, red):
    red, green, blue = red/255.0, green/255.0, blue/255.0
    maxcol= max(red, green, blue)
    mincol = min(red, green, blue)
    rangecol = maxcol - mincol
    if maxcol == mincol:
        h = 0
    elif maxcol == red:
        h = (60 * (((green-blue)/rangecol) % 6))
    elif maxcol == green:
        h = (60 * (((blue-red)/rangecol) + 2))
    elif maxcol == blue:
        h = (60 * (((red-green)/rangecol) + 4))
    if maxcol == 0:
        s = 0
    else:
        s = rangecol/maxcol
    v = maxcol
    return int(round(h/2)), int(round(s*255)), int(round(v*255 ))

print(RGBtoHSVconverter(100,120,170))


img = cv2.imread('capture.jpg')

print(img.shape)
handmaded = np.zeros([img.shape[0],img.shape[1],img.shape[2]],dtype = np.uint8)

for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        a,b,c = (RGBtoHSVconverter(img[i,j,0],img[i,j,1],img[i,j,2]))
        print(a)
        handmaded[i,j,0] = int(a)
        print(handmaded[i,j,0])
        handmaded[i,j,1] = int(b)
        handmaded[i,j,2] = int(c)
        print(handmaded[i,j,:])
        # print(RGBtoHSVconverter(img[i,j,0],img[i,j,1],img[i,j,2]))



new_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


cv2.imshow('image',img)
cv2.imshow('new_image', new_img)
cv2.imshow("hand_maded_image", handmaded)
cv2.waitKey(0)
cv2.destroyAllWindows()

