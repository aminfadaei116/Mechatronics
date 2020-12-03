import numpy as np
import cv2
import math
from sklearn.metrics import accuracy_score


TrueLabel = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
def distance(data1, data2):
    tmp = (np.subtract(data1, data2))
    return np.sum(tmp*tmp)

Label = {0:[],1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[]}

for i in range(10):
    # print("0_simple number/"+str(i)+".PNG is loaded")
    im1 = cv2.imread("0_simple number/"+str(i)+".PNG",cv2.IMREAD_GRAYSCALE)
    _,im1 = cv2.threshold(im1, 128, 255, cv2.THRESH_BINARY)
    Label[i].append(im1)


predictScale = []
for i in range(10):
    # print("1_Scale Number/" + str(i) + ".PNG is loaded")
    im2 = cv2.imread("1_Scale Number/" + str(i) + ".PNG", cv2.IMREAD_GRAYSCALE)
    _, im2 = cv2.threshold(im2, 128, 255, cv2.THRESH_BINARY)
    dist = []
    for j in range(10):
        dist.append(cv2.matchShapes(im2, Label[j][0], cv2.CONTOURS_MATCH_I2, 0))
    predictScale.append(dist.index(min(dist)))

print("The predicted labels for scale is",predictScale)
print("And the accuracy is:",accuracy_score(TrueLabel, predictScale)*100,'%')


predictRotation = []
for i in range(10):
    # print("1_Scale Number/" + str(i) + ".PNG is loaded")
    im2 = cv2.imread("2_Rotation Number/" + str(i) + ".PNG", cv2.IMREAD_GRAYSCALE)
    _, im2 = cv2.threshold(im2, 128, 255, cv2.THRESH_BINARY)
    dist = []
    for j in range(10):
        dist.append(cv2.matchShapes(im2, Label[j][0], cv2.CONTOURS_MATCH_I2, 0))
    predictRotation.append(dist.index(min(dist)))

print("The predicted labels for scale is",predictRotation)
print("And the accuracy is:",accuracy_score(TrueLabel, predictRotation)*100,'%')

predictRotationScale = []
for i in range(10):
    # print("1_Scale Number/" + str(i) + ".PNG is loaded")
    im2 = cv2.imread("3_Scale_Rotation_Number/" + str(i) + ".PNG", cv2.IMREAD_GRAYSCALE)
    _, im2 = cv2.threshold(im2, 128, 255, cv2.THRESH_BINARY)
    dist = []
    for j in range(10):
        dist.append(cv2.matchShapes(im2, Label[j][0], cv2.CONTOURS_MATCH_I2, 0))
    predictRotationScale.append(dist.index(min(dist)))

print("The predicted labels for scale is",predictRotationScale)
print("And the accuracy is:",accuracy_score(TrueLabel, predictRotationScale)*100,'%')
