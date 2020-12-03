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
    im = cv2.imread("0_simple number/"+str(i)+".PNG",cv2.IMREAD_GRAYSCALE)
    _,im = cv2.threshold(im, 128, 255, cv2.THRESH_BINARY)
# Calculate Moments
    moments = cv2.moments(im)
# Calculate Hu Moments
    huMoments = cv2.HuMoments(moments)
    for j in range(0,7):
        huMoments[j] = -1* math.copysign(1.0, huMoments[j]) * math.log10(abs(huMoments[j]))

    Label[i].append(huMoments)


predictScale = []
for i in range(10):
    # print("1_Scale Number/" + str(i) + ".PNG is loaded")
    im = cv2.imread("1_Scale Number/" + str(i) + ".PNG", cv2.IMREAD_GRAYSCALE)
    _, im = cv2.threshold(im, 128, 255, cv2.THRESH_BINARY)
    # Calculate Moments
    moments = cv2.moments(im)
    # Calculate Hu Moments
    huMoments = cv2.HuMoments(moments)
    for j in range(0, 7):
        huMoments[j] = -1 * math.copysign(1.0, huMoments[j]) * math.log10(abs(huMoments[j]))
    dist = []
    for j in range(10):
        dist.append(distance(huMoments, Label[j]))
    predictScale.append(dist.index(min(dist)))

print("The predicted labels for scale is",predictScale)
print("And the accuracy is:",accuracy_score(TrueLabel, predictScale)*100,'%')



predictRotation = []
for i in range(10):
    # print("2_Rotation Number/" + str(i) + ".PNG is loaded")
    im = cv2.imread("2_Rotation Number/" + str(i) + ".PNG", cv2.IMREAD_GRAYSCALE)
    _, im = cv2.threshold(im, 128, 255, cv2.THRESH_BINARY)
    # Calculate Moments
    moments = cv2.moments(im)
    # Calculate Hu Moments
    huMoments = cv2.HuMoments(moments)
    for j in range(0, 7):
        huMoments[j] = -1 * math.copysign(1.0, huMoments[j]) * math.log10(abs(huMoments[j]))
    dist = []
    for j in range(10):
        dist.append(distance(huMoments, Label[j]))
    predictRotation.append(dist.index(min(dist)))

print("The predicted labels for rotation is",predictRotation)
print("And the accuracy is:",accuracy_score(TrueLabel, predictRotation)*100,'%')
predictRotationScale = []
for i in range(10):
    # print("3_Scale_Rotation_Number/" + str(i) + ".PNG is loaded")
    im = cv2.imread("3_Scale_Rotation_Number/" + str(i) + ".PNG", cv2.IMREAD_GRAYSCALE)
    _, im = cv2.threshold(im, 128, 255, cv2.THRESH_BINARY)
    # Calculate Moments
    moments = cv2.moments(im)
    # Calculate Hu Moments
    huMoments = cv2.HuMoments(moments)
    for j in range(0, 7):
        huMoments[j] = -1 * math.copysign(1.0, huMoments[j]) * math.log10(abs(huMoments[j]))
    dist = []
    for j in range(10):
        dist.append(distance(huMoments, Label[j]))
    predictRotationScale.append(dist.index(min(dist)))

print("The predicted labels for scale and rotation is",predictRotationScale)
print("And the accuracy is:",accuracy_score(TrueLabel, predictRotationScale)*100,'%')