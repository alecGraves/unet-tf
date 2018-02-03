import cv2
import numpy as np

img = cv2.imread('C:\\Users\\Alec\\Desktop\\test_masks\\AOI_2_Vegas_Roads_Test_Public\\masks\\RGB-PanSharpen_AOI_2_Vegas_img9.jpeg')
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

threshold = 20
img[img < threshold] = 0 # lowpass filter
# img[img >= threshold] = 255

# edges = cv2.Canny(img, 10, 50, apertureSize=3)
# cv2.imshow('edge', edges)
# cv2.waitKey(1000)

minLineLength = 15
maxLineGap = 10
lines = cv2.HoughLinesP(img,1,np.pi/360,350,minLineLength,maxLineGap)
img[img==255] = 30

for line in lines:
    for x1,y1,x2,y2 in line:
        cv2.line(img,(x1,y1),(x2,y2),(255),1)
print(lines[2])

cv2.imshow('houghlines5.jpg',img)
cv2.waitKey(10000)