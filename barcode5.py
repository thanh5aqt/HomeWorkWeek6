#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 23:05:56 2019

@author: tttt
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 21:22:09 2019

@author: tttt
"""
import matplotlib.pyplot as plt
import imutils
import numpy as np
import cv2

image = cv2.imread("./input/05.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.imshow(gray, cmap='gray')
plt.title('Input Photo')

gradX = cv2.Sobel(gray, cv2.CV_32F, dx=1, dy=0, ksize=-1)
gradY = cv2.Sobel(gray, cv2.CV_32F, dx=0, dy=1, ksize=-1) 

gradient = cv2.add(gradX, gradY)
gradient = cv2.convertScaleAbs(gradient)

blurred = cv2.blur(gradient, (11, 11))

(_, thresh) = cv2.threshold(blurred, 175, 255, cv2.THRESH_BINARY)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (63, 21))
closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)


closed = cv2.erode(closed, None, iterations = 10)
closed = cv2.dilate(closed, None, iterations = 10)


contours = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)
c = sorted(contours, key = cv2.contourArea, reverse = True)[0]
 
rect = cv2.minAreaRect(c)
box = cv2.boxPoints(rect)
box = np.int0(box)

cv2.drawContours(image, [box], -1, (0, 255, 0), 10)
plt.figure()
plt.imshow(image)
plt.title('Result')

cv2.imwrite('./output/05.jpg', image[:,:,::-1])










