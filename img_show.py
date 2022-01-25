import cv2 
import sys

img=cv2.imread('juice.jpg', cv2.IMREAD_GRAYSCALE)

cv2.imshow('image2', img)
while True:
    if cv2.waitKey() == 27:
        break