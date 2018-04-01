# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 23:27:18 2018
@author: abhi
"""
import cv2

#apply adaptive thresholding    
def adaptiveThresholdMode(frame, x0, y0, width, height ):
    
    cv2.rectangle(frame, (x0,y0),(x0+width,y0+height),(0,255,0),1)
    roi = frame[y0:y0+height, x0:x0+width]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)  
    blur = cv2.GaussianBlur(gray,(5,5),2)
    res = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
    return res  
    
#original image without any filter    
def noFilterMode(frame, x0, y0, width, height ):
     
    cv2.rectangle(frame, (x0,y0),(x0+width,y0+height),(0,255,0),1)
    roi = frame[y0:y0+height, x0:x0+width]
    return roi   
 
# apply sift with adaptive thresholding   
def siftMode(frame, x0, y0, width, height ):
    
    cv2.rectangle(frame, (x0,y0),(x0+width,y0+height),(0,255,0),1)
    roi = frame[y0:y0+height, x0:x0+width]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)  
    blur = cv2.GaussianBlur(gray,(5,5),2)
    res = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
    img = res
    #apply sift algorithm
    sift = cv2.xfeatures2d.SIFT_create()
    kp = sift.detect(img, None)
    #draw keypoints
    img = cv2.drawKeypoints(img, kp, img, (0,0,255))
    return img 
