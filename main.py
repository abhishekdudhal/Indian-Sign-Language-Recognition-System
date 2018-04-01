# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 16:55:41 2018
@author: abhi
"""

import cv2
import cnnPredict
import cnnFilters
import cnnModel
import time

x0 = 400
y0 = 200
height = 200
width = 200
isBgModeOn = 0
isAdaptiveThresholdMode = True
roi = None
isPredictionMode = False
model= None
menu = "\n c-Change Filter\n p-Predict Sign\n w-Move ROI Upside\n s-Move ROI Downside\n a-Move ROI Rightside\n d-Move ROI Leftside\n ESC-exit\n"



def Main():
    
      global isAdaptiveThresholdMode, isBgModeOn,x0,y0,roi,isPredictionMode,model,menu
      cap = cv2.VideoCapture(0)
      ret = cap.set(3,640)
      ret = cap.set(4,480)
      print(menu)  
      while(True):
        ret, frame = cap.read()
        #invert frame
        frame = cv2.flip(frame, 3)

        if ret == True:
            if isBgModeOn == 0:
                roi = cnnFilters.adaptiveThresholdMode(frame, x0, y0, width, height)
            elif isBgModeOn == 1:
                roi = cnnFilters.siftMode(frame, x0, y0, width, height)
            else :
                 roi = cnnFilters.noFilterMode(frame, x0, y0, width, height)
            if isPredictionMode :
                 result = cnnPredict.predictSign(roi,model)
                 cv2.putText(frame,result ,(10,355 + 108), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,255,0),1,1)
                 time.sleep(0.04 )
 
        cv2.imshow('Sign Language Detactor',frame)
        cv2.imshow('ROI',roi)
        key = cv2.waitKey(10) & 0xff

        if key == ord('c'):
            if(isBgModeOn == 2):
                isBgModeOn = 0
            else:    
                isBgModeOn = isBgModeOn+1
            if isBgModeOn == 0:
                print ("Adaptive Threshold Mode active")
            elif isBgModeOn == 1:
                print ("Sift Mode active")
            else :
                print ("No Filter Mode active")
            if isPredictionMode:
                   model=cnnModel.createCNNModel(isBgModeOn)
        elif key == ord('p'):
               isPredictionMode = not isPredictionMode
               if isPredictionMode:
                   model=cnnModel.createCNNModel(isBgModeOn)
               print ("Prediction Mode - {}".format(isPredictionMode))
        elif key == ord('w'):
            y0 = y0 - 5
        elif key == ord('s'):
            y0 = y0 + 5
        elif key == ord('a'):
            x0 = x0 - 5
        elif key == ord('d'):
            x0 = x0 + 5
        elif key == 27:
            break;
      cap.release()
      cv2.destroyAllWindows()



if __name__ == "__main__":
    Main()

