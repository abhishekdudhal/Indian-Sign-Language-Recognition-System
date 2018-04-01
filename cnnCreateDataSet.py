# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 01:09:23 2018
@author: abhi
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Jan 21 16:55:41 2018
@author: abhi
"""

import cv2
import cnnPredict
import cnnFilters
import cnnTrain
import cnnModel
import cnnCreateDataSet
import time
import os

x0 = 400
y0 = 200
height = 200
width = 200
isBgModeOn = 0
isAdaptiveThresholdMode = True
roi = None
isPredictionMode = False
model= None
menu = "\n c-Change Filter\n p-Predict Sign\n n-Save ROI in data set \n q-Close ROI Window \n w-Move ROI Upside\n s-Move ROI Downside\n a-Move ROI Rightside\n d-Move ROI Leftside\n ESC-exit\n"

#remove background

def Main():
    global isAdaptiveThresholdMode, isBgModeOn,x0,y0,roi,isPredictionMode,model,menu
    isQuit=0
    cap = cv2.VideoCapture(0)
    ret = cap.set(3,640)
    ret = cap.set(4,480)
    i=0
    j=0
    #sign array should be in order 
    signnamearray = ["Aboard", "Baby", "Bowl","Friend"," House" ,"IorMe","Money","Opposite","Prisoner","You"]
    while(True):
        ret, frame = cap.read()
        #invert frame
        frame = cv2.flip(frame, 3)

        roi1 = cnnFilters.adaptiveThresholdMode(frame, x0, y0, width, height)
        roi2 = cnnFilters.siftMode(frame, x0, y0, width, height)
        roi3 = cnnFilters.noFilterMode(frame, x0, y0, width, height)
        cv2.imshow('Sign Language Detactor',frame)

        if not isQuit:
            cv2.imshow('ROI1', roi1)
            cv2.imshow('ROI2', roi2)
            cv2.imshow('ROI3', roi3)

        key = cv2.waitKey(10) & 0xff

        if key == ord('n'):
            '''
            signname = input("Enter a sign name \n")
            '''
            signname = signnamearray[i]
            path1="./AdaptiveThresholdModeDataSet/"
            path2="./SiftModeDataSet/"
            path3="./NoFilterModeDataSet/"
            ts = int(time.time())
            name = signname + str(ts)
            print ("creating image...")
            cv2.imwrite(path1+name + str(j) + "1.png", roi1)
            cv2.imwrite(path2+name + str(j) + "2.png", roi2)
            cv2.imwrite(path3+name + str(j) + "3.png", roi3)
            print ("created image: "+str(signname)+ " " + str(j) + " for word " + str(signname))
            j=j+1
            #number of images per sign
            if(j==10):
              i=i+1
              j=0
              os.system('spd-say "Change The Gesture"')
              #Total number of signs
              if(i==10):
                  break
            time.sleep(0.04 )
        elif key == ord('q'):
             isQuit = not isQuit
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

