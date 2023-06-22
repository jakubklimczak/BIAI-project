import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf
import pandas as pd
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import cv2
from datetime import datetime
from enum import Enum
import time

#moje
from cvzone.HandTrackingModule import HandDetector
import math


all_images = []
training_location = "content/dataset/training"
imgSize = 350
detector = HandDetector(mode=True, maxHands=1)


for folder in os.listdir(training_location):
    label_folder = os.path.join(training_location, folder)
    all_files = [{'label':folder,'path':os.path.join(label_folder, f)} for f in os.listdir(label_folder) if os.path.isfile(os.path.join(label_folder, f))]
    all_images += all_files
print("Found "+str(len(all_images))+" images")


for index,filename in enumerate(all_images):
    #print("Processing photo #"+str(index))
    img = cv2.imread(filename['path'],1)

    try:
        hands = detector.findHands(img, draw=False)
        if hands:
            hand = hands[0]
            x,y,w,h = hand['bbox']
            if x > 10 and y > 10:
                imgWhite = np.ones((imgSize,imgSize,3),np.uint8)*255

                imgCrop = img[y-10:y+h+10,x-10:x+w+10]

                imgCropShape = imgCrop.shape

                aspectRatio = h/w

                if aspectRatio > 1:
                    k = imgSize/h
                    wCal = math.ceil(k*w)
                    imgResize = cv2.resize(imgCrop,(wCal,imgSize))
                    imgResizeShape = imgResize.shape
                    wGap = math.ceil((imgSize-wCal)/2)
                    imgWhite[:,wGap:wCal+wGap] = imgResize
                else:
                    k = imgSize/w
                    hCal = math.ceil(k*h)
                    imgResize = cv2.resize(imgCrop,(imgSize,hCal))
                    imgResizeShape = imgResize.shape
                    hGap = math.ceil((imgSize-hCal)/2)
                    imgWhite[hGap:hCal+hGap,:] = imgResize
                #cv2.imshow("ImageCropWhite",imgWhite)
                cv2.imwrite(filename['path'],imgWhite)
    except: 
        print("dupa")
            #print("Saving photo #"+str(index)+"as "+filename['path'])
    #else:
        #print("Failed to find hand on photo #"+str(index)+" aka: "+filename['path'])
    key = cv2.waitKey(1)