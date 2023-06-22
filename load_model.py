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


testing_folder = "testing/"

current_datetime = datetime.now()
current_datetime_str = current_datetime.strftime("%Y-%m-%d %H:%M:%S")

# Initialize the camera
camera = cv2.VideoCapture(0)  # Use 0 for the default camera
detector = HandDetector(maxHands=1)

loaded_model = tf.keras.models.load_model("model_hands_labels.h5")

occurencies = {'A':0, 'B':0, 'C':0, 'D':0, 'E':0, 'F':0, 'G':0, 'H':0, 'I':0, 'J':0, 'K':0, 'L':0, 'M':0, 'N':0, 'O':0, 'P':0, 'Q':0, 'R':0, 'S':0, 'T':0, 'U':0, 'V':0, 'W':0, 'X':0, 'Y':0, 'Z':0, 'del':0, 'nothing':0, 'space': 0}

alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']

font = cv2.FONT_HERSHEY_SIMPLEX
org = (50, 50)

fontScale = 1
color = (0, 0, 0)
thickness = 2


imgSize = 350
folder = "dataset/"

last_photo = time.time()
photo_delay = 0.500



while True:
    success, img = camera.read()
    hands = detector.findHands(img, draw=False)
    if hands:
        hand = hands[0]
        x,y,w,h = hand['bbox']
        if x > 10 and y > 10:
            imgWhite = np.ones((imgSize,imgSize,3),np.uint8)*255

            imgCrop = img[y-10:y+h+10,x-10:x+w+10]

            imgCropShape = imgCrop.shape

            aspectRatio = h/w
            try:
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
            except:
                print("dupsko tłuste")
            cv2.imshow("ImageCrop",imgCrop)
            cv2.imshow("ImageCropWhite",imgWhite)
            #print(imgWhite.shape)
            if imgWhite.shape == (350, 350, 3):
                imgWhite = imgWhite[np.newaxis, ...]
                predictions = loaded_model.predict([imgWhite], verbose = 0)
                predicted_class = np.argmax(predictions)
                #print("Predicted class:", predicted_class)
                #print("Alphabet:", alphabet[predicted_class])
                img = cv2.putText(img, alphabet[predicted_class], org, font, 
                   fontScale, color, thickness, cv2.LINE_AA)



    
    
    cv2.imshow("Image",img)
    key = cv2.waitKey(1)
    current_time = time.time()
    
    
    if key == ord("s") and last_photo+photo_delay<current_time:
        last_photo = current_time
        cv2.imwrite(f'{folder}/Image_{current_time}.jpg',imgWhite)

    if key == ord("c"):
        break

#    if key == ord("dupa"):
#        słowo zrób





#character = input("Enter what you are going to show (for correction checking): ")
#character


'''
print("Prepare your sign!")
print("3!")
time.sleep(1)
print("2!")
time.sleep(1)
print("1!")
time.sleep(1)

#+current_datetime_str+

current_datetime_spaceless_str = current_datetime_str.replace(" ", "_")
current_datetime_spaceless_str = current_datetime_spaceless_str.replace(":", "-")

path_to_save = "testing\\Photo" + current_datetime_spaceless_str + ".jpg"

if not camera.isOpened():
    print("Failed to open camera")
else:
    ret, frame = camera.read()
    if not ret:
        print("Failed to capture frame")
    else:
        cv2.imwrite(path_to_save, frame)
        print("Photo successfully saved and added to testing set at location: " + path_to_save)
    camera.release()

'''

#Labels = []
#labels_path = "labels.txt"

#debil_alphabet = ['A', 'B', 'C']


#with open(labels_path, "r") as file:
#    for line in file:
#        value = line.strip()  # Remove leading/trailing whitespaces or newlines
#        Labels.append(value)  # Add the value to the array

# Print the array to verify the result
#print(Labels)




#for filename in os.listdir(folder): #testing_folder
    # Construct the full path of the image
#    image_path = os.path.join(folder, filename) #testing_folder
    
    # Load and preprocess the image
#    img = image.load_img(image_path, target_size=(350, 350))
#    img_array = image.img_to_array(img)
#    img_array = np.expand_dims(img_array, axis=0)
#    img_array /= 255.0  # Normalize the image
    
    # Make predictions using the model




#loaded_model.summary()



'''
prediction = loaded_model.predict(validation_generator,
    verbose=0
    )
'''
'''
for i in range(len(prediction)):
    print(i)
    print(np.argmax(prediction[i]))

    print(labels[np.argmax(prediction[i])])
    print(all_data[i]['label'])
    if all_data[i]['label'] == labels[np.argmax(prediction[i])]:
        print("Correct")
    else:
        print("Bad")
    print("")
'''


