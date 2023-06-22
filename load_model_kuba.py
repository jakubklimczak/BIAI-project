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


testing_folder = "testing/"

current_datetime = datetime.now()
current_datetime_str = current_datetime.strftime("%Y-%m-%d %H:%M:%S")

# Initialize the camera
camera = cv2.VideoCapture(0)  # Use 0 for the default camera

#character = input("Enter what you are going to show (for correction checking): ")
#character

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

Labels = []
labels_path = "labels.txt"

with open(labels_path, "r") as file:
    for line in file:
        value = line.strip()  # Remove leading/trailing whitespaces or newlines
        Labels.append(value)  # Add the value to the array

# Print the array to verify the result
print(Labels)


loaded_model = tf.keras.models.load_model("model_deluxe_labels.h5")

for filename in os.listdir(testing_folder):
    # Construct the full path of the image
    image_path = os.path.join(testing_folder, filename)
    
    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(64, 64))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image
    
    # Make predictions using the model
    predictions = loaded_model.predict(img_array)
    
    # Get the predicted class
    predicted_class = np.argmax(predictions)
    
    # Print the filename and predicted class
    print("Image:", filename)
    print("Predicted class:", predicted_class)
    print("Letter:", Labels[predicted_class])
    print()  # Add a line break between images




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


