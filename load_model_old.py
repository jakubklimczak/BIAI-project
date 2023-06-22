import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf
import pandas as pd
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import cv2
from datetime import datetime
from enum import Enum
import time


testing_folder = "testing/"

current_datetime = datetime.now()
current_datetime_str = current_datetime.strftime("%Y-%m-%d %H:%M:%S")

# Initialize the camera
camera = cv2.VideoCapture(0)  # Use 0 for the default camera

character = input("Enter what you are going to show (for correction checking): ")


print("Prepare your sign!")
print("3!")
time.sleep(1)
print("2!")
time.sleep(1)
print("1!")
time.sleep(1)

#+current_datetime_str+

path_to_save = "testing\\" + character + "\\Photo.jpg"

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

#Labels = ['A', 'B', 'C', 'D', 'del', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'nothing', 'O', 'P', 'Q', 'R', 'S', 'space', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']


all_data = []
for folder in os.listdir(testing_folder):
    
    label_folder = os.path.join(testing_folder, folder)
    onlyfiles = [{'label':folder,'path':os.path.join(label_folder, f)} for f in os.listdir(label_folder) if os.path.isfile(os.path.join(label_folder, f))]
    print(onlyfiles)
    all_data += onlyfiles
data_df = pd.DataFrame(all_data)
test_images = data_df

print(test_images)

labels = []


img_width, img_height = 64, 64
batch_size = 256
y_col = 'label'
x_col = 'path'
no_of_classes = len(test_images[y_col].unique())

for n in test_images[y_col].unique():
    print(n)
    labels.append(n)

print(labels)

validation_datagen = ImageDataGenerator(rescale = 1/255.0)
validation_generator = validation_datagen.flow_from_dataframe(
    dataframe=test_images, x_col=x_col, y_col=y_col,
    target_size=(img_width, img_height), class_mode='categorical', batch_size=batch_size,
    shuffle=False
)


loaded_model = tf.keras.models.load_model("model_deluxe_labels.h5")
#loaded_model.summary()



gowno = loaded_model.evaluate(
                    validation_generator,
                    verbose=1,
                    )


print(gowno)


'''
prediction = loaded_model.predict(validation_generator,
    verbose=0
    )
#print(prediction)

#print(len(labels))

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

