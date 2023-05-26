import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf

import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.layers import Dropout, Flatten,Activation
from keras.layers import Conv2D

from keras.layers import MaxPooling2D
from PIL import Image

from keras.preprocessing.image import ImageDataGenerator

#aPath = '--xla_gpu_cuda_data_dir=/home/jaklimczak/miniconda3/pkgs/cudatoolkit-11.8.0-h6a678d5_0/lib'

#print(aPath)
#os.environ['XLA_FLAGS'] = aPath


'''
to_create = {
    'root': 'content/dataset',
    'train_dir': 'content/dataset/training',
    'test_dir': 'content/dataset/testing',
    'a_train_dir': 'content/dataset/training/A',
    'b_train_dir': 'content/dataset/training/B',
    'c_train_dir': 'content/dataset/training/C',
    'd_train_dir': 'content/dataset/training/D',
    'e_train_dir': 'content/dataset/training/E',
    'f_train_dir': 'content/dataset/training/F',
    'g_train_dir': 'content/dataset/training/G',
    'h_train_dir': 'content/dataset/training/H',
    'i_train_dir': 'content/dataset/training/I',
    'j_train_dir': 'content/dataset/training/J',
    'k_train_dir': 'content/dataset/training/K',
    'l_train_dir': 'content/dataset/training/L',
    'm_train_dir': 'content/dataset/training/M',
    'n_train_dir': 'content/dataset/training/N',
    'o_train_dir': 'content/dataset/training/O',
    'q_train_dir': 'content/dataset/training/Q',
    'p_train_dir': 'content/dataset/training/P',
    'r_train_dir': 'content/dataset/training/R',
    's_train_dir': 'content/dataset/training/S',
    't_train_dir': 'content/dataset/training/T',
    'u_train_dir': 'content/dataset/training/U',
    'v_train_dir': 'content/dataset/training/V',
    'w_train_dir': 'content/dataset/training/W',
    'x_train_dir': 'content/dataset/training/X',
    'y_train_dir': 'content/dataset/training/Y',
    'z_train_dir': 'content/dataset/training/Z',
    'del_train_dir': 'content/dataset/training/del',
    'nothing_train_dir': 'content/dataset/training/nothing',
    'space_train_dir': 'content/dataset/training/space'      
}

for directory in to_create.values():
    try:
        os.mkdir(directory)
        print(directory, 'created')         #iterating through dictionary to make new dirs
    except:
        print(directory, 'failed')

'''

import cv2
import numpy as np

from tqdm import tqdm

batch_size = 256
img_height = 200
img_width = 200



'''
model = tf.keras.models.Sequential([
  tf.keras.layers.Input((200, 200, 3)),
  #tf.keras.layers.Conv2D(16, 3, padding='same' ),
  tf.keras.layers.Conv2D(32, 3, padding='same' ),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(10)
])
'''




'''
ds_train = tf.keras.preprocessing.image_dataset_from_directory(
    'content/dataset/training/',
    labels='inferred',
    label_mode = "int",
    color_mode='grayscale',
    batch_size=batch_size,
    image_size=(img_height, img_width), #should be forced to reshape if not this size
    shuffle=True,
    seed=2137,
    validation_split=0.1,
    subset="training"
    )

ds_validate = tf.keras.preprocessing.image_dataset_from_directory(
    'content/dataset/training/',
    labels='inferred',
    label_mode = "int",
    color_mode='grayscale',
    batch_size=batch_size,
    image_size=(img_height, img_width), #should be forced to reshape if not this size
    shuffle=True,
    seed=2137,
    validation_split=0.1,
    subset="validation"
    )
'''


'''
def augment(x, y):
        image = tf.image.random_brightness(x, max_delta=0.05)
        return image, y
ds_train = ds_train.map(augment)
'''

#for epochs in range(10):
#    for x, y in ds_train:
        #train
#        pass


train_folder = 'content/dataset/training/'
all_data = []
for folder in os.listdir(train_folder):
    
    label_folder = os.path.join(train_folder, folder)
    onlyfiles = [{'label':folder,'path':os.path.join(label_folder, f)} for f in os.listdir(label_folder) if os.path.isfile(os.path.join(label_folder, f))]
    #print(onlyfiles)
    all_data += onlyfiles
data_df = pd.DataFrame(all_data)
data_df

x_train,x_holdout = train_test_split(data_df, test_size= 0.10, random_state=42,stratify=data_df[['label']])
x_train,x_test = train_test_split(x_train, test_size= 0.25, random_state=42,stratify=x_train[['label']])


img_width, img_height = 64, 64
batch_size = 256
y_col = 'label'
x_col = 'path'
no_of_classes = len(data_df[y_col].unique())


train_datagen = ImageDataGenerator(rescale = 1/255.0)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=x_train,x_col=x_col, y_col=y_col,
    target_size=(img_width, img_height),class_mode='categorical', batch_size=batch_size,
    shuffle=False,
)

validation_datagen = ImageDataGenerator(rescale = 1/255.0)
validation_generator = validation_datagen.flow_from_dataframe(
    dataframe=x_test, x_col=x_col, y_col=y_col,
    target_size=(img_width, img_height), class_mode='categorical', batch_size=batch_size,
    shuffle=False
)

holdout_datagen = ImageDataGenerator(rescale = 1/255.0)
holdout_generator = holdout_datagen.flow_from_dataframe(
    dataframe=x_holdout, x_col=x_col, y_col=y_col,
    target_size=(img_width, img_height), class_mode='categorical', batch_size=batch_size,
    shuffle=False
)



model = Sequential()

model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',activation ='relu', input_shape = (64,64,3)))
model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2)))
model.add(Dropout(0.3))

model.add(Conv2D(filters = 64, kernel_size = (5,5),padding = 'Same',activation ='relu', input_shape = (150,150,3)))
model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2)))
model.add(Dropout(0.3))

model.add(Conv2D(filters = 64, kernel_size = (5,5),padding = 'Same',activation ='relu', input_shape = (150,150,3)))
model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2)))
model.add(Dropout(0.3))

model.add(Conv2D(filters = 128, kernel_size = (5,5),padding = 'Same',activation ='relu', input_shape = (150,150,3)))
model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2)))

model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dense(29, activation = "softmax"))




model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()


from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
early_stop = EarlyStopping(monitor='val_loss',patience=5)

batch_size=128
epochs=20

history = model.fit(train_generator,
                    epochs=epochs,
                    verbose=1,
                    validation_data=validation_generator,
                    callbacks = [early_stop])


model.save('dupa.h5')