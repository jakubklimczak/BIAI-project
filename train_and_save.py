### Imports



import os

# to prevent annoying warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}

import tensorflow as tf
import cv2
import pandas as pd
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras import activations
from tensorflow.keras.optimizers import Adam
from PIL import Image
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
from datetime import datetime

###


### for gpu on linux:

aPath = '--xla_gpu_cuda_data_dir=/home/jaklimczak/miniconda3/pkgs/cudatoolkit-11.8.0-h6a678d5_0/lib'

print(aPath)
os.environ['XLA_FLAGS'] = aPath


#setting memory growth

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    print(e)


###




batch_size = 256
image_height = 200
image_width = 200


training_location = 'content/dataset/training/'
all_images = []

for folder in os.listdir(training_location):
    label_folder = os.path.join(training_location, folder)
    all_files = [{'label':folder,'path':os.path.join(label_folder, f)} for f in os.listdir(label_folder) if os.path.isfile(os.path.join(label_folder, f))]
    all_images += all_files

dataframe = pd.DataFrame(all_images)
dataframe

# returns list of len[2], left is always training data, right is test. default test_size = 0.25. if random_state is set to an integer, the results are replicable
'''random_state = 2'''
x_training,x_holdout = train_test_split(dataframe, test_size = 0.10, stratify = dataframe[['label']])
x_training,x_testing = train_test_split(x_training, stratify = x_training[['label']])


image_width, image_height = 64, 64
batch_size = 256
y_col = 'label'
x_col = 'path'
number_of_classes = len(dataframe[y_col].unique())


training_datagen = ImageDataGenerator(rescale = 1/255.0)

training_generator = training_datagen.flow_from_dataframe(
    dataframe = x_training, x_col=x_col, y_col=y_col,
    target_size = (image_width, image_height), class_mode='categorical', batch_size=batch_size,
    shuffle = False,
)

validation_datagen = ImageDataGenerator(rescale = 1/255.0)

validation_generator = validation_datagen.flow_from_dataframe(
    dataframe=x_testing, x_col=x_col, y_col=y_col,
    target_size=(image_width, image_height), class_mode='categorical', batch_size=batch_size,
    shuffle=False
)

holdout_datagen = ImageDataGenerator(rescale = 1/255.0)
holdout_generator = holdout_datagen.flow_from_dataframe(
    dataframe=x_holdout, x_col=x_col, y_col=y_col,
    target_size=(image_width, image_height), class_mode='categorical', batch_size=batch_size,
    shuffle=False
)



model = Sequential()

model.add(tf.keras.layers.Conv2D(filters = 32, kernel_size = (5,5), padding = 'Same', activation = activations.relu, input_shape = (64,64,3)))
model.add(tf.keras.layers.MaxPooling2D(pool_size = (2,2), strides = (2,2)))
model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Conv2D(filters = 64, kernel_size = (5,5), padding = 'Same', activation = activations.relu, input_shape = (150,150,3)))
model.add(tf.keras.layers.MaxPooling2D(pool_size = (2,2), strides = (2,2)))
model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Conv2D(filters = 64, kernel_size = (5,5), padding = 'Same', activation = activations.relu, input_shape = (150,150,3)))
model.add(tf.keras.layers.MaxPooling2D(pool_size = (2,2), strides = (2,2)))
model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Conv2D(filters = 128, kernel_size = (5,5), padding = 'Same', activation = activations.relu, input_shape = (150,150,3)))
model.add(tf.keras.layers.MaxPooling2D(pool_size = (2,2), strides = (2,2)))

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(256))
model.add(tf.keras.layers.Activation(activations.relu))
model.add(tf.keras.layers.Dense(29, activations.softmax))



#Adam learning_rate default 0.001

model.compile(optimizer = tf.keras.optimizers.Adam(), loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.summary()


early_stop = EarlyStopping(monitor='val_loss', patience=5)

batch_size=128
epochs=20

logdir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
options = tf.profiler.experimental.ProfilerOptions(host_tracer_level = 3, python_tracer_level = 1, device_tracer_level = 1)

print(tf.config.list_logical_devices('GPU'))

tf.profiler.experimental.start(logdir = logdir, options = options)

history = model.fit(training_generator,
                    epochs = epochs,
                    verbose = 1,
                    validation_data = validation_generator,
                    callbacks = [early_stop])


#print(history.history.keys())

tf.profiler.experimental.stop()

model.save('model.h5')



'''
@tf.function
def traceme(x):
    return model(x)

logdir = "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
writer = tf.summary.create_file_writer(logdir)
tf.summary.trace_on(graph=True, profiler=True)

traceme(tf.zeros((0, 64, 64, 3)))
with writer.as_default():
    tf.summary.trace_export(name="model_trace", step=0, profiler_outdir=logdir)
'''