import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator

testing_folder = "testing/"

all_data = []
for folder in os.listdir(testing_folder):
    
    label_folder = os.path.join(testing_folder, folder)
    onlyfiles = [{'label':folder,'path':os.path.join(label_folder, f)} for f in os.listdir(label_folder) if os.path.isfile(os.path.join(label_folder, f))]
    #print(onlyfiles)
    all_data += onlyfiles
data_df = pd.DataFrame(all_data)
test_images = data_df


img_width, img_height = 64, 64
batch_size = 256
y_col = 'label'
x_col = 'path'
no_of_classes = len(test_images[y_col].unique())

validation_datagen = ImageDataGenerator(rescale = 1/255.0)
validation_generator = validation_datagen.flow_from_dataframe(
    dataframe=test_images, x_col=x_col, y_col=y_col,
    target_size=(img_width, img_height), class_mode='categorical', batch_size=batch_size,
    shuffle=True
)


loaded_model = tf.keras.models.load_model("model.h5")
loaded_model.summary()



loaded_model.evaluate(
                    validation_generator,
                    verbose=1,
                    )