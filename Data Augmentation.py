#   Data augmentation is process of creating new training data by processes like flipping,zooming,tilting exisiting data.
#   This process is used to prevent overfitting when available data is limited

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from tensorflow.keras import layers

#   Downloading dataset
URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
zip_path=keras.utils.get_file('cats_and_dogs.zip',URL,extract=True)
path=os.path.join(os.path.dirname(zip_path),'cats_and_dogs_filtered')

#   Getting path to dataset
train_dir = os.path.join(path, 'train')
validation_dir = os.path.join(path, 'validation')

#   Image data generator is used to load images to memory. It can also be used to augment image data 
train_generator=(ImageDataGenerator(rescale=1./255.,
                                    rotation_range=45,
                                    width_shift_range=.15,
                                    height_shift_range=.15,
                                    horizontal_flip=True,
                                    zoom_range=0.5))
test_generator=(ImageDataGenerator(rescale=1./255.))

train_data=train_generator.flow_from_directory(train_dir,batch_size=32,target_size=(160,160),class_mode='binary',shuffle=True)
test_data=test_generator.flow_from_directory(validation_dir,batch_size=32,target_size=(160,160),class_mode='binary',)


model = keras.models.Sequential([
    keras.layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=(160, 160 ,3)),
    keras.layers.MaxPooling2D(),
    #   We use dropout to reduce training parameters and prevent overfitting
    keras.layers.Dropout(0.2),
    keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    keras.layers.MaxPooling2D(),
    keras.layers.Dropout(0.2),
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dense(1,activation='sigmoid')
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])
              
history = model.fit_generator(
    train_data,
    steps_per_epoch=2000 // 32,
    epochs=15,
    validation_data=test_data,
    validation_steps=1000 // 32
)

#   Plotting training graph
loss=history.history['loss']
val_loss=history.history['val_loss']
accuracy=history.history['accuracy']
val_accuracy=history.history['val_accuracy']
epoch=[i for i in range(15)]
plt.subplot(1,2,1)
plt.plot(epoch,accuracy,label='accuracy')
plt.plot(epoch,val_accuracy,label='val.accuracy')
plt.title('accuracy vs val.accuracy')
plt.legend(loc='upper right')
plt.subplot(1,2,2)
plt.plot(epoch,loss,label='loss')
plt.plot(epoch,val_loss,label='val.loss')
plt.title('loss vs val.loss')
plt.legend(loc='upper right')
