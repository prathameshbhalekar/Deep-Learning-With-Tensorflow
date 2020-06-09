#   Transfer learning is process of using an already trained model or part of a trained model for different model with or without 
#   adding some extra layers.

import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras

(raw_train,raw_test,raw_val),metadata=tfds.load('cats_vs_dogs',split=['train[:80%]','train[80%:90%]','train[90%:]'],
                                       with_info=True,as_supervised=True)
                                       
def format_image(image,label):
  #   Since the image is in form of int8, we convert it to float
  image=tf.cast(image,tf.float64)
  #   Normalizing
  image=image/255.0
  #   Since images have variable sizes, we resize them to a particular size
  image=tf.image.resize(image,(160,160))
  return image,label
  
train=raw_train.map(format_image)
validation=raw_val.map(format_image)
test=raw_test.map(format_image)

train_batchs=train.batch(32)
test_batchs=test.batch(32)
validation_batchs=validation.batch(32)

#   Getting the base model
base_model = tf.keras.applications.MobileNetV2(input_shape=(160,160,3),
                                               include_top=False,
                                               weights='imagenet')
base_model.trainable=False

#   As evident from the summary the base_model is a complex model which will take large data to train without overfitting
#   We overcme this using transfer learning
base_model.summary()


model=keras.models.Sequential()

model.add(base_model)
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(30,activation='relu'))
model.add(keras.layers.Dense(30,activation='relu'))
model.add(keras.layers.Dense(1),activation='sigmoid')
#   The base layer consistes only of convolution networks, thus we add some dense layers wihich we train with our limited data

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

model.fit(trauyhin_batchs.take(100),epochs=7,validation_data=validation_batchs)

_,accuracy=model.evaluate(test_batchs)
print(accuracy)
