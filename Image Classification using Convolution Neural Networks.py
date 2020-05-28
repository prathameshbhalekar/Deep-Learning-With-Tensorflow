'''
    Convolution neural network detect features globally unlike their traditional conterparts which look for feature at a particular
    part of data which makes them suitable for image recognition.
'''


import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt 
from tensorflow.keras import layers,models,datasets

(train_featureset,train_labels),(test_featureset,test_labels)=datasets.mnist.load_data()
#   We need an original copy of images as we will normalize images
og_featureset=test_featureset.copy()

#   Generally image data has shape x*y*3 representing R,G,B values. But since this is a greyscale image we only have single value.
#   Therefore we need to reshape data for tensorflow to accept it.

train_featureset=train_featureset.reshape(60000,28,28,1)
test_featureset=test_featureset.reshape(10000,28,28,1)

#   Normalizing
train_featureset=train_featureset/255.0
test_featureset=test_featureset/255.0


model = models.Sequential()
'''   
    The 32 here specifies the number of channels, where each channel is some function of the inital RGB channel which gives
    information about some different combination of RGB.
      
    (3,3) represents the size of filter.
'''

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28,1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10,activation='softmax'))

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

#   Training
model.fit(train_featureset,train_labels,epochs=10)

#   Testing
_,accuracy=model.evaluate(test_featureset,test_labels)
print(accuracy)

#   Predicting
def predict(num):
    prediction=model.predict(np.array([test_featureset[num]]))
    plt.imshow(og_featureset[num])
    plt.colorbar()
    plt.grid(False)
    plt.show()
    print(np.argmax(prediction))

predict(23)
