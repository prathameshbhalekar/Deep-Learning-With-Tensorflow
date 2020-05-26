import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

#   The dataset contains 28x28 images of multiple fasion accesories
fashion_mnist = keras.datasets.fashion_mnist
(train_images,train_labels),(test_images,test_labels)=fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

#   Since the max walue of a pixel is 255, We can normalize the data by dividing it by 255
train_images=train_images/255.0
test_images=test_images/255.0

def show_image(image):
  plt.imshow(image)
  plt.colorbar()
  plt.grid(False)
  plt.show()
  
#     The first layer consists of 28^2 nodes, followed by two hidden layer followed by 9 layers which output probablity of each class
#     which is between -1 and 1 since we are using softmax activation function
model=keras.Sequential([
                        keras.layers.Flatten(input_shape=(28,28)),
                        keras.layers.Dense(128,activation='relu'),
                        keras.layers.Dense(128,activation='relu'),
                        keras.layers.Dense(10,activation='softmax')
])

#     The loss function and the optimizer are stadard for classification
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images,train_labels,epochs=10)

_,accuracy=model.evaluate(test_images,test_labels,verbose=1)
print(accuracy)

def predict(num):
  prediction=model.predict(np.array([test_images[num]]))
  print('Expected: ',class_names[test_labels[num]])
  #   We can use argmax function to get the class with highest probablity
  print('Predicted: ',class_names[np.argmax(prediction)])
  show_image(test_images[num])

predict(11)
 
