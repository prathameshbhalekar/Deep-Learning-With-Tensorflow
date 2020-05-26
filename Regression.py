import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.model_selection import train_test_split

dataset_path = keras.utils.get_file("auto-mpg.data", "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight',
                'Acceleration', 'Model Year', 'Origin']
df=pd.read_csv(dataset_path,names=column_names,sep=' ',na_values='?',skipinitialspace=True,comment='\t')
df.dropna(subset=['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight','Acceleration', 'Model Year', 'Origin'],inplace=True)

#   Converting origin to one hot encoding
df['Europe']=0
df['USA']=0
df['Japan']=0
for i in range(l):
    if(df.Origin.get(i)==1):
      df.at[i,'USA']=1
    if(df.Origin.get(i)==2):
      df.at[i,'Europe']=1
    if(df.Origin.get(i)==3):
      df.at[i,'Japan']=1

df=df.drop("Origin",1)

train_featureset,test_featureset=train_test_split(df,test_size=0.2)
train_values=train_featureset.MPG
train_featureset=train_featureset.drop("MPG",1)
test_values=test_featureset.MPG
test_featureset=test_featureset.drop("MPG",1)

train_featureset=(train_featureset-train_featureset.mean())/train_featureset.std()
test_featureset=(test_featureset-test_featureset.mean())/test_featureset.std()

#   Using two hidden layers along with one input layer consisting of 10 features and the output layer having linear activation function
model=keras.Sequential(layers=[keras.layers.Dense(64,activation='relu',input_shape=[len(test_featureset.columns)]),
                               keras.layers.Dense(64,activation='relu'),                               
                               keras.layers.Dense(1)])

#   rms prop is standard optimizer for regression and mean sq. error or mean abs. error can be used as loss functions
optimizer = tf.keras.optimizers.RMSprop(0.001)
model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])


#   Using early stoping to prevent overfitting
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

model.fit(train_featureset, train_values, 
                    epochs=1000, validation_split = 0.2, verbose=0, 
                    callbacks=[early_stop, tfdocs.modeling.EpochDots()])
loss, mae, mse = model.evaluate(test_featureset, test_values, verbose=2)

print(loss, mae, mse)
