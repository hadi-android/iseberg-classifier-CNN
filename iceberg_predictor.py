# -*- coding: utf-8 -*-
"""
Created on Sat Dec 16 17:12:29 2017

@author: admin
"""

import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
np.random.seed(1337) # The seed I used - pick your own or comment out for a random seed. A constant seed allows for better comparisons though

# Import Keras 
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation, Merge
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam

import os
os.chdir("D:\Kaggle\Iceberg Cassification1\iceberg-classification")

df_train = pd.read_json('train.json') # this is a dataframe

def get_scaled_imgs(df):
    imgs = []
    
    for i, row in df.iterrows():
        #make 75x75 image
        band_1 = np.array(row['band_1']).reshape(75, 75)
        band_2 = np.array(row['band_2']).reshape(75, 75)
        band_3 = band_1 + band_2 # plus since log(x*y) = log(x) + log(y)
        
        # Rescale
#        a = (band_1 - band_1.mean()) / (band_1.max() - band_1.min())
#        b = (band_2 - band_2.mean()) / (band_2.max() - band_2.min())
#        c = (band_3 - band_3.mean()) / (band_3.max() - band_3.min())
        a = (band_1 - band_1.mean()) / (band_1.std())
        b = (band_2 - band_2.mean()) / (band_2.std())
        c = (band_3 - band_3.mean()) / (band_3.std())

        imgs.append(np.dstack((a, b, c)))

    return np.array(imgs)

Xtrain = get_scaled_imgs(df_train)
Ytrain = np.array(df_train['is_iceberg'])

df_train.inc_angle = df_train.inc_angle.replace('na',0)
df_train.inc_angle = df_train.inc_angle.astype(float).fillna(0.0)
X_angle_train = np.array(df_train.inc_angle)

x_train, x_val, y_train, y_val = train_test_split(Xtrain, Ytrain, random_state = 1, train_size=0.75)


def getModel():
    #Build keras model
    
    model=Sequential()
    
    # CNN 1
    model.add(Conv2D(64, kernel_size=(3, 3),activation='relu', input_shape=(75, 75, 3)))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2),  dim_ordering="th"))
    model.add(Dropout(0.2))

    # CNN 2
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu' ))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2),  dim_ordering="th"))
    model.add(Dropout(0.2))

    # CNN 3
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2),  dim_ordering="th"))
    model.add(Dropout(0.2))

    #CNN 4
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2),  dim_ordering="th"))
    model.add(Dropout(0.2))

    # You must flatten the data for the dense layers
    model.add(Flatten())

    #Dense 1
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))

    #Dense 2
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))

    # Output 
    model.add(Dense(1, activation="sigmoid"))

    optimizer = Adam(lr=0.001, decay=0.0)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    return model

model = getModel()
model.summary()

batch_size = 32
earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
mcp_save = ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='val_loss', mode='min')

model.fit(x_train, y_train, batch_size=batch_size, 
          epochs=20, verbose=1, 
          callbacks=[earlyStopping, mcp_save], 
          validation_data = (x_val, y_val))

model.load_weights(filepath = '.mdl_wts.hdf5')
scores = model.evaluate(x_val, y_val, verbose=0)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

#Now load the test data and predict the labels
df_test = pd.read_json('test.json')
df_test.inc_angle = df_test.inc_angle.replace('na',0)
Xtest = (get_scaled_imgs(df_test))
pred_test = model.predict(Xtest)

submission = pd.DataFrame({'id': df_test["id"], 'is_iceberg': pred_test.reshape((pred_test.shape[0]))})
print(submission.head(10))

submission.to_csv('submission.csv', index=False)
