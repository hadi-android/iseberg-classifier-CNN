# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 18:23:47 2018

@author: admin
"""

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

# Any results you write to the current directory are saved as output.

import pandas as pd 
import numpy as np 
import time # to track elapsed time
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from scipy.stats import randint as sp_randint
from keras.wrappers.scikit_learn import KerasClassifier

import os
os.chdir("D:\Kaggle\Iceberg Cassification1\iceberg-classification")
np.random.seed(1337) # The seed I used - pick your own or comment out for a random seed. A constant seed allows for better comparisons though

# Import Keras 
import keras
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, Callback
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, SGD, RMSprop, Adagrad, Adadelta, Adamax, Nadam

df_train = pd.read_json('../input/train.json') # this is a dataframe

def get_scaled_imgs(df):
    imgs = []
    
    for i, row in df.iterrows():
        #make 75x75 image
        band_1 = np.array(row['band_1']).reshape(75, 75)
        band_2 = np.array(row['band_2']).reshape(75, 75)
        band_3 = band_1 + band_2 # plus since log(x*y) = log(x) + log(y)
        
        # Rescale
        a = (band_1 - band_1.mean()) / (band_1.std())
        b = (band_2 - band_2.mean()) / (band_2.std())
        c = (band_3 - band_3.mean()) / (band_3.std())

        imgs.append(np.dstack((a, b, c)))

    return np.array(imgs)

Xtrain = get_scaled_imgs(df_train)
Ytrain = np.array(df_train['is_iceberg'])

df_train.inc_angle = df_train.inc_angle.replace('na',0)
idx_tr = np.where(df_train.inc_angle>0)

x_train, x_val, y_train, y_val = train_test_split(Xtrain, Ytrain, random_state = 1, train_size=0.75)

def getModel(optimizer='adam'):
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

    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    return model

#model = getModel()
#model.summary()

model = KerasClassifier(build_fn=getModel, epochs=20, batch_size=10, verbose=2)
# define the grid search parameters
optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']

param_dist = {"batch_size":sp_randint(20,50),
              "optimizer":optimizer}

grid = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=10, verbose=2)

#batch_size = 32
earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
mcp_save = ModelCheckpoint('.mdl_wts.hdf5', save_best_only=True, monitor='val_loss', mode='min')

grid_result = grid.fit(x_train, y_train, callbacks=[earlyStopping, mcp_save])
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
# Best: 0.877805 using {'batch_size': 48, 'optimizer': 'SGD'}
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
#0.876143 (0.013091) with: {'batch_size': 43, 'optimizer': 'Adamax'}
#0.877805 (0.010774) with: {'batch_size': 48, 'optimizer': 'SGD'}
#0.868662 (0.015282) with: {'batch_size': 27, 'optimizer': 'RMSprop'}
#0.873649 (0.005124) with: {'batch_size': 27, 'optimizer': 'Adagrad'}
#0.857024 (0.017076) with: {'batch_size': 38, 'optimizer': 'Adagrad'}
#0.861180 (0.011214) with: {'batch_size': 40, 'optimizer': 'SGD'}
#0.857855 (0.017751) with: {'batch_size': 29, 'optimizer': 'Nadam'}
#0.865337 (0.024006) with: {'batch_size': 42, 'optimizer': 'Adagrad'}
#0.873649 (0.005124) with: {'batch_size': 43, 'optimizer': 'SGD'}
#0.861180 (0.014730) with: {'batch_size': 21, 'optimizer': 'Adadelta'}
#
#model.fit(x_train, y_train, batch_size=batch_size, 
#          epochs=20, verbose=1, 
#          callbacks=[earlyStopping, mcp_save], 
#          validation_data = (x_val, y_val))
#
#model.load_weights(filepath = '.mdl_wts.hdf5')
#scores = model.evaluate(x_val, y_val, verbose=0)
#print('Test loss:', scores[0])
#print('Test accuracy:', scores[1])

#Now load the test data and predict the labels
#df_test = pd.read_json('../input/test.json')
#df_test.inc_angle = df_test.inc_angle.replace('na',0)
#Xtest = (get_scaled_imgs(df_test))
#pred_test = model.predict(Xtest)
#
#submission = pd.DataFrame({'id': df_test["id"], 'is_iceberg': pred_test.reshape((pred_test.shape[0]))})
#print(submission.head(10))

#submission.to_csv('submission.csv', index=False)
