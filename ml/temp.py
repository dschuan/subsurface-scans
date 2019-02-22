import math
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pickle
import random
import json
import glob
import os
from mpl_toolkits.mplot3d import Axes3D
from processArray import processArray
from datetime import datetime
import itertools
import copy
import time
from filters import create_filter
import keras
from preprocess import loadData
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier




model  = keras.models.Sequential()
model.add(keras.layers.Flatten( data_format = 'channels_last' ))

model.compile(loss='mse', optimizer='adam')

input = [
    [[[1,2],[3,4]],[[4,3],[7,3]]],
    [[[5,6],[7,8]],[[3,7],[2,5]]],
]
x = np.array([input,input])
print('x',x.shape)

pred = model.predict(x)
print('pred',pred.shape)
print(pred)

x = np.reshape(x,(x.shape[0],-1))
print('x',x.shape)
print(x)
