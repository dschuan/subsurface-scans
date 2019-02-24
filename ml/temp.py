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
import sklearn


item = np.load('./np_save/gridsearch0223-1633.npy')
print(item)
# print(sklearn.metrics.SCORERS.keys())
# dict_keys(['f1_macro', 'f1_samples', 'balanced_accuracy', 'neg_mean_squared_error',
# 'f1_micro', 'precision_samples', 'recall_weighted', 'recall', 'homogeneity_score',
# 'recall_micro', 'f1_weighted', 'precision_weighted', 'normalized_mutual_info_score',
# 'fowlkes_mallows_score', 'recall_samples', 'roc_auc', 'neg_log_loss', 'explained_variance',
# 'brier_score_loss', 'adjusted_mutual_info_score', 'completeness_score', 'adjusted_rand_score',
#  'neg_mean_absolute_error', 'precision', 'precision_macro', 'neg_mean_squared_log_error', 'r2',
#  'f1', 'precision_micro', 'accuracy', 'recall_macro', 'v_measure_score', 'neg_median_absolute_error',
#  'average_precision', 'mutual_info_score'])
#
# model  = keras.models.Sequential()
# model.add(keras.layers.Flatten( data_format = 'channels_last' ))
#
# model.compile(loss='mse', optimizer='adam')
#
# input = [
#     [[[1,2],[3,4]],[[4,3],[7,3]]],
#     [[[5,6],[7,8]],[[3,7],[2,5]]],
# ]
# x = np.array([input,input])
# print('x',x.shape)
#
# pred = model.predict(x)
# print('pred',pred.shape)
# print(pred)
#
# x = np.reshape(x,(x.shape[0],-1))
# print('x',x.shape)
# print(x)
