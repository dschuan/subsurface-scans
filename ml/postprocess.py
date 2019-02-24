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
import sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from keras.wrappers.scikit_learn import KerasClassifier
import keras.backend as K
import codecs
import numpy as np
from pathlib import Path



plot_totals = 50

def plot_fourd(data,figurename):
	print('plot fourd receiving data shape',data.shape)
	global plot_totals
	#pvc,wood,metal,aluminum
	colourmaps = ['Reds','Greens','Blues','Greys']

	figure = plt.figure(plot_totals)

	plot_totals = plot_totals + 1
	figure.suptitle(figurename + 'PVC:red, wood:green, metal:blue, aluminum:grey')
	ax = figure.add_subplot(111, projection='3d')
	for materialIndex ,cmap in zip(range(NUM_MATERIALS),colourmaps):
		plot_object = data[...,materialIndex+1]
		plot_threed_helper(np.squeeze(plot_object), ax = ax,figure = figure,cmap = cmap)


def plot_threed_helper(plot_object,ax ,figure ,cmap = 'coolwarm',threshold = 0.5):
	global plot_totals
	plot_totals = plot_totals + 1

	axes = plt.gca()
	axes.set_xlim([0,plot_object.shape[0]])
	axes.set_ylim([0,plot_object.shape[1]])
	axes.set_zlim([0,plot_object.shape[2]])
	x = np.arange(plot_object.shape[0])[:, None, None]
	y = np.arange(plot_object.shape[1])[None, :, None]
	z = np.arange(plot_object.shape[2])[None, None, :]
	x,y,z = np.broadcast_arrays(x,y,z)
	# c = np.tile(plot_object.ravel()[:, None], [1, 3])
	filtered_colour = []
	filtered_x = []
	filtered_y = []
	filtered_z = []
	ravel_x = x.ravel()
	ravel_y = y.ravel()
	ravel_z = z.ravel()

	for index,item in enumerate(plot_object.ravel()):
		if item > threshold:
			filtered_x.append(ravel_x[index])
			filtered_y.append(ravel_y[index])
			filtered_z.append(ravel_z[index])
			filtered_colour.append(item)

	if len(filtered_colour) > 0:

		scatter = ax.scatter(filtered_x,filtered_y,filtered_z,c=filtered_colour,cmap=plt.get_cmap(cmap),norm=mpl.colors.Normalize(vmin=-0.5, vmax=1.5))

		plt.xlabel('depth',fontsize=16)
		plt.ylabel('x',fontsize=16)
		figure.colorbar(scatter)
		# plt.clim(0,1)
	else:
		print('empty array received')



def plot_threed(plot_object,figurename = 'default',cmap = 'Blues',threshold = 0.3):
	global plot_totals
	figure = plt.figure(plot_totals)
	plot_totals = plot_totals + 1

	figure.suptitle(figurename)
	ax = figure.add_subplot(111, projection='3d')


	axes = plt.gca()
	axes.set_xlim([0,plot_object.shape[0]])
	axes.set_ylim([0,plot_object.shape[1]])
	axes.set_zlim([0,plot_object.shape[2]])
	x = np.arange(plot_object.shape[0])[:, None, None]
	y = np.arange(plot_object.shape[1])[None, :, None]
	z = np.arange(plot_object.shape[2])[None, None, :]
	x,y,z = np.broadcast_arrays(x,y,z)
	# c = np.tile(plot_object.ravel()[:, None], [1, 3])
	filtered_colour = []
	filtered_x = []
	filtered_y = []
	filtered_z = []
	ravel_x = x.ravel()
	ravel_y = y.ravel()
	ravel_z = z.ravel()

	for index,item in enumerate(plot_object.ravel()):
		if item > threshold:
			filtered_x.append(ravel_x[index])
			filtered_y.append(ravel_y[index])
			filtered_z.append(ravel_z[index])
			filtered_colour.append(item)

	if len(filtered_colour) > 0:

		scatter = ax.scatter(filtered_x,filtered_y,filtered_z,c=filtered_colour,cmap=plt.get_cmap(cmap),norm=mpl.colors.Normalize(vmin=0, vmax=1))
		plt.xlabel('depth',fontsize=16)
		plt.ylabel('x',fontsize=16)
		figure.colorbar(scatter)
		# plt.clim(0,1)
	else:
		print('empty array received')




def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    """pretty print for confusion matrixes"""
    columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
    empty_cell = " " * columnwidth

    # Begin CHANGES
    fst_empty_cell = (columnwidth-3)//2 * " " + "t/p" + (columnwidth-3)//2 * " "

    if len(fst_empty_cell) < len(empty_cell):
        fst_empty_cell = " " * (len(empty_cell) - len(fst_empty_cell)) + fst_empty_cell
    # Print header
    print("    " + fst_empty_cell, end=" ")
    # End CHANGES

    for label in labels:
        print("%{0}s".format(columnwidth) % label, end=" ")

    print()
    # Print rows
    for i, label1 in enumerate(labels):
        print("    %{0}s".format(columnwidth) % label1, end=" ")
        for j in range(len(labels)):
            cell = "%{0}.1f".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print(cell, end=" ")
        print()

def get_report(testX,trainX,testY,trainY):

	pred = predict(testX)
	pred_reshape = np.reshape(pred,(-1,20,21,OUTPUT_CHANNELS))
	labelY,labelpred = onehot_to_label(testY,pred_reshape)
	print('test stats********************************************************')
	print(sklearn.metrics.classification_report(labelY,labelpred,target_names = ['empty','pvc','wood','metal','aluminum']))
	cm = sklearn.metrics.confusion_matrix(labelY, labelpred)
	print_cm(cm, labels =  ['empty','pvc','wood','metal','aluminum'])
	print()
	print()
	pred = predict(trainX)
	pred_reshape = np.reshape(pred,(-1,20,21,OUTPUT_CHANNELS))
	labelY,labelpred = onehot_to_label(trainY,pred_reshape)
	print('train stats*******************************************************')
	print(sklearn.metrics.classification_report(labelY,labelpred,target_names = ['empty','pvc','wood','metal','aluminum']))
	cm = sklearn.metrics.confusion_matrix(labelY, labelpred)
	print_cm(cm, labels =  ['empty','pvc','wood','metal','aluminum'])
	print('macro f1',sklearn.metrics.f1_score(labelY,labelpred,average = 'macro'))
