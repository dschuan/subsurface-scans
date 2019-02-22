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
import keras.backend as K

epochs = 1000
OUTPUT_CHANNELS = 4
tf.set_random_seed(10)
seed = 10
np.random.seed(seed)

def variable_summaries(var):
	"""Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
	with tf.name_scope('summaries'):
		mean = tf.reduce_mean(var)
		tf.summary.scalar('mean', mean)
		with tf.name_scope('stddev'):
			stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
		tf.summary.scalar('stddev', stddev)
		tf.summary.scalar('max', tf.reduce_max(var))
		tf.summary.scalar('min', tf.reduce_min(var))
		tf.summary.histogram('histogram', var)


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
	for materialIndex ,cmap in zip(range(4),colourmaps):
		plot_object = data[...,materialIndex]
		plot_threed_helper(np.squeeze(plot_object), ax = ax,figure = figure,cmap = cmap)


def plot_threed_helper(plot_object,figurename = 'default',ax = '',figure = plt.figure(plot_totals),cmap = 'coolwarm',threshold = 0.3):
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

def f1(y_true, y_pred):
    y_pred = K.round(y_pred)
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    # tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0)

    p = tp / (tp + fp + K.epsilon())
    r = tp / (tp + fn + K.epsilon())

    f1 = 2*p*r / (p+r+K.epsilon())
    f1 = tf.where(tf.is_nan(f1), tf.zeros_like(f1), f1)
    return K.mean(f1)

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

def create_model(hidden_layers = [(5,10),(5,10),(5,10),(5,OUTPUT_CHANNELS)], filters = 10, activation = 'relu', kernel_initializer = 'glorot_uniform',dropout_rate = 0.0,batch_norm = False,loss='mse', optimizer='rmsprop', metrics=["accuracy"]  ):

	layer_params = {
		"filters": 10,
		"kernel_size": (5,5,5),
		"padding":'same',
		"data_format":'channels_last',
		"activation":'relu',
		"use_bias":True,
		"kernel_initializer":'glorot_uniform',
		"bias_initializer":'zeros',
		# "kernel_regularizer":keras.regularizers.l2(0.01)

	}

	model = keras.models.Sequential()
	for index, layer in enumerate(hidden_layers):
		kernel_size, num_filters = layer
		layer_params["kernel_size"] = (kernel_size,kernel_size,kernel_size)
		layer_params["filters"] = num_filters
		layer_params["activation"] = activation
		layer_params["kernel_initializer"] = kernel_initializer

		lastlayer = (index == len(hidden_layers) - 1)
		# last layer enforce filter size = OUTPUT_CHANNELS
		if lastlayer:
			layer_params["activation"] = None
			layer_params["filters"] = OUTPUT_CHANNELS


		model.add(keras.layers.Conv3D(**layer_params))

		if not lastlayer:

			if batch_norm:
				model.add(keras.layers.BatchNormalization())
			if dropout_rate:
				model.add(keras.layers.Dropout(p = dropout_rate))

	model.add(keras.layers.Flatten( data_format = 'channels_last' ))
	model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

	return model


def run(trainX, testX, trainY, testY):
	model  = keras.models.Sequential()

	layer1_params = {
		"filters": 10,
		"kernel_size": (5,5,5),
		"padding":'same',
		"data_format":'channels_last',
		"activation":'relu',
		"use_bias":True,
		"kernel_initializer":'glorot_uniform',
		"bias_initializer":'zeros',
		# "kernel_regularizer":keras.regularizers.l2(0.01)

	}


	layer1 = keras.layers.Conv3D(**layer1_params)
	model.add(layer1)

	layer2_params = {
		"filters": 10,
		"kernel_size": (5,5,5),
		"padding":'same',
		"data_format":'channels_last',
		"activation":'relu',
		"use_bias":True,
		"kernel_initializer":'glorot_uniform',
		"bias_initializer":'zeros',
		# "kernel_regularizer":keras.regularizers.l2(0.01)

	}
	layer2 = keras.layers.Conv3D(**layer2_params)
	model.add(layer2)

	layer3_params = {
		"filters": 10,
		"kernel_size": (5,5,5),
		"padding":'same',
		"data_format":'channels_last',
		"activation":'relu',
		"use_bias":True,
		"kernel_initializer":'glorot_uniform',
		"bias_initializer":'zeros',
		# "kernel_regularizer":keras.regularizers.l2(0.01)

	}
	layer3 = keras.layers.Conv3D(**layer3_params)
	model.add(layer3)

	layer_final_params = {
		"filters": OUTPUT_CHANNELS,
		"kernel_size": (5,5,5),
		"padding":'same',
		"data_format":'channels_last',
		"activation":None,
		"use_bias":True,
		"kernel_initializer":'glorot_uniform',
		"bias_initializer":'zeros',
		# "kernel_regularizer":keras.regularizers.l2(0.01)

	}
	layerfinal = keras.layers.Conv3D(**layer_final_params)
	model.add(layerfinal)

	model.compile(optimizer='rmsprop',loss='mse',metrics=['accuracy'])

	model.fit(trainX, trainY, epochs=epochs, batch_size=4)
	score = model.evaluate(testX, testY, verbose=1)
	print(score)
	print('Test loss:', score[0])
	print('Test accuracy:', score[1])
	save_dir = './model/keras_model.h5'
	model.save(save_dir)
	print('model saved at',save_dir)

def predict(data):
	model = keras.models.load_model('./model/keras_model.h5')
	pred = model.predict(data)
	return pred

def grid_search(X,Y):

	model = KerasClassifier(build_fn=create_model, verbose=1)
	# define the grid search parameters
	batch_size = [1,2,4]
	epochs = [100, 500, 1000]
	hidden_layers = [
		[(5,10),(5,10),(5,10),(5,10),(5,OUTPUT_CHANNELS)],
		[(5,10),(5,10),(5,10),(5,OUTPUT_CHANNELS)],
		[(5,10),(5,10),(5,OUTPUT_CHANNELS)],
		[(5,10),(5,OUTPUT_CHANNELS)],
		[(5,OUTPUT_CHANNELS)]
		]
	filters = [3,5,7,9]
	activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
	kernel_initializer = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
	dropout_rate = [0.0,0.2,0.4,0.6]
	batch_norm = [True, False]
	loss=['mean_squared_error','mean_absolute_error','mean_absolute_percentage_error','mean_squared_logarithmic_error','squared_hinge','hinge','logcosh','categorical_crossentropy','binary_crossentropy','kullback_leibler_divergence','poisson','cosine_proximity']
	optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']

	param_grid = dict(batch_size=[2], epochs=[500],loss=loss)

	grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1)
	grid_result = grid.fit(X,Y)
	# summarize results
	print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
	means = grid_result.cv_results_['mean_test_score']
	stds = grid_result.cv_results_['std_test_score']
	params = grid_result.cv_results_['params']
	for mean, stdev, param in zip(means, stds, params):
		print("%f (%f) with: %r" % (mean, stdev, param))



if __name__ == '__main__':

	model_params = {
		'backproj_filter_size': 20,
		'backproj_filter_depth': 20,
		# 'layerlist': [NUM_CHANNELS,10,20,OUTPUT_CHANNELS],
		'keep': 1.0, #dropout rate for first cnn layer
		'filter_height': 1,
		'filter_size' : 5,
		'use_saved_model':False, #when true, uses the previous model and continue training
		'exp_name': 'backprop_norm_drop1',
		'num_backproj_filter': 1,
		'show_results': False #when true, uses the previously trained model and shows results of test, overrides use saved model to true

	}

	data_params = {

		'reload': False, #When True, parse time domain raw data again, use when data changes
		'max_items_per_scan': 2, # maximum number of items in a scanf
		'train_test_split': 0.7, #size of training data
		'only_max': False,
		'saved_path': "../new_res/*.json"
	}
	# reload_data()

	trainX, testX, trainY, testY = loadData(**data_params)

	combinedX = np.concatenate((trainX,testX),axis = 0)

	combinedY = np.concatenate((trainY,testY),axis = 0)# (34, 40, 20, 21, 4)
	combinedY = np.reshape(combinedY,(combinedY.shape[0],-1))

	# grid_search(combinedX,combinedY)

	# model = create_model()
	# trainY_flat = np.reshape(trainY,(trainY.shape[0],-1))
	# testY_flat = np.reshape(testY,(testY.shape[0],-1))
	# model.fit(trainX, trainY_flat, epochs=100, batch_size=4)
	# score = model.evaluate(testX, testY_flat, verbose=1)
	# print('Test loss:', score[0])
	# print('Test accuracy:', score[1])
	# save_dir = './model/keras_model.h5'
	# model.save(save_dir)
	# print('model saved at',save_dir)

	#
	# run(trainX, testX, trainY, testY)
	#

	# pred = predict(trainX)
	# for i in range(trainY.shape[0]):
	# 	plot_threed(trainX[i],'input')
	# 	pred_reshape = np.reshape(pred[i],(40,20,21,4))
	# 	plot_fourd(pred_reshape,'pred')
	# 	plot_fourd(trainY[i],'truth')
	# 	plt.show()
	#
	# pred = predict(testX)
	# for i in range(testY.shape[0]):
	# 	plot_threed(testX[i],'input')
	# 	pred_reshape = np.reshape(pred[i],(40,20,21,4))
	# 	plot_fourd(pred_reshape,'pred')
	# 	plot_fourd(testY[i],'truth')
	# 	plt.show()
