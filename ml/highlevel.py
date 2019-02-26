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
from postprocess import plot_fourd,plot_threed,print_cm
import unicodedata
import re

NUM_MATERIALS = 4
OUTPUT_CHANNELS = NUM_MATERIALS + 1




def onehot_to_label(testY,pred):

	flatpred = np.round_(np.reshape(pred,(-1,OUTPUT_CHANNELS)))
	flatY = np.reshape(testY,(-1,OUTPUT_CHANNELS))

	labelY = []
	labelpred = []
	for i in range(flatpred.shape[0]):
		Y_one_hot = list(flatY[i])
		if sum(Y_one_hot) != 1:
			raise ValueError('Truth one hot does not sum to one:',sum(Y_one_hot))
		labelY.append(Y_one_hot.index(1))


		pred_one_hot = list(flatpred[i])
		maxindex = pred_one_hot.index(max(pred_one_hot))
		if max(pred_one_hot) == 0:
			labelpred.append(0)
		else:
			labelpred.append(maxindex)

	return labelY,labelpred

# trainY_onehot (36, 16800, 5)
def onehot_to_label_single(testY):
	flatY = np.reshape(testY,(-1,16800,OUTPUT_CHANNELS))
	labelY = np.zeros((flatY.shape[0],flatY.shape[1]))
	for samples in range(flatY.shape[0]):
		for pixle in range(flatY.shape[1]):

			labelY[samples][pixle] = int(list(flatY[samples][pixle]).index(1))

	return labelY

def my_f1_metric(Y,pred):
	print('mymetric Y',Y.shape)
	print('mymetric pred',pred.shape)
	pred_reshape = np.reshape(pred,(-1,20,21,OUTPUT_CHANNELS))
	labelY,labelpred = onehot_to_label(Y,pred_reshape)
	return sklearn.metrics.f1_score(labelY,labelpred,average = 'macro')

def saveHist(path,history):

	new_hist = {}
	for key in list(history.history.keys()):
		if type(history.history[key]) == np.ndarray:
			new_hist[key] == history.history[key].tolist()
		elif type(history.history[key]) == list:
		   if  type(history.history[key][0]) == np.float64:
			   new_hist[key] = list(map(float, history.history[key]))

	with codecs.open(path, 'w', encoding='utf-8') as f:
		json.dump(new_hist, f, separators=(',', ':'), sort_keys=True, indent=4)

def loadHist(path):
	with codecs.open(path, 'r', encoding='utf-8') as f:
		n = json.loads(f.read())
	return n


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


def create_model(hidden_layers = [(5,5,10),(2,5,10),(2,5,10),(2,5,OUTPUT_CHANNELS)], activation = 'relu', kernel_initializer = 'glorot_uniform',dropout_rate = 0.0,batch_norm = False,loss='mse', optimizer='rmsprop', kernel_regularizer = None, metrics=['accuracy']  ):
	dilation_rate=(1, 1, 1)
	K.clear_session()
	tf.reset_default_graph()
	layer_params = {
		"filters": 10,
		"kernel_size": (5,5,5),
		"padding":'same',
		"data_format":'channels_last',
		"activation":'relu',
		"use_bias":True,
		"kernel_initializer":'glorot_uniform',
		"bias_initializer":'zeros',
		"kernel_regularizer":None,
		'dilation_rate':(1, 1, 1)

	}

	model = keras.models.Sequential()
	for index, layer in enumerate(hidden_layers):
		z_stride, kernel_size, num_filters = layer
		layer_params["kernel_size"] = (kernel_size,kernel_size,kernel_size)
		layer_params["filters"] = num_filters
		layer_params["activation"] = activation
		layer_params["kernel_initializer"] = kernel_initializer
		layer_params["kernel_regularizer"] = kernel_regularizer
		layer_params["strides"] = (z_stride,1,1)

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

	# model.add(keras.layers.Conv3D(filters=OUTPUT_CHANNELS, kernel_size=(1,1,1),kernel_initializer=keras.initializers.Ones(), padding='same', data_format='channels_last', activation='softmax'))
	# model.add(keras.layers.Flatten( data_format = 'channels_last' ))
	model.add(keras.layers.Reshape((-1,OUTPUT_CHANNELS)))
	# model.add(keras.layers.Lambda((depth_softmax)))
	# model.add(keras.layers.Softmax(axis = -1))


	model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

	return model

def to_acronym(item):
	splititem = item.split('_')

	if len(splititem) == 1:
		return item
	else:
		return "".join(e[0] for e in splititem)

def make_run_name(model_params,tag = ''):
	"""
	Normalizes string, converts to lowercase, removes non-alpha characters,
	and converts spaces to hyphens.
	"""

	value = tag + '_'
	value = value + 'opt_' + str(model_params['optimizer'])
	value = value + '_loss_' + to_acronym(str(model_params['loss']))
	value = value + '_bnorm_' + str(model_params['batch_norm'])[0]
	value = value + '_kinit_' + to_acronym(str(model_params['kernel_initializer']))
	value = value + '_act_' + str(model_params['activation'])
	value = value + '_drop_' + str(model_params['dropout_rate'])
	value = value + '_strisizexnum_'

	for layer in model_params['hidden_layers']:
		zdil,size,num = layer
		value = value + str(zdil) + 'x' + str(size) + 'x' + str(num) + '_'


	return value

def run(trainX, testX, trainY_flat, testY_flat ,epochs ,batch_size,model_params,load_model = False,show_plot=True,tag = '' ):

	print('running with params')
	print(model_params)
	if load_model:
		model = keras.models.load_model('./model/keras_model.h5')#,custom_objects={'my_f1_metric': my_f1_metric}
	else:
		model = create_model(**model_params)

	now = datetime.now()
	tensorboard = keras.callbacks.TensorBoard(log_dir="tblogs\\"+ make_run_name(model_params,tag) +"{}".format(now.strftime("%m%d-%H%M")),histogram_freq=10,write_images=True)
	history = model.fit(trainX, trainY_flat, epochs=epochs, batch_size=4,validation_split=0.3,callbacks=[tensorboard])

	score = model.evaluate(testX, testY_flat, verbose=1)

	print(score)
	print('Test loss:', score[0])
	print('Test f1:', score[1])
	save_dir = './model/keras_model.h5'
	model.save(save_dir)
	print('model saved at',save_dir)
	histpath = './model/keras_history.json'
	saveHist(histpath,history)
	# loadedhist = loadHist(histpath)
	# print(loadedhist)

	my_file = Path('./np_save/highlevellogs.npy')
	if my_file.is_file():
		logs = list(np.load('./np_save/highlevellogs.npy'))
	else:
		logs = []
	current = {'model_params':model_params,'loss':score[0],'f1':score[1],'epochs':epochs,'batch_size':batch_size}
	logs.append(current)
	np.save('./np_save/highlevellogs.npy', logs)
	print('logs saved at ./np_save/highlevellogs.npy')



	# # list all data in history
	# print(history.history.keys())
	if show_plot:
		plot_num = 0
		for metric in history.history.keys():
			# summarize history for accuracy
			plt.figure(plot_num)
			plot_num = plot_num + 1
			plt.plot(history.history[metric])
			plt.title(metric)
			plt.xlabel('epoch')
		plt.legend([*history.history.keys()], loc='upper left')
		plt.show()


def predict(data):
	model = keras.models.load_model('./model/keras_model.h5',custom_objects={'f1': f1})
	pred = model.predict(data)
	return pred

def grid_search(X,Y):


	model = KerasClassifier(build_fn=create_model,verbose=1)
	# define the grid search parameters
	batch_size = [1,2,4]
	epochs = [100, 200, 400]
	hidden_layers = [
		[(5,10),(5,10),(5,10),(5,10),(5,OUTPUT_CHANNELS)],
		[(5,10),(5,10),(5,10),(5,OUTPUT_CHANNELS)],
		[(5,10),(5,10),(5,OUTPUT_CHANNELS)],
		[(5,10),(5,OUTPUT_CHANNELS)],
		[(5,OUTPUT_CHANNELS)]
		]
	activation = ['relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
	kernel_initializer = ['uniform', 'lecun_uniform', 'normal', 'glorot_normal', 'glorot_uniform']
	dropout_rate = [0.0,0.2,0.4,0.6]
	batch_norm = [True, False]
	loss=['mean_squared_error','mean_absolute_error','mean_absolute_percentage_error','mean_squared_logarithmic_error','squared_hinge','hinge','logcosh','categorical_crossentropy','binary_crossentropy','kullback_leibler_divergence','poisson','cosine_proximity']
	optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']

	param_grid = dict(batch_size=[1], epochs=[1],optimizer = optimizer)

	# def print_mse(y,ypred):
	# 	print('y',y.shape)
	# 	print('ypred',ypred.shape)
	# 	return sklearn.metrics.mean_squared_error(y,ypred)
	#
	# scorer = make_scorer(print_mse) ,scoring=['f1_macro'], refit='f1_macro'

	grid = GridSearchCV(estimator=model, param_grid=param_grid,scoring=['neg_mean_squared_error'], refit='neg_mean_squared_error',n_jobs=1)

	print('grid search starting')
	print('X',X.shape)
	print('Y',Y.shape)
	grid_result = grid.fit(X,Y)
	# summarize results
	print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
	means = grid_result.cv_results_['mean_test_neg_mean_squared_error']
	stds = grid_result.cv_results_['std_test_neg_mean_squared_error']
	params = grid_result.cv_results_['params']

	for mean, stdev, param in zip(means, stds, params):
		print("%f (%f) with: %r" % (mean, stdev, param))
	now = datetime.now()
	np.save('./np_save/gridsearch' + now.strftime("%m%d-%H%M"), {'mean':means, 'stdev':stds, 'param':params,'best_score':grid_result.best_score_,'best_param':grid_result.best_params_})

if __name__ == '__main__':



	data_params = {

		'reload': False, #When True, parse time domain raw data again, use when data changes
		'max_items_per_scan': 2, # maximum number of items in a scanf
		'train_test_split': 0.7, #size of training data
		'only_max': False,
		'saved_path': "../new_res/*.json",
		'z_len':32
	}
	# reload_data()

	trainX, testX, trainY, testY = loadData(**data_params)



	combinedX = np.concatenate((trainX,testX),axis = 0)
	combinedY = np.concatenate((trainY,testY),axis = 0)# (34, 40, 20, 21, 5)
	combinedY = np.reshape(combinedY,(combinedY.shape[0],-1))

	trainY_flat= np.reshape(trainY,(trainY.shape[0],-1))
	testY_flat= np.reshape(testY,(testY.shape[0],-1))

	trainY_onehot= np.reshape(trainY,(trainY.shape[0],-1,OUTPUT_CHANNELS))
	testY_onehot= np.reshape(testY,(testY.shape[0],-1,OUTPUT_CHANNELS))

	testY_label = np.array(onehot_to_label_single(testY_flat))
	trainY_label = np.array(onehot_to_label_single(trainY_flat))

	trainY_slice = trainY[:,13,:,:,:]
	testY_slice = testY[:,13,:,:,:]
	trainY_slice = np.reshape(trainY_slice,(trainY_slice.shape[0],20*21,OUTPUT_CHANNELS))
	testY_slice = np.reshape(testY_slice,(testY_slice.shape[0],20*21,OUTPUT_CHANNELS))
	print('trainY',trainY.shape)
	print('trainY_onehot',trainY_onehot.shape)
	print('trainY_label',trainY_label.shape)
	print('trainY_slice',trainY_slice.shape)



	# grid_search(combinedX,combinedY)
	hidden_layers_space = [
		[(3,15),(3,15),(3,15),(3,15),(3,15),(3,OUTPUT_CHANNELS)],
		[(5,15),(5,15),(5,15),(5,15),(5,15),(5,OUTPUT_CHANNELS)],
		[(7,15),(7,15),(7,15),(7,15),(7,15),(5,OUTPUT_CHANNELS)],
		[(9,15),(9,15),(9,15),(9,15),(9,15),(5,OUTPUT_CHANNELS)],
		[(5,15),(5,15),(7,15),(7,15),(9,15),(5,OUTPUT_CHANNELS)],
		[(9,15),(9,15),(7,15),(5,15),(5,15),(5,OUTPUT_CHANNELS)],
		[(5,30),(5,20),(5,15),(5,10),(5,10),(5,OUTPUT_CHANNELS)],
		[(5,10),(5,10),(5,15),(5,20),(5,30),(5,OUTPUT_CHANNELS)],
		[(11,10),(5,10),(5,10),(5,10),(5,10),(5,OUTPUT_CHANNELS)],
		[(5,10),(5,10),(5,10),(5,10),(5,10),(5,OUTPUT_CHANNELS)],
		[(5,10),(5,10),(5,10),(5,10),(5,10),(5,10),(5,OUTPUT_CHANNELS)],
		[(5,10),(5,10),(5,10),(5,10),(5,10),(5,10),(5,10),(5,OUTPUT_CHANNELS)],
		[(5,10),(5,10),(5,10),(5,10),(5,OUTPUT_CHANNELS)],
		[(5,10),(5,10),(5,10),(5,OUTPUT_CHANNELS)],

		]
	activation_space = ['relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
	kernel_initializer_space = ['uniform', 'lecun_uniform', 'normal', 'glorot_normal', 'glorot_uniform']
	dropout_rate_space = [0.0,0.1,0.2,0.3,0.4,0.5]
	batch_norm_space = [True, False]
	#bad 'categorical_crossentropy',,'mean_absolute_percentage_error' 'squared_hinge','hinge'
	loss_space=['mean_squared_error','mean_absolute_error','mean_squared_logarithmic_error','logcosh','binary_crossentropy','kullback_leibler_divergence','poisson','cosine_proximity']
	#bad  'Adagrad', 'Adadelta', 'Nadam'
	optimizer_space = ['SGD', 'RMSprop', 'Adam', 'Adamax']
	regularizer_space = [None,keras.regularizers.l1(0.01),keras.regularizers.l2(0.01),keras.regularizers.l1_l2(l1=0.01, l2=0.01)]

	model_params = {
		'hidden_layers': [(1,7,10),(2,7,10),(1,7,10),(2,7,10),(2,7,10),(2,5,10),(2,2,10),(1,1,10),(1,5,OUTPUT_CHANNELS)], #format: kernel_size, num_filters
		'activation':'relu',
		'kernel_initializer': 'glorot_uniform',
		'dropout_rate' : 0.2,
		'batch_norm':True,
		'loss': 'mse',
		'optimizer': 'rmsprop',
		'kernel_regularizer': None
	}


	trainY_type = trainY_onehot
	testY_type =testY_onehot
	seed = 10
	tf.set_random_seed(seed)
	np.random.seed(seed)
	continue_training = False
	epochs = 1000
	batch_size = 4
	tag = ''
	working_model_params = model_params.copy()

	run(trainX, testX,trainY_slice , testY_slice ,epochs = epochs ,batch_size = batch_size,model_params=working_model_params,load_model = continue_training,show_plot=False,tag=tag )
	# for _ in range(3):
	# 	print('cooling down')
	# 	time.sleep(30)
	# 	# working_model_params = model_params.copy()
	# 	# for kernel_initializer_item in kernel_initializer_space:
	# 	# 	working_model_params['kernel_initializer'] = kernel_initializer_item
	# 	# 	run(trainX, testX,trainY_type , testY_type ,epochs = epochs ,batch_size = batch_size,model_params=working_model_params,load_model = continue_training,show_plot=False,tag=tag )
	#
	# 	tag = 'hiddengrid'
	# 	working_model_params = model_params.copy()
	# 	for hidden_layers_item in hidden_layers_space:
	# 		working_model_params['hidden_layers'] = hidden_layers_item
	# 		run(trainX, testX,trainY_type , testY_type ,epochs = epochs ,batch_size = batch_size,model_params=working_model_params,load_model = continue_training,show_plot=False,tag=tag )
	#
	# 	# tag = 'actgrid'
		# working_model_params = model_params.copy()
		# for activation_item in activation_space:
		# 	working_model_params['activation'] = activation_item
		# 	run(trainX, testX,trainY_type , testY_type ,epochs = epochs ,batch_size = batch_size,model_params=working_model_params,load_model = continue_training,show_plot=False,tag=tag )

		# tag = 'dropgrid'
		# working_model_params = model_params.copy()
		# for dropout_item in dropout_rate_space:
		# 	working_model_params['dropout_rate'] = dropout_item
		# 	run(trainX, testX,trainY_type , testY_type ,epochs = epochs ,batch_size = batch_size,model_params=working_model_params,load_model = continue_training,show_plot=False,tag=tag )
		#
		# tag = 'reggrid'
		# working_model_params = model_params.copy()
		# for regularizer_item in regularizer_space:
		# 	working_model_params['kernel_regularizer'] = regularizer_item
		# 	run(trainX, testX,trainY_type , testY_type ,epochs = epochs ,batch_size = batch_size,model_params=working_model_params,load_model = continue_training,show_plot=False,tag=tag )

		# tag = 'optgrid'
		# working_model_params = model_params.copy()
		# for optimizer_item in optimizer_space:
		# 	working_model_params['optimizer'] = optimizer_item
		# 	run(trainX, testX,trainY_type , testY_type ,epochs = epochs ,batch_size = batch_size,model_params=working_model_params,load_model = continue_training,show_plot=False,tag=tag )
		#


		# tag = 'lossgrid'
		# working_model_params = model_params.copy()
		# for loss_item in loss_space:
		# 	working_model_params['loss'] = loss_item
		# 	run(trainX, testX,trainY_type , testY_type ,epochs = epochs ,batch_size = batch_size,model_params=working_model_params,load_model = continue_training,show_plot=False,tag=tag )
		#

	get_report(testX,trainX,testY_slice,trainY_slice)


	# pred = predict(trainX)
	# print('pred',pred.shape)
	# remade = np.zeros((pred.shape[0],trainX.shape[1],20,21,5))
	# print('remade',remade.shape)
	# pred = np.reshape(pred,(pred.shape[0],20,21,5))
	# print('pred',pred.shape)
	# print('remade_slice',remade[:,13,:,:,:].shape)
	# remade[:,13,:,:,:] = pred
	#
	# for i in range(trainY.shape[0]):
	# 	plot_threed(trainX[i],'input')
	# 	plot_fourd(remade[i],'pred')
	# 	plot_fourd(trainY[i],'truth')
	# 	plt.show()

	pred = predict(testX)
	print('pred',pred.shape)
	remade = np.zeros((pred.shape[0],trainX.shape[1],20,21,5))
	print('remade',remade.shape)
	pred = np.reshape(pred,(pred.shape[0],20,21,5))
	print('pred',pred.shape)
	print('remade_slice',remade[:,13,:,:,:].shape)
	remade[:,13,:,:,:] = pred
	for i in range(testY.shape[0]):
		plot_threed(testX[i],'input')
		plot_fourd(remade[i],'pred')
		plot_fourd(testY[i],'truth')
		plt.show()



	# np.save('./np_save/' + 'trainX' + 'Final',trainX)
	# np.save('./np_save/' + 'testX' + 'Final',testX)
	# np.save('./np_save/' + 'trainY' + 'Final',trainY)
	# np.save('./np_save/' + 'testY' + 'Final',testY)
	# np.save('./np_save/' + 'combinedX' + 'Final',combinedX)
	# np.save('./np_save/' + 'combinedY_flat' + 'Final',combinedY)
