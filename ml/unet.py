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
from preprocess import loadData, processData
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


def get_report(testX,trainX,testY,trainY):

	pred = predict(testX)
	pred_reshape = np.reshape(pred,(-1,16,16,OUTPUT_CHANNELS))
	labelY,labelpred = onehot_to_label(testY,pred_reshape)
	print('test stats********************************************************')
	print(sklearn.metrics.classification_report(labelY,labelpred,target_names = ['empty','pvc','wood','metal','aluminum']))
	cm = sklearn.metrics.confusion_matrix(labelY, labelpred)
	print_cm(cm, labels =  ['empty','pvc','wood','metal','aluminum'])
	print()
	print()
	pred = predict(trainX)
	pred_reshape = np.reshape(pred,(-1,16,16,OUTPUT_CHANNELS))
	labelY,labelpred = onehot_to_label(trainY,pred_reshape)
	print('train stats*******************************************************')
	print(sklearn.metrics.classification_report(labelY,labelpred,target_names = ['empty','pvc','wood','metal','aluminum']))
	cm = sklearn.metrics.confusion_matrix(labelY, labelpred)
	print_cm(cm, labels =  ['empty','pvc','wood','metal','aluminum'])
	print('macro f1',sklearn.metrics.f1_score(labelY,labelpred,average = 'macro'))

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


def create_model(input_size = (32,16,16,1,)):
	K.clear_session()
	tf.reset_default_graph()
	filter_size = 5
	print("input_size",input_size)
	inputs = keras.engine.input_layer.Input(shape = input_size)
	print(inputs)
	print(type(inputs))
	print("inputs",keras.backend.shape(inputs))
	conv1 = keras.layers.Conv3D(18, filter_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
	conv1 = keras.layers.Conv3D(18, filter_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
	pool1 = keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(conv1)
	print("pool1",keras.backend.shape(pool1))
	conv2 = keras.layers.Conv3D(24, filter_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
	conv2 = keras.layers.Conv3D(24, filter_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
	pool2 = keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(conv2)
	conv3 = keras.layers.Conv3D(36, filter_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
	conv3 = keras.layers.Conv3D(36, filter_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
	pool3 = keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(conv3)
	conv4 = keras.layers.Conv3D(48, filter_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
	conv4 = keras.layers.Conv3D(48, filter_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
	drop4 = keras.layers.Dropout(0.4)(conv4)
	pool4 = keras.layers.MaxPooling3D(pool_size=(2, 2, 2))(drop4)

	conv5 = keras.layers.Conv3D(60, filter_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
	conv5 = keras.layers.Conv3D(60, filter_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
	drop5 = keras.layers.Dropout(0.4)(conv5)

	up6 = keras.layers.Conv3D(60, filter_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(keras.layers.UpSampling3D(size = (2,2,2))(drop5))

	print("drop4",keras.backend.shape(drop4))
	print("up6",keras.backend.shape(up6))
	merge6 = keras.layers.concatenate([drop4,up6], axis = 4)
	conv6 = keras.layers.Conv3D(48, filter_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
	conv6 = keras.layers.Conv3D(48, filter_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

	up7 = keras.layers.Conv3D(48, filter_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(keras.layers.UpSampling3D(size = (2,2,2))(conv6))
	merge7 = keras.layers.concatenate([conv3,up7], axis = 4)
	conv7 = keras.layers.Conv3D(36, filter_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
	conv7 = keras.layers.Conv3D(36, filter_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

	up8 = keras.layers.Conv3D(36, filter_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(keras.layers.UpSampling3D(size = (2,2,2))(conv7))
	merge8 = keras.layers.concatenate([conv2,up8], axis = 4)
	conv8 = keras.layers.Conv3D(24, filter_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
	conv8 = keras.layers.Conv3D(24, filter_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

	up9 = keras.layers.Conv3D(24, filter_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(keras.layers.UpSampling3D(size = (2,2,2))(conv8))
	merge9 = keras.layers.concatenate([conv1,up9], axis = 4)
	conv9 = keras.layers.Conv3D(18, filter_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
	conv9 = keras.layers.Conv3D(18, filter_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
	conv9 = keras.layers.Conv3D(18, filter_size, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
	conv10 = keras.layers.Conv3D(5, 1, activation = 'sigmoid')(conv9)

	model = keras.models.Model(input = inputs, output = conv10)

	model.compile(optimizer = keras.optimizers.Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])

	print(model.summary())

	return model


def to_acronym(item):
	splititem = item.split('_')

	if len(splititem) == 1:
		return item
	else:
		return "".join(e[0] for e in splititem)


def run(trainX, testX, trainY_flat, testY_flat ,epochs ,batch_size,model_params,load_model = False,show_plot=True,tag = '',validation_split = 0.0 ):

	if load_model:
		model = keras.models.load_model('./model/keras_model.h5')#,custom_objects={'my_f1_metric': my_f1_metric}
	else:
		model = create_model(**model_params)

	now = datetime.now()

	if validation_split > 0.0:
		tensorboard = keras.callbacks.TensorBoard(log_dir="tblogs\\"+ tag + "{}".format(now.strftime("%m%d-%H%M")),histogram_freq=10,write_images=True)
		history = model.fit(trainX, trainY_flat, epochs=epochs, batch_size=4,validation_split=validation_split,callbacks=[tensorboard])

	else:
		history = model.fit(trainX, trainY_flat, epochs=epochs, batch_size=4,validation_split=validation_split)

	save_dir = './model/keras_model.h5'
	model.save(save_dir)

	score = model.evaluate(testX, testY_flat, verbose=1)

	print(score)
	print('Test loss:', score[0])
	print('Test f1:', score[1])

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
	model = keras.models.load_model('./model/keras_model.h5',custom_objects={})
	pred = model.predict(data)
	return pred

def print_transformations(trainX_,trainY_):
	print("before crop")
	# for item in [trainX_, testX_, trainY_, testY_]:
	# 	print(item.shape)
	item_number = 2
	single_item_x = trainX_[None,item_number,:,:,:]
	single_item_y = trainY_[None,item_number,:,:,:]
	print(single_item_x.shape)
	print(single_item_y.shape)

	single_item_x,single_item_y = processData([single_item_x,single_item_y],commands = ["crop","transpose","flip_x","flip_y"])

	print("after trans")
	print(single_item_x.shape)
	print(single_item_y.shape)

	for i in range(single_item_x.shape[0]):

		plot_threed(single_item_x[i],'input',threshold = 0.3,plot_num = 4)
		plt.savefig('./plot/x' + str(i) + '.png', bbox_inches='tight')
		plot_fourd(single_item_y[i],'truth',plot_num = 5)
		plt.savefig('./plot/y' + str(i) + '.png', bbox_inches='tight')
		# plt.show(block = False)
		# plt.pause(0.3)

if __name__ == '__main__':



	data_params = {

		'reload': False, #When True, parse time domain raw data again, use when data changes
		'max_items_per_scan': 2, # maximum number of items in a scanf
		'train_test_split': 0.7, #size of training data
		'only_max': False,
		'saved_path': "../new_res/*.json",
		'use_backproj': True # set to false to use clean signal instead of backproj
	}
	# reload_data()

	trainX_, testX_, trainY_, testY_ = loadData(**data_params)
	trainX, trainY = processData([trainX_, trainY_],commands = ["crop","transpose","flip_x","flip_y"])
	testX, testY = processData([testX_, testY_],commands = ["crop"])

	# trainX, trainY = processData([trainX_, trainY_],commands = ["crop"])
	# testX, testY = processData([testX_, testY_],commands = ["crop"])


	N = len(trainX)
	idx = np.arange(N)
	np.random.seed(5)
	np.random.shuffle(idx)
	trainX, trainY = trainX[idx], trainY[idx]


	for item in [trainX, testX, trainY, testY]:
		print(item.shape)

	model_params = {

	}
	trainY_type = trainY
	testY_type =testY
	seed = 10
	tf.set_random_seed(seed)
	np.random.seed(seed)
	continue_training = False
	epochs = 100
	batch_size = 16
	tag = 'final_model'
	validation = 0
	working_model_params = model_params.copy()

	# print_transformations(trainX_,trainY_)

	# run(trainX, testX,trainY_type , testY_type ,epochs = epochs ,batch_size = 2,model_params=working_model_params,load_model = continue_training,show_plot=False,validation_split = validation,tag=tag)
	#
	#
	trainY_onehot= np.reshape(trainY,(trainY.shape[0],-1,OUTPUT_CHANNELS))
	testY_onehot= np.reshape(testY,(testY.shape[0],-1,OUTPUT_CHANNELS))
	print('trainY_onehot',trainY_onehot.shape)
	get_report(testX,trainX,testY_onehot,trainY_onehot)


	# pred = predict(trainX)
	# for i in range(trainY.shape[0]):
	# 	plot_threed(trainX[i],'input',threshold = 0.3)
	# 	pred_reshape = np.reshape(pred[i],(32,16,16,OUTPUT_CHANNELS))
	# 	plot_fourd(pred_reshape,'pred')
	# 	plot_fourd(trainY[i],'truth')
	# 	plt.show()

	# pred = predict(testX)
	# for i in range(testY.shape[0]):
	# 	plot_threed(testX[i],'input',threshold = 0.7)
	# 	plt.savefig('./raw/x' + str(i) + '.png', bbox_inches='tight')
	# 	plot_fourd(testY[i],'truth')
	# 	plt.savefig('./raw/y' + str(i) + '.png', bbox_inches='tight')
	# 	# plt.show()



	# np.save('./np_save/' + 'trainX' + 'Final',trainX)
	# np.save('./np_save/' + 'testX' + 'Final',testX)
	# np.save('./np_save/' + 'trainY' + 'Final',trainY)
	# np.save('./np_save/' + 'testY' + 'Final',testY)
	# np.save('./np_save/' + 'combinedX' + 'Final',combinedX)
	# np.save('./np_save/' + 'combinedY_flat' + 'Final',combinedY)
