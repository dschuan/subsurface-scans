import math
import tensorflow as tf
import numpy as np
import pylab as plt
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

def load_data(files,use_preprocessed = True):
	X = []
	Y = []
	for file in files:
		scanArray,truthArray = processArray(file,use_preprocessed = use_preprocessed)
		print('preprocess looking at',file,'truthArray.shape',truthArray.shape)
		X.append(np.transpose(scanArray, (2, 0, 1)))
		Y.append(truthArray)


	return (X,Y)

def getFiles(path,max_items_per_scan,only_max):
	all_files = glob.glob(path)

	scans = []
	for file in all_files:
		if "desc" not in file:
			filename, file_extension = os.path.splitext(file)
			scans.append(filename)

	filtered_scans = []
	for scan in scans:

		with open(scan + '_desc.json') as f:
			data = json.load(f)
			if only_max:
				if len(data['truth']) == max_items_per_scan:
					 filtered_scans.append(scan)
			else:
				if len(data['truth']) <= max_items_per_scan:
					filtered_scans.append(scan)

	return filtered_scans



def loadData(reload, max_items_per_scan = 2, train_test_split = 0.7, only_max = False, saved_path = "../new_res/*.json"):

	trainX_file = './np_save/trainX_v2_' + str(max_items_per_scan)
	trainY_file = './np_save/trainY_v2_' + str(max_items_per_scan)

	if only_max:
		trainX_file = trainX_file + 'only'
		trainY_file = trainY_file + 'only'



	if reload:
		print('preprocess Loading Data for num_items', max_items_per_scan, 'only_max',str(only_max))
		files = getFiles(saved_path,max_items_per_scan,only_max)
		trainX, trainY = load_data(files)
		trainX = np.array(trainX)
		trainY = np.array(trainY)
		np.save(trainX_file, trainX)
		np.save(trainY_file, trainY)

	else:
		trainX =  np.load(trainX_file+'.npy')
		trainY = np.load(trainY_file+'.npy')



	#Shuffle data
	N = len(trainX)
	idx = np.arange(N)
	np.random.seed(5)
	np.random.shuffle(idx)
	trainX, trainY = trainX[idx], trainY[idx]



	if train_test_split == 0:
		v_min = trainX.min(axis=(0, 1, 2,3), keepdims=True)
		v_max = trainX.max(axis=(0, 1, 2,3), keepdims=True)
		testX = (trainX - v_min)/(v_max - v_min)
		testY = trainY
		return [[], flatten4D(testX), [], flatten4D(testY)]

	elif train_test_split == 1:
		v_min = trainX.min(axis=(0, 1, 2,3), keepdims=True)
		v_max = trainX.max(axis=(0, 1, 2,3), keepdims=True)
		trainX = (trainX - v_min)/(v_max - v_min)
		return [flatten4D(trainX), [], flatten4D(trainY), []]

	numSamples = trainX.shape[0]
	#create train test set
	testX, testY = trainX[int(numSamples*train_test_split):],trainY[int(numSamples*train_test_split):]
	print('preprocess Loading test data testX.shape',testX.shape, 'testY.shape', testY.shape)
	trainX, trainY = trainX[:int(numSamples*train_test_split)],trainY[:int(numSamples*train_test_split)]
	print('preprocess Loading train data trainX.shape',trainX.shape,'trainY.shape',trainY.shape)

	# TODO: CHECK THIS NORM!(11, 60, 20, 21)
	v_min = trainX.min(axis=(0, 1, 2, 3), keepdims=True)
	v_max = trainX.max(axis=(0, 1, 2, 3), keepdims=True)
	trainX = (trainX - v_min)/(v_max - v_min)
	testX = (testX - v_min)/(v_max - v_min)
	return [trainX.reshape(trainX.shape + (1,)), testX.reshape(testX.shape + (1,)), trainY, testY]
	# return [flatten4D(trainX), flatten4D(testX),flatten4D(trainY), flatten4D(testY)]

def flatten4D(nparray):
	return nparray.reshape(nparray.shape[0],-1)


def reload_data():

	for num_items in range(2):
		loadData(reload = True, max_items_per_scan = num_items+1)

	for num_items in range(2):
		loadData(reload = True, max_items_per_scan = num_items+1, only_max = True)

if __name__ == '__main__':
	data_params = {

		'reload': False, #When True, parse time domain raw data again, use when data changes
		'max_items_per_scan': 2, # maximum number of items in a scanf
		'train_test_split': 0.7, #size of training data
		'only_max': False,
		'saved_path': "../new_res/*.json"
	}
	reload_data()
	trainX, testX, trainY, testY = loadData(**data_params)

	print('trainX',trainX.shape,'trainY',trainY.shape)
	print('testX',testX.shape,'testY',testY.shape)
