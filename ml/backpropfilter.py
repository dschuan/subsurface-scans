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



#size of input
IMG_SIZE_X = 20
IMG_SIZE_Y = 21
IMG_SIZE_Z = 60
NUM_CHANNELS = 1
OUTPUT_CHANNELS = 1

#ml variables
learning_rate = 0.01
epochs = 2000
batch_size = 1

# raw json processing variables
# SAMPLE_SIZE = 1000
SAMPLE_KEEP_PROB = 1

PRINT_INFO = True

seed = int(time.time())
np.random.seed(seed)
tf.set_random_seed(seed)

def load_data(files):
	X = []
	Y = []
	for file in files:
		scanArray,truthArray = processArray(file)
		print('looking at',file,'truthArray.shape',truthArray.shape)
		X.append(scanArray)
		Y.append(truthArray)
	#
	# X = [sampleArray(scanArray).flatten() for i in range(SAMPLE_SIZE)]
	# Y = [truthArray.flatten() for i in range(SAMPLE_SIZE)]

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


def sampleArray(input):

	sampleArray = np.zeros((input.shape[0],input.shape[1],2))
	for x, row in enumerate(input):
			for y, value in enumerate(row):
				include = poll(SAMPLE_KEEP_PROB)
				if include:
					sampleArray[x][y][1] = 1
					sampleArray[x][y][0] = value
	return sampleArray

def poll(probability):
	return random.random() < probability


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


def oneFilter(input,num_input_layers,num_output_layers,filter_height,filter_size,name,keep_prob,apply_dropout):

	images = tf.reshape(input, [-1, IMG_SIZE_Z, IMG_SIZE_X, IMG_SIZE_Y, num_input_layers])


	with tf.variable_scope('conv' + name) as scope:
		#Conv 1
		with tf.name_scope('weights'):
			# ini w 1.0/np.sqrt(num_input_layers*filter_size*filter_size) previously
			W1 = tf.Variable(tf.ones([filter_height, filter_size, filter_size, num_input_layers, num_output_layers]), name='weights_' + name)
			#variable_summaries(W1)


			# filterImage, presenceImage = tf.split(W1, num_or_size_splits=2, axis=2)
			# print("filterImage",filterImage.get_shape())
			# filterImage = tf.reshape (filterImage, [-1, FILTER_SIZE, FILTER_SIZE, 1])
			# presenceImage = tf.reshape (presenceImage, [-1, FILTER_SIZE, FILTER_SIZE, 1])
			# print("filterImage",filterImage.get_shape())
			# tf.summary.image('filterImage',filterImage)
			# tf.summary.image('presenceImage',presenceImage)

		with tf.name_scope('biases'):
			b1 = tf.Variable(tf.zeros([num_output_layers]), name='biases_' + name)
			#variable_summaries(b1)

		#tf.nn.conv2d(input,filter,strides,padding,)
		with tf.name_scope('output'):
			conv_1 = tf.nn.relu(tf.nn.conv2d(images, W1, [1, 1, 1, 1], padding='SAME') + b1)
			if apply_dropout:
				conv_1 = tf.nn.dropout(conv_1, keep_prob)

			# tf.summary.image('conv_1',conv_1)

		dim_1 = conv_1.get_shape()[1].value * conv_1.get_shape()[2].value * conv_1.get_shape()[3].value
		pool_1_flat = tf.reshape(conv_1, [-1, dim_1])


		if(PRINT_INFO):
			print('convolution layer',name)
			print('filter input shape',input.shape)
			print('reshaping to',images.shape)
			print("shape of filters",W1.get_shape())
			print("shape of output",conv_1.get_shape())
			print("shape of output flattened",pool_1_flat.get_shape())

	return pool_1_flat


def backPropFilter(input,num_input_layers,num_output_layers,filter_depth,filter_size,name,keep_prob,apply_dropout):

	images = tf.reshape(input, [-1, IMG_SIZE_Z, IMG_SIZE_X, IMG_SIZE_Y, num_input_layers])


	with tf.variable_scope('conv' + name) as scope:
		with tf.name_scope('weights'):
			W1 = tf.Variable(tf.ones([filter_depth, filter_size, filter_size, num_input_layers, num_output_layers]), name='weights_' + name)
			#variable_summaries(W1)
		mask = create_filter(filter_z_len = filter_depth,filter_x_len = filter_size,filter_y_len = filter_size,distance_to_top = 14,visualise = False)
		mask = mask.reshape( [filter_depth,filter_size,filter_size,num_input_layers,num_output_layers] )
		mask_variable = tf.Variable( mask , dtype=tf.float32 )
		mask = tf.stop_gradient( mask_variable )



		with tf.name_scope('biases'):
			b1 = tf.Variable(tf.zeros([num_output_layers]), name='biases_' + name)
			#variable_summaries(b1)

		with tf.name_scope('output'):
			# With the default format "NDHWC", the data is stored in the order of: [batch, in_depth, in_height, in_width, in_channels]
			conv_1 = tf.nn.relu(tf.nn.conv3d(images, W1*mask, [1, 1, 1, 1,1], padding='SAME') + b1)
			if apply_dropout:
				conv_1 = tf.nn.dropout(conv_1, keep_prob)

			# tf.summary.image('conv_1',conv_1)

		dim_1 = conv_1.get_shape()[1].value * conv_1.get_shape()[2].value * conv_1.get_shape()[3].value
		pool_1_flat = tf.reshape(conv_1, [-1, dim_1])


		if(PRINT_INFO):
			print('convolution layer',name)
			print('filter input shape',input.shape)
			print('reshaping to',images.shape)
			print("shape of filters",W1.get_shape())
			print('shape of mask',mask.get_shape())
			print("shape of output",conv_1.get_shape())
			print("shape of output flattened",pool_1_flat.get_shape())

	return pool_1_flat

def chainFilter(input, layerlist, filter_size,keep_prob):
	layer = input
	first_layer = True
	for index in range(len(layerlist)-1):

		layer = oneFilter(layer,layerlist[index],layerlist[index+1],filter_size,str(index),keep_prob,first_layer)
		first_layer = False

	return layer

def loadData(reload, max_items_per_scan = 4, train_test_split = 0.7, only_max = False,saved_path = "../new_res/*.json"):
	#load files

	trainX_file = './np_save/trainX' + str(max_items_per_scan)
	trainY_file = './np_save/trainY' + str(max_items_per_scan)

	if only_max:
		trainX_file = trainX_file + 'only'
		trainY_file = trainY_file + 'only'



	if reload:
		print('Loading Data for num_items', max_items_per_scan, 'only_max',str(only_max))
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
	np.random.shuffle(idx)
	trainX, trainY = trainX[idx], trainY[idx]



	if train_test_split == 0:
		v_min = trainX.min(axis=(0, 1, 2), keepdims=True)
		v_max = trainX.max(axis=(0, 1, 2), keepdims=True)
		testX = (trainX - v_min)/(v_max - v_min)
		testY = trainY
		return [[], flatten4D(testX), [], flatten4D(testY)]

	elif train_test_split == 1:
		v_min = trainX.min(axis=(0, 1, 2), keepdims=True)
		v_max = trainX.max(axis=(0, 1, 2), keepdims=True)
		trainX = (trainX - v_min)/(v_max - v_min)
		return [flatten4D(trainX), [], flatten4D(trainY), []]

	numSamples = trainX.shape[0]
	#create train test set
	testX, testY = trainX[int(numSamples*train_test_split):],trainY[int(numSamples*train_test_split):]
	print('Loading test data testX.shape',testX.shape, 'testY.shape', testY.shape)
	trainX, trainY = trainX[:int(numSamples*train_test_split)],trainY[:int(numSamples*train_test_split)]
	print('Loading train data trainX.shape',trainX.shape,'trainY.shape',trainY.shape)

	#TODOD: CHECK THIS NORM!
	v_min = trainX.min(axis=(0, 1, 2), keepdims=True)
	v_max = trainX.max(axis=(0, 1, 2), keepdims=True)
	trainX = (trainX - v_min)/(v_max - v_min)
	testX = (testX - v_min)/(v_max - v_min)

	return [flatten4D(trainX), flatten4D(testX),flatten4D(trainY), flatten4D(testY)]

def flatten4D(nparray):
	return nparray.reshape(nparray.shape[0],-1)

def run(trainX, testX, trainY, testY, filter_depth, filter_size, keep, use_saved_model, exp_name, show_results):
	modelpath = "./model/testmodel.ckpt"
	print('Using test data testX.shape',testX.shape, 'testY.shape', testY.shape)
	if not show_results:
		print('Using train data trainX.shape',trainX.shape,'trainY.shape',trainY.shape)

	now = datetime.now()
	logdir = "./logs/" + exp_name + '/'
	logdir = logdir + str(filter_depth) + 'x' +str(filter_size) + '_'
	logdir = logdir + str(keep) + "keepprob/"+ now.strftime("%m%d-%H%M%S")
	logdir = logdir + '/'
	print('params',logdir)
	tf.reset_default_graph()

	# Create the model
	x = tf.placeholder(tf.float32, [None, IMG_SIZE_Z*IMG_SIZE_X*IMG_SIZE_Y*NUM_CHANNELS],name='x')
	y_ = tf.placeholder(tf.float32, [None, IMG_SIZE_Z*IMG_SIZE_X*IMG_SIZE_Y*OUTPUT_CHANNELS],name='y')

	keep_prob = tf.placeholder(tf.float32,name = 'keep_prob')
	logits = backPropFilter(x,NUM_CHANNELS,OUTPUT_CHANNELS,filter_depth,filter_size,'1',keep_prob,apply_dropout = True)
	# logits = chainFilter(x,layerlist,filter_size,keep_prob)

	print("labels y_.shape",y_.shape)
	print("logits logits.shape", logits.shape)
	# cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=logits)
	# loss = tf.reduce_mean(cross_entropy)
	with tf.name_scope('loss'):
		loss = tf.reduce_mean(tf.squared_difference(logits, y_))
	tf.summary.scalar('loss', loss)


	# correct_prediction = tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1)), tf.float32)
	# accuracy = tf.reduce_mean(correct_prediction)

	#train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
	train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
	N = len(trainX)
	idx = np.arange(N)
	saver = tf.train.Saver()

	if not show_results:
		print('session starting')
		with tf.Session() as sess:
			print('session started')
			if use_saved_model:
				print('using saved model at',modelpath)
				saver.restore(sess, modelpath)
			else:
				sess.run(tf.global_variables_initializer())

			merged = tf.summary.merge_all()
			train_writer = tf.summary.FileWriter(logdir + 'train', sess.graph)
			test_writer = tf.summary.FileWriter(logdir + 'test')



			for e in range(epochs):
				np.random.shuffle(idx)
				trainX, trainY = trainX[idx], trainY[idx]

				for start, end in zip(range(0, trainX.shape[0], batch_size), range(batch_size, trainX.shape[0], batch_size)):
					sess.run(train_step,feed_dict={x: trainX[start:end], y_: trainY[start:end], keep_prob :keep})

				if e%50 == 0:
					loss_ = loss.eval({x: trainX, y_: trainY, keep_prob : 1.0})
					test_loss_ = loss.eval({x: testX, y_: testY, keep_prob : 1.0})
					summary = sess.run(merged, feed_dict={x: trainX, y_: trainY, keep_prob : 1.0})
					train_writer.add_summary(summary, e)
					summary = sess.run(merged, feed_dict={x: testX, y_: testY, keep_prob : 1.0})
					test_writer.add_summary(summary, e)

					print('epoch', e, 'loss', loss_,'test_loss',test_loss_)

			save_path = saver.save(sess, modelpath)
			print("Model saved in path: %s" % save_path)

	else:
		with tf.Session() as sess:
			# Restore variables from disk.
			saver.restore(sess, modelpath)

			results = sess.run(logits,feed_dict={x: testX,keep_prob : 1.0})

			print('len result',len(results))
			print('type result',type(results))

			for raw,pred,truth in zip(testX,results,testY):

				figure = plt.figure(1)
				figure.suptitle('prediction')
				# x,y,z = np.nonzero(np.reshape(pred,(IMG_SIZE_X,IMG_SIZE_Y,OUTPUT_CHANNELS)) > 0.5)
				# ax = figure.add_subplot(111, projection='3d')
				# axes = plt.gca()
				# axes.set_xlim([0,20])
				# axes.set_ylim([0,19])
				# axes.set_zlim([0,OUTPUT_CHANNELS-1])
				# ax.scatter(x, y, z)
				ax = figure.add_subplot(111, projection='3d')
				axes = plt.gca()
				axes.set_xlim([0,20])
				axes.set_ylim([0,19])
				axes.set_zlim([0,OUTPUT_CHANNELS-1])
				pred_reshaped = np.reshape(pred,(IMG_SIZE_X,IMG_SIZE_Y,OUTPUT_CHANNELS))
				x = np.arange(pred_reshaped.shape[0])[:, None, None]
				y = np.arange(pred_reshaped.shape[1])[None, :, None]
				z = np.arange(pred_reshaped.shape[2])[None, None, :]
				x, y, z = np.broadcast_arrays(x, y, z)
				c = np.tile(pred_reshaped.ravel()[:, None], [1, 3])
				ax.scatter(x.ravel(),
		           y.ravel(),
		           z.ravel(),
		           c=pred_reshaped.ravel(),
				   cmap=plt.get_cmap('Reds'))

				figure = plt.figure(2)
				figure.suptitle('truth')
				reshaped_truth = np.reshape(truth,(IMG_SIZE_X,IMG_SIZE_Y,OUTPUT_CHANNELS))
				# print('truth',list(truth))
				x,y,z = reshaped_truth.nonzero()
				ax = figure.add_subplot(111, projection='3d')
				axes = plt.gca()
				axes.set_xlim([0,20])
				axes.set_ylim([0,19])
				axes.set_zlim([0,OUTPUT_CHANNELS-1])
				ax.scatter(x, y, z)

				figure = plt.figure(3)
				figure.suptitle('prediction flat')
				flat_pred = np.reshape(pred,(IMG_SIZE_X,IMG_SIZE_Y,OUTPUT_CHANNELS)).sum(axis=(2))

				axes = plt.gca()
				axes.set_xlim([0,20])
				axes.set_ylim([0,19])

				plt.imshow(flat_pred)

				figure = plt.figure(4)
				figure.suptitle('truth flat')
				flat_truth = np.reshape(truth,(IMG_SIZE_X,IMG_SIZE_Y,OUTPUT_CHANNELS)).sum(axis=(2))

				axes = plt.gca()
				axes.set_xlim([0,20])
				axes.set_ylim([0,19])

				plt.imshow(flat_truth)

				figure = plt.figure(5)
				figure.suptitle('raw1')

				flat_truth = np.reshape(raw,(IMG_SIZE_X,IMG_SIZE_Y,NUM_CHANNELS))[:,:,0]
				axes = plt.gca()
				axes.set_xlim([0,20])
				axes.set_ylim([0,19])

				plt.imshow(flat_truth)

				figure = plt.figure(6)
				figure.suptitle('raw2')

				flat_truth = np.reshape(raw,(IMG_SIZE_X,IMG_SIZE_Y,NUM_CHANNELS))[:,:,3]
				axes = plt.gca()
				axes.set_xlim([0,20])
				axes.set_ylim([0,19])

				plt.imshow(flat_truth)

				figure = plt.figure(7)
				figure.suptitle('rawwide')
				ax = figure.add_subplot(111, projection='3d')
				axes = plt.gca()
				axes.set_xlim([0,20])
				axes.set_ylim([0,19])
				axes.set_zlim([0,NUM_CHANNELS-1])
				raw_reshaped = np.reshape(raw,(IMG_SIZE_X,IMG_SIZE_Y,NUM_CHANNELS))
				x = np.arange(raw_reshaped.shape[0])[:, None, None]
				y = np.arange(raw_reshaped.shape[1])[None, :, None]
				z = np.arange(raw_reshaped.shape[2])[None, None, :]
				x, y, z = np.broadcast_arrays(x, y, z)
				c = np.tile(raw_reshaped.ravel()[:, None], [1, 3])
				ax.scatter(x.ravel(),
		           y.ravel(),
		           z.ravel(),
		           c=raw_reshaped.ravel(),
				   cmap=plt.get_cmap('Reds'))

				figure = plt.figure(8)
				figure.suptitle('rawsum')

				flat_truth = np.reshape(raw,(IMG_SIZE_X,IMG_SIZE_Y,NUM_CHANNELS)).sum(axis = (2))
				axes = plt.gca()
				axes.set_xlim([0,20])
				axes.set_ylim([0,19])

				plt.imshow(flat_truth)

				plt.show()

def generate_layers(layer_space,num_layers,NUM_CHANNELS,OUTPUT_CHANNELS):
	output = []
	for layer in range(num_layers):
		combi = list(itertools.combinations_with_replacement(layer_space,layer))
		combi = [[NUM_CHANNELS] + list(x) + [OUTPUT_CHANNELS] for x in combi]
		output = output + combi

	return output


def reload_data():

	for num_items in range(2):
		loadData(reload = True, max_items_per_scan = num_items+1)

	# for num_items in range(2):
	# 	loadData(reload = True, max_items_per_scan = num_items+1, only_max = True)



if __name__ == '__main__':

	model_params = {
		'filter_size': 40,
		'filter_depth': 40,
		# 'layerlist': [NUM_CHANNELS,10,20,OUTPUT_CHANNELS],
		'keep': 0.7, #dropout rate for first cnn layer
		'use_saved_model':False, #when true, uses the previous model and continue training
		'exp_name': 'backprop_norm_drop1',
		'show_results': False #when true, uses the previously trained model and shows results of test,

	}

	data_params = {

		'reload': False, #When True, parse time domain raw data again, use when data changes
		'max_items_per_scan': 2, # maximum number of items in a scan
		'train_test_split': 0.7, #size of training data
		'only_max': False,
		'saved_path': "../new_res/*.json"
	}
	trainX, testX, trainY, testY = loadData(**data_params)
	run(trainX, testX, trainY, testY, **model_params)

	# reload_data()
