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
from processJSON import processJSON
from datetime import datetime
import itertools




#size of input
IMG_SIZE_X = 20
IMG_SIZE_Y = 21
NUM_CHANNELS = 5
OUTPUT_CHANNELS = 4

#ml variables
learning_rate = 0.005
epochs = 100
batch_size = 1

# raw json processing variables
# SAMPLE_SIZE = 1000
SAMPLE_KEEP_PROB = 1

seed = 10
np.random.seed(seed)
tf.set_random_seed(seed)

def load_data(files):
	X = []
	Y = []
	for file in files:

		print('looking at',file)
		scanArray,truthArray = processJSON(file,NUM_CHANNELS)
		print('truthArray.shape',truthArray.shape)
		X.append(scanArray.flatten())
		Y.append(truthArray.flatten())
	#
	# X = [sampleArray(scanArray).flatten() for i in range(SAMPLE_SIZE)]
	# Y = [truthArray.flatten() for i in range(SAMPLE_SIZE)]

	return (X,Y)

def getFiles(path):
	all_files = glob.glob(path)

	scans = []
	for file in all_files:
		if "desc" not in file:
			filename, file_extension = os.path.splitext(file)
			scans.append(filename)

	return scans


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


def oneFilter(input,num_input_layers,num_output_layers,filter_size,name,keep_prob,apply_dropout):
	print('convolution layer',name)
	print('filter input shape',input.shape)
	images = tf.reshape(input, [-1, IMG_SIZE_X, IMG_SIZE_Y, num_input_layers])

	print('reshaping to',images.shape)
	with tf.variable_scope('conv' + name) as scope:
		#Conv 1
		with tf.name_scope('weights'):
			# ini w 1.0/np.sqrt(num_input_layers*filter_size*filter_size) previously
			W1 = tf.Variable(tf.truncated_normal([filter_size, filter_size, num_input_layers, num_output_layers], stddev=0.1), name='weights_' + name)
			#variable_summaries(W1)
			print("shape of filters",W1.get_shape())

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
			print("shape of output",conv_1.get_shape())
			# tf.summary.image('conv_1',conv_1)

		dim_1 = conv_1.get_shape()[1].value * conv_1.get_shape()[2].value * conv_1.get_shape()[3].value
		pool_1_flat = tf.reshape(conv_1, [-1, dim_1])
		print("shape of output flattened",pool_1_flat.get_shape())






	# dim_1 = pool_1.get_shape()[1].value * pool_1.get_shape()[2].value * pool_1.get_shape()[3].value
	# pool_1_flat = tf.reshape(pool_1, [-1, dim_1])
	#
	# #Conv 2
	# W2 = tf.Variable(tf.truncated_normal([5, 5, 50, 60], stddev=1.0/np.sqrt(50*5*5)), name='weights_2')
	# b2 = tf.Variable(tf.zeros([60]), name='biases_2')
	#
	# #tf.nn.conv2d(input,filter,strides,padding,)
	# conv_2 = tf.nn.relu(tf.nn.conv2d(pool_1, W2, [1, 1, 1, 1], padding='VALID') + b2)
	# pool_2 = tf.nn.max_pool(conv_2, ksize= [1, 2, 2, 1], strides= [1, 2, 2, 1], padding='VALID', name='pool_2')
	#
	# dim_2 = pool_2.get_shape()[1].value * pool_2.get_shape()[2].value * pool_2.get_shape()[3].value
	# pool_2_flat = tf.reshape(pool_2, [-1, dim_2])
	# print("pool_2_flat",pool_2_flat.get_shape())
	# print("pool_2_flat",pool_2_flat.get_shape()[1])
	# #fully connected
	# W3 = tf.Variable(tf.truncated_normal([int(pool_2_flat.get_shape()[1]), 300], stddev=1.0 / np.sqrt(300), dtype=tf.float32), name='weights_3')
	# b3 = tf.Variable(tf.zeros([300]), dtype=tf.float32, name='biases_3')
	# u = tf.add(tf.matmul(pool_2_flat, W3), b3)
	# output_1 = tf.nn.relu(u)
	# print("output_1",output_1.get_shape())
	# #Softmax
	# W4 = tf.Variable(tf.truncated_normal([int(output_1.get_shape()[1]), NUM_CLASSES], stddev=1.0/np.sqrt(NUM_CLASSES)), name='weights_4')
	# b4 = tf.Variable(tf.zeros([NUM_CLASSES]), name='biases_4')
	# logits = tf.matmul(output_1, W4) + b4

	return pool_1_flat

def chainFilter(input, layerlist, filter_size,keep_prob):
	layer = input
	first_layer = True
	for index in range(len(layerlist)-1):

		layer = oneFilter(layer,layerlist[index],layerlist[index+1],filter_size,str(index),keep_prob,first_layer)
		first_layer = False

	return layer

def run(filter_size,layerlist,keep,use_saved_model,reload,exp_name):

	num_layers = len(layerlist)
	now = datetime.now()
	logdir = "./logs/" + exp_name + '/'
	logdir = logdir + str(num_layers-1) + 'layer_'
	logdir = logdir + str(filter_size) + 'x_'
	logdir = logdir + str(layerlist) + 'filter_'
	logdir = logdir + str(keep) + "keepprob/"+ now.strftime("%m%d-%H%M%S")
	logdir = logdir + '/'

	tf.reset_default_graph()

	#load files
	path ="../results/*.json"
	files = getFiles(path)

	trainX_file = './np_save/trainX'
	trainY_file = './np_save/trainY'


	modelpath = "./model/testmodel.ckpt"
	if reload:
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

	numSamples = trainX.shape[0]
	print('type(trainX)',type(trainX))

	#create train test set
	testX, testY = trainX[int(numSamples*0.7):],trainY[int(numSamples*0.7):]
	print('testX.shape',testX.shape, 'testY.shape', testY.shape)
	trainX, trainY = trainX[:int(numSamples*0.7)],trainY[:int(numSamples*0.7)]
	print('trainX.shape',trainX.shape,'trainY.shape',trainY.shape)

	v_min = trainX.min(axis=(0, 1), keepdims=True)
	v_max = trainX.max(axis=(0, 1), keepdims=True)
	trainX = (trainX - v_min)/(v_max - v_min)
	testX = (testX - v_min)/(v_max - v_min)


	# Create the model
	x = tf.placeholder(tf.float32, [None, IMG_SIZE_X*IMG_SIZE_Y*NUM_CHANNELS],name='x')
	y_ = tf.placeholder(tf.float32, [None, IMG_SIZE_X*IMG_SIZE_Y*OUTPUT_CHANNELS],name='y')

	keep_prob = tf.placeholder(tf.float32,name = 'keep_prob')
	logits = chainFilter(x,layerlist,filter_size,keep_prob)



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

	if not use_saved_model:
		with tf.Session() as sess:
			merged = tf.summary.merge_all()
			train_writer = tf.summary.FileWriter(logdir + 'train', sess.graph)
			test_writer = tf.summary.FileWriter(logdir + 'test')

			sess.run(tf.global_variables_initializer())

			for e in range(epochs):
				np.random.shuffle(idx)
				trainX, trainY = trainX[idx], trainY[idx]

				for start, end in zip(range(0, trainX.shape[0], batch_size), range(batch_size, trainX.shape[0], batch_size)):
					sess.run(train_step,feed_dict={x: trainX[start:end], y_: trainY[start:end], keep_prob :keep})

				if e%10 == 0:
					loss_ = loss.eval({x: trainX, y_: trainY, keep_prob : 1.0})
					summary = sess.run(merged, feed_dict={x: trainX, y_: trainY, keep_prob : 1.0})
					train_writer.add_summary(summary, e)
					summary = sess.run(merged, feed_dict={x: testX, y_: testY, keep_prob : 1.0})
					test_writer.add_summary(summary, e)

					print('epoch', e, 'entropy', loss_)

			#save_path = saver.save(sess, modelpath)
			#print("Model saved in path: %s" % save_path)

	else:
		with tf.Session() as sess:
			# Restore variables from disk.
			saver.restore(sess, modelpath)

			results = sess.run(logits,feed_dict={x: testX,keep_prob : 1.0})

			print('len result',len(results))
			print('type result',type(results))

			for pred,truth in zip(results,testY):

				figure = plt.figure(1)
				figure.suptitle('prediction')
				x,y,z = np.reshape(pred,(IMG_SIZE_X,IMG_SIZE_Y,OUTPUT_CHANNELS)).nonzero()
				ax = figure.add_subplot(111, projection='3d')
				axes = plt.gca()
				axes.set_xlim([0,20])
				axes.set_ylim([0,19])
				axes.set_zlim([0,OUTPUT_CHANNELS-1])
				ax.scatter(x, y, z)

				figure = plt.figure(2)
				figure.suptitle('truth')
				reshaped_truth = np.reshape(truth,(IMG_SIZE_X,IMG_SIZE_Y,OUTPUT_CHANNELS))
				print('truth',list(truth))
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

				plt.show()

def generate_layers(layer_space,num_layers,NUM_CHANNELS,OUTPUT_CHANNELS):
	output = []
	for layer in range(num_layers):
		combi = list(itertools.combinations_with_replacement(layer_space,layer))
		combi = [[NUM_CHANNELS] + list(x) + [OUTPUT_CHANNELS] for x in combi]
		output = output + combi

	return output

if __name__ == '__main__':
	reload = False
	use_saved_model = False
	filter_size = 8
	layerlist = [NUM_CHANNELS,30,5,OUTPUT_CHANNELS]

	exp_name = str(NUM_CHANNELS) + 'channel_nolog_grid_norm_drop1_runstest'

	for i in range(10):
		filter_size = 3
		layerlist = [NUM_CHANNELS,5,10,30,OUTPUT_CHANNELS]
		keep = 0.9
		run(filter_size,layerlist,keep,use_saved_model,reload,exp_name)

	for i in range(10):
		filter_size = 3
		layerlist = [NUM_CHANNELS,10,15,15,OUTPUT_CHANNELS]
		keep = 0.7
		run(filter_size,layerlist,keep,use_saved_model,reload,exp_name)

	for i in range(10):
		filter_size = 3
		layerlist = [NUM_CHANNELS,10,15,15,30,OUTPUT_CHANNELS]
		keep = 0.7
		run(filter_size,layerlist,keep,use_saved_model,reload,exp_name)

	# layer_space = [5,10,20,30]
	# num_layers = 4
	# iter_layers = generate_layers(layer_space,num_layers,NUM_CHANNELS,OUTPUT_CHANNELS)
	#
	# iter_keep = [1.0,0.9,0.7]
	#
	# iter_filter_size = [3,5,7]
	#
	# iteration_length = len(iter_filter_size)*len(iter_keep)*len(iter_layers)
	# print('total iterations', iteration_length)
	# for runs in range(5):
	# 	for keep in iter_keep:
	# 		for layerlist in iter_layers:
	# 			for filter_size in iter_filter_size:
	# 				run(filter_size,layerlist,keep,use_saved_model,reload,exp_name)
