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
IMG_SIZE_Z = 40
NUM_CHANNELS = 1
OUTPUT_CHANNELS = 1

#ml variables
learning_rate = 0.01
epochs = 1000
batch_size = 1

# raw json processing variables
# SAMPLE_SIZE = 1000
SAMPLE_KEEP_PROB = 1

PRINT_INFO = True

# seed = int(time.time())
# np.random.seed(seed)
tf.set_random_seed(10)

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


def oneFilter(input,num_input_layers,num_output_layers,filter_depth,filter_size,name,keep_prob,apply_dropout,lastlayer):

	images = tf.reshape(input, [-1, IMG_SIZE_Z, IMG_SIZE_X, IMG_SIZE_Y, num_input_layers])


	with tf.variable_scope('conv' + name) as scope:
		with tf.name_scope('weights'):
			init = tf.truncated_normal([filter_depth, filter_size, filter_size, num_input_layers, num_output_layers], mean = 0.2,stddev=0.01)
			# init = tf.constant()
			regularizer = tf.contrib.layers.l2_regularizer(scale=1e-6)
			W1 = tf.get_variable( name='weights_' + name, initializer=init,trainable=True,regularizer = regularizer)

			#variable_summaries(W1)


		with tf.name_scope('biases'):
			b1 = tf.Variable(tf.zeros([num_output_layers]), name='biases_' + name)
			#variable_summaries(b1)

		with tf.name_scope('output'):
			# With the default format "NDHWC", the data is stored in the order of: [batch, in_depth, in_height, in_width, in_channels]
			if lastlayer:
				conv_1 = tf.nn.conv3d(images, W1, [1, 1, 1, 1,1], padding='SAME') + b1
			else:
				conv_1 = tf.nn.relu(tf.nn.conv3d(images, W1, [1, 1, 1, 1,1], padding='SAME') + b1)
			if apply_dropout:
				conv_1 = tf.nn.dropout(conv_1, keep_prob)



		dim_1 = conv_1.get_shape()[1].value * conv_1.get_shape()[2].value * conv_1.get_shape()[3].value* conv_1.get_shape()[4].value
		pool_1_flat = tf.reshape(conv_1, [-1, dim_1])


		if(PRINT_INFO):
			print('convolution layer',name)
			print('filter input shape',input.shape)
			print('reshaping to',images.shape)
			print("shape of filters",W1.get_shape())
			print("shape of output",conv_1.get_shape())
			print("shape of output flattened",pool_1_flat.get_shape())

	return pool_1_flat


def backPropFilter(input,num_input_layers,num_backproj_filter,filter_depth,filter_size,name,keep_prob,apply_dropout):

	images = tf.reshape(input, [-1, IMG_SIZE_Z, IMG_SIZE_X, IMG_SIZE_Y, num_input_layers])


	with tf.variable_scope('conv' + name) as scope:
		with tf.name_scope('weights'):
			# init = tf.truncated_normal([filter_depth, filter_size, filter_size, num_input_layers, num_backproj_filter],mean = 0.01,stddev = 0.001)
			init = tf.ones([filter_depth, filter_size, filter_size, num_input_layers, num_backproj_filter])
			print('type init',type(init))
			print('init shape',init.shape)
			regularizer = tf.contrib.layers.l2_regularizer(scale=1e-6)
			W1 = tf.get_variable(  name='backproj_weights_' + name, initializer=init,trainable=True,regularizer = regularizer)
						#variable_summaries(W1)
		mask = create_filter(filter_z_len = filter_depth,filter_x_len = filter_size,filter_y_len = filter_size,distance_to_top = 14,visualise = False)
		mask = np.float32(mask)

		mask = mask.reshape( [filter_depth,filter_size,filter_size,num_input_layers,1] )
		mask = tf.concat([mask for i in range(num_backproj_filter)], axis=4)

		mask_variable = tf.Variable( mask , dtype=tf.float32,name='mask_' + name )
		mask = tf.stop_gradient( mask_variable )



		with tf.name_scope('biases'):
			b1 = tf.Variable(tf.zeros([num_backproj_filter]), name='biases_' + name)
			#variable_summaries(b1)

		with tf.name_scope('output'):
			# With the default format "NDHWC", the data is stored in the order of: [batch, in_depth, in_height, in_width, in_channels]
			conv_1 = tf.nn.relu(tf.nn.conv3d(images, W1*mask, [1, 1, 1, 1,1], padding='SAME') + b1)
			if apply_dropout:
				conv_1 = tf.nn.dropout(conv_1, keep_prob)

			# tf.summary.image('conv_1',conv_1)
		print("shape of output",conv_1.get_shape())


		dim_1 = conv_1.get_shape()[1].value * conv_1.get_shape()[2].value * conv_1.get_shape()[3].value* conv_1.get_shape()[4].value
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
	print('Loading test data testX.shape',testX.shape, 'testY.shape', testY.shape)
	trainX, trainY = trainX[:int(numSamples*train_test_split)],trainY[:int(numSamples*train_test_split)]
	print('Loading train data trainX.shape',trainX.shape,'trainY.shape',trainY.shape)

	# TODO: CHECK THIS NORM!(11, 60, 20, 21)
	v_min = trainX.min(axis=(0, 1, 2, 3), keepdims=True)
	v_max = trainX.max(axis=(0, 1, 2, 3), keepdims=True)
	trainX = (trainX - v_min)/(v_max - v_min)
	testX = (testX - v_min)/(v_max - v_min)

	return [flatten4D(trainX), flatten4D(testX),flatten4D(trainY), flatten4D(testY)]

def flatten4D(nparray):
	return nparray.reshape(nparray.shape[0],-1)

plot_totals = 50
def plot_threed(plot_object,name):
	global plot_totals
	figure = plt.figure(plot_totals)
	plot_totals = plot_totals + 1
	figure.suptitle(name)
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
		if item > 0.1:
			filtered_x.append(ravel_x[index])
			filtered_y.append(ravel_y[index])
			filtered_z.append(ravel_z[index])
			filtered_colour.append(item)

	if len(filtered_colour) > 0:
		scatter = ax.scatter(filtered_x,filtered_y,filtered_z,c=filtered_colour,cmap=plt.get_cmap('coolwarm'))
		figure.colorbar(scatter)
	else:
		print('empty array received')


def run(trainX, testX, trainY, testY, num_backproj_filter,backproj_filter_depth,backproj_filter_size,keep,filter_height,filter_size, use_saved_model, exp_name, show_results):
	modelpath = "./model/testmodel.ckpt"
	print('Using test data testX.shape',testX.shape, 'testY.shape', testY.shape)
	if not show_results:
		print('Using train data trainX.shape',trainX.shape,'trainY.shape',trainY.shape)

	now = datetime.now()
	logdir = "./logs/" + exp_name + '/'
	logdir = logdir + str(num_backproj_filter) + 'filter_' + str(backproj_filter_depth) + 'x' +str(backproj_filter_size)  + 'x' +str(backproj_filter_size) + '_'
	logdir = logdir + str(filter_height) + 'x' +str(filter_size)  + 'x' +str(filter_size) + '_'
	logdir = logdir + str(keep) + "keepprob/"+ now.strftime("%m%d-%H%M%S")
	logdir = logdir + '/'
	print('params',logdir)
	tf.reset_default_graph()

	# Create the model
	x = tf.placeholder(tf.float32, [None, IMG_SIZE_Z*IMG_SIZE_X*IMG_SIZE_Y*NUM_CHANNELS],name='x')
	y_ = tf.placeholder(tf.float32, [None, IMG_SIZE_Z*IMG_SIZE_X*IMG_SIZE_Y*OUTPUT_CHANNELS],name='y')

	keep_prob = tf.placeholder(tf.float32,name = 'keep_prob')

	layer_1_args = {
		'input': x,
		'num_input_layers': NUM_CHANNELS,
		'num_backproj_filter': num_backproj_filter,
		'filter_depth': backproj_filter_depth,
		'filter_size': backproj_filter_size,
		'name': '1',
		'keep_prob': keep_prob,
		'apply_dropout': True
	}
	logits = backPropFilter(**layer_1_args)
	layer_1_args['name'] = '2'
	backproj = backPropFilter(**layer_1_args)
	#
	# layer_2_args = {
	# 	'input': backproj,
	# 	'num_input_layers': num_backproj_filter,
	# 	'num_output_layers': OUTPUT_CHANNELS,
	# 	'filter_depth': filter_height,
	# 	'filter_size': filter_size,
	# 	'name': '2',
	# 	'keep_prob': keep_prob,
	# 	'apply_dropout': False,
	# 	'lastlayer': True
	# }
	#
	# logits = oneFilter(**layer_2_args)

	print("labels y_.shape",y_.shape)
	print("logits logits.shape", logits.shape)
	y_shaped = tf.reshape(y_, [-1, IMG_SIZE_Z, IMG_SIZE_X, IMG_SIZE_Y, OUTPUT_CHANNELS])[:,12:15,:,:,:]
	logits_shaped = tf.reshape(logits, [-1, IMG_SIZE_Z, IMG_SIZE_X, IMG_SIZE_Y, OUTPUT_CHANNELS])[:,12:15,:,:,:]

	with tf.name_scope('loss'):
		# loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=y_))#+ tf.losses.get_regularization_loss()

		loss = tf.reduce_mean(tf.squared_difference(logits_shaped, y_shaped))#+ tf.losses.get_regularization_loss()
	tf.summary.scalar('loss', loss)
	# tf.summary.scalar('regloss', tf.losses.get_regularization_loss())


	# correct_prediction = tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1)), tf.float32)
	# accuracy = tf.reduce_mean(correct_prediction)

	# train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
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

				if e%10 == 0:
					loss_ = loss.eval({x: trainX, y_: trainY, keep_prob : 1.0})
					test_loss_ = loss.eval({x: testX, y_: testY, keep_prob : 1.0})
					summary = sess.run(merged, feed_dict={x: trainX, y_: trainY, keep_prob : 1.0})
					train_writer.add_summary(summary, e)
					summary = sess.run(merged, feed_dict={x: testX, y_: testY, keep_prob : 1.0})
					test_writer.add_summary(summary, e)

					print('epoch', e, 'loss', loss_,'test_loss',test_loss_,'reg loss',tf.losses.get_regularization_loss().eval())

				if e%50 ==0:
					results, backproj_result = sess.run([logits_shaped,backproj],feed_dict={x: trainX,keep_prob : 1.0})
					count = 0
					for raw_,pred_,truth_,backproj_pred_ in zip(testX,results,testY,backproj_result):
						count = count + 1
						if count>3:
							break
						raw = np.reshape(raw_,(IMG_SIZE_Z,IMG_SIZE_X,IMG_SIZE_Y))[10:20,:,:].copy()
						raw = np.transpose(raw, (1,2,0)).copy()
						pred = np.reshape(pred_,(pred_.shape[0],pred_.shape[1],pred_.shape[2]))[:,:,:].copy()
						pred = np.transpose(pred, (1,2,0)).copy()
						backproj_pred = np.reshape(backproj_pred_,(IMG_SIZE_Z,IMG_SIZE_X,IMG_SIZE_Y,num_backproj_filter))[:,:,:,0].copy()
						backproj_pred = np.transpose(backproj_pred, (1,2,0)).copy()
						truth = np.reshape(truth_,(IMG_SIZE_Z,IMG_SIZE_X,IMG_SIZE_Y))[10:20,:,:].copy()
						truth = np.transpose(truth, (1,2,0)).copy()
						plot_threed(pred,'prediction')
						plt.savefig('./plot/' +'pred' +now.strftime("%m%d-%H%M") + '_#'  + str(count) + '_epoch_'+ str(e)+  '.png')
						plot_threed(backproj_pred,'backproj_pred')
						plt.savefig('./plot/' +'backproj' +now.strftime("%m%d-%H%M")   + '_#' + str(count) + '_epoch_'+ str(e)+'.png')



			save_path = saver.save(sess, modelpath)
			print("Model saved in path: %s" % save_path)

	else:
		with tf.Session() as sess:
			# Restore variables from disk.
			saver.restore(sess, modelpath)

			results, backproj_result = sess.run([logits_shaped,backproj],feed_dict={x: testX,keep_prob : 1.0})

			print('len result',len(results))
			print('type result',type(results))

			for raw_,pred_,truth_,backproj_pred_ in zip(testX,results,testY,backproj_result):
				print('pred_',pred_.shape)
				raw = np.reshape(raw_,(IMG_SIZE_Z,IMG_SIZE_X,IMG_SIZE_Y))[10:20,:,:].copy()
				raw = np.transpose(raw, (1,2,0)).copy()
				pred = np.reshape(pred_,(pred_.shape[0],pred_.shape[1],pred_.shape[2]))[:,:,:].copy()
				pred = np.transpose(pred, (1,2,0)).copy()
				backproj_pred = np.reshape(backproj_pred_,(backproj_pred_.shape[0],backproj_pred_.shape[1],backproj_pred_.shape[2]))[:,:,:].copy()
				backproj_pred = np.transpose(backproj_pred, (1,2,0)).copy()
				truth = np.reshape(truth_,(IMG_SIZE_Z,IMG_SIZE_X,IMG_SIZE_Y))[10:20,:,:].copy()
				truth = np.transpose(truth, (1,2,0)).copy()

				print('raw',raw.shape)

				figure = plt.figure(0)
				figure.suptitle('pred 0')
				x,y,z = (pred > 0.5).nonzero()
				ax = figure.add_subplot(111, projection='3d')
				axes = plt.gca()
				axes.set_xlim([0,IMG_SIZE_X - 1])
				axes.set_ylim([0,IMG_SIZE_Y - 1])
				axes.set_zlim([0,11])
				ax.scatter(x, y, z)


				plot_threed(pred,'prediction')

				plot_threed(backproj_pred,'backproj_pred')


				figure = plt.figure(2)
				figure.suptitle('truth')
				# print('truth',list(truth))
				x,y,z = truth.nonzero()
				ax = figure.add_subplot(111, projection='3d')
				axes = plt.gca()
				axes.set_xlim([0,IMG_SIZE_X - 1])
				axes.set_ylim([0,IMG_SIZE_Y - 1])
				axes.set_zlim([0,11])
				ax.scatter(x, y, z)

				figure = plt.figure(3)
				figure.suptitle('prediction flat')
				axes = plt.gca()
				axes.set_xlim([0,IMG_SIZE_X - 1])
				axes.set_ylim([0,IMG_SIZE_Y - 1])

				plt.imshow(pred.sum(axis=(2)))

				figure = plt.figure(4)
				figure.suptitle('truth flat')

				axes = plt.gca()
				axes.set_xlim([0,IMG_SIZE_X - 1])
				axes.set_ylim([0,IMG_SIZE_Y - 1])

				plt.imshow(truth.sum(axis=(2)))



				figure = plt.figure(6)
				figure.suptitle('raw14')
				axes = plt.gca()
				axes.set_xlim([0,IMG_SIZE_X - 1])
				axes.set_ylim([0,IMG_SIZE_Y - 1])

				plt.imshow(raw[:,:,1])

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

	for num_items in range(2):
		loadData(reload = True, max_items_per_scan = num_items+1, only_max = True)



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

	model_params['show_results'] = False
	run(trainX, testX, trainY, testY, **model_params)

	model_params['show_results'] = True
	run(trainX, trainX, trainY, trainY, **model_params)
	# run(trainX, testX, trainY, testY, **model_params)
