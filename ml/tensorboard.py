import math
import tensorflow as tf
import numpy as np
import pylab as plt
import pickle
import random
import json
from processJSON import processJSON
from datetime import datetime

now = datetime.now()
logdir = "./logs/" + now.strftime("%Y%m%d-%H%M%S") + "/"

#size of input
IMG_SIZE_X = 21
IMG_SIZE_Y = 5
NUM_CHANNELS = 2

#ml variables
learning_rate = 0.005
epochs = 1000
batch_size = 128

# raw json processing variables
SAMPLE_SIZE = 1000
SAMPLE_KEEP_PROB = 0.9

seed = 10
np.random.seed(seed)
tf.set_random_seed(seed)

def load_data(file):
	with open(file) as f:
		data = json.load(f)
		scanArray,truthArray = processJSON(data)

	X = [sampleArray(scanArray).flatten() for i in range(SAMPLE_SIZE)]
	Y = [truthArray.flatten() for i in range(SAMPLE_SIZE)]

	return (X,Y)


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


def oneFilter(images):

	NUM_FILTERS = 1
	FILTER_SIZE = 5
	images = tf.reshape(images, [-1, IMG_SIZE_X, IMG_SIZE_Y, NUM_CHANNELS])

	with tf.variable_scope('conv1') as scope:
		#Conv 1
		with tf.name_scope('weights'):
			W1 = tf.Variable(tf.truncated_normal([FILTER_SIZE, FILTER_SIZE, NUM_CHANNELS, NUM_FILTERS], stddev=1.0/np.sqrt(NUM_CHANNELS*FILTER_SIZE*FILTER_SIZE)), name='weights_1')
			variable_summaries(W1)
			print("w1",W1.get_shape())

			filterImage, presenceImage = tf.split(W1, num_or_size_splits=2, axis=2)
			print("filterImage",filterImage.get_shape())
			filterImage = tf.reshape (filterImage, [-1, FILTER_SIZE, FILTER_SIZE, 1])
			presenceImage = tf.reshape (presenceImage, [-1, FILTER_SIZE, FILTER_SIZE, 1])
			print("filterImage",filterImage.get_shape())
			tf.summary.image('filterImage',filterImage)
			tf.summary.image('presenceImage',presenceImage)

		with tf.name_scope('biases'):
			b1 = tf.Variable(tf.zeros([NUM_FILTERS]), name='biases_1')
			variable_summaries(b1)

		print("images",images.get_shape())

		#tf.nn.conv2d(input,filter,strides,padding,)
		with tf.name_scope('output'):
			conv_1 = tf.nn.relu(tf.nn.conv2d(images, W1, [1, 1, 1, 1], padding='SAME') + b1)
			print("conv_1",conv_1.get_shape())
			tf.summary.image('conv_1',conv_1)

		dim_1 = conv_1.get_shape()[1].value * conv_1.get_shape()[2].value * conv_1.get_shape()[3].value
		pool_1_flat = tf.reshape(conv_1, [-1, dim_1])
		print("pool_1_flat",pool_1_flat.get_shape())






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


def main():

	trainX, trainY = load_data('../results/dec_corner_reflector_openbox4.json')
	trainX = np.array(trainX)
	trainY = np.array(trainY)
	numSamples = trainX.shape[0]
	print(type(trainX))

	testX, testY = trainX[int(numSamples*0.7):],trainY[int(numSamples*0.7):]
	print(testX.shape, testY.shape)
	trainX, trainY = trainX[:int(numSamples*0.7)],trainY[:int(numSamples*0.7)]
	print(trainX.shape, trainY.shape)

	# testX = (testX - np.min(trainX, axis = 0))/np.max(trainX, axis = 0)
	# trainX = (trainX - np.min(trainX, axis = 0))/np.max(trainX, axis = 0)

	# Create the model
	x = tf.placeholder(tf.float32, [None, IMG_SIZE_X*IMG_SIZE_Y*NUM_CHANNELS])
	y_ = tf.placeholder(tf.float32, [None, IMG_SIZE_X*IMG_SIZE_Y])


	logits = oneFilter(x)
	print("labels",y_.shape)
	print("logits", logits.shape)
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


	with tf.Session() as sess:
		merged = tf.summary.merge_all()
		train_writer = tf.summary.FileWriter(logdir + 'train', sess.graph)
		test_writer = tf.summary.FileWriter(logdir + 'test')

		sess.run(tf.global_variables_initializer())

		for e in range(epochs):
			np.random.shuffle(idx)
			trainX, trainY = trainX[idx], trainY[idx]

			for start, end in zip(range(0, trainX.shape[0], batch_size), range(batch_size, trainX.shape[0], batch_size)):
				sess.run(train_step,feed_dict={x: trainX[start:end], y_: trainY[start:end]})

			if e%10 == 0:
				loss_ = loss.eval({x: trainX, y_: trainY})
				summary = sess.run(merged, feed_dict={x: trainX, y_: trainY})
				train_writer.add_summary(summary, e)
				summary = sess.run(merged, feed_dict={x: testX, y_: testY})
				test_writer.add_summary(summary, e)

				print('epoch', e, 'entropy', loss_)

		save_path = saver.save(sess, "./model/testmodel.ckpt")
		print("Model saved in path: %s" % save_path)




if __name__ == '__main__':
  main()
