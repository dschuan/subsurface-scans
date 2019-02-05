import math
import tensorflow as tf
import numpy as np
import pylab as plt
import pickle
import random
import json
from processJSON import processJSON

#size of input
IMG_SIZE_X = 21
IMG_SIZE_Y = 5
NUM_CHANNELS = 2

#ml variables
learning_rate = 0.001
epochs = 1000
batch_size = 128

# raw json processing variables
SAMPLE_SIZE = 1000
SAMPLE_KEEP_PROB = 0.3

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


def oneFilter(images):

	NUM_FILTERS = 1
	FILTER_SIZE = 5
	images = tf.reshape(images, [-1, IMG_SIZE_X, IMG_SIZE_Y, NUM_CHANNELS])

	#Conv 1
	W1 = tf.Variable(tf.truncated_normal([FILTER_SIZE, FILTER_SIZE, NUM_CHANNELS, NUM_FILTERS], stddev=1.0/np.sqrt(NUM_CHANNELS*FILTER_SIZE*FILTER_SIZE)), name='weights_1')
	b1 = tf.Variable(tf.zeros([NUM_FILTERS]), name='biases_1')
	print("images",images.get_shape())
	#tf.nn.conv2d(input,filter,strides,padding,)
	conv_1 = tf.nn.relu(tf.nn.conv2d(images, W1, [1, 1, 1, 1], padding='SAME') + b1)
	print("conv_1",conv_1.get_shape())

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

	loss = tf.reduce_mean(tf.squared_difference(logits, y_))


	# correct_prediction = tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1)), tf.float32)
	# accuracy = tf.reduce_mean(correct_prediction)

	#train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
	train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
	N = len(trainX)
	idx = np.arange(N)
	lossArr = []
	testArr = []
	saver = tf.train.Saver()
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		for e in range(epochs):
			np.random.shuffle(idx)
			trainX, trainY = trainX[idx], trainY[idx]

			for start, end in zip(range(0, trainX.shape[0], batch_size), range(batch_size, trainX.shape[0], batch_size)):
				sess.run([train_step, loss],feed_dict={x: trainX[start:end], y_: trainY[start:end]})


			loss_ = loss.eval({x: trainX, y_: trainY})
			lossArr.append(loss_)
			testArr.append(loss.eval(feed_dict={x: testX, y_: testY}))
			if e%100 == 0:

				print('epoch', e, 'entropy', loss_)

		save_path = saver.save(sess, "./testmodel.ckpt")
		print("Model saved in path: %s" % save_path)


		plt.figure(1)
		plt.plot(range(epochs), lossArr)
		plt.xlabel(str(epochs) + ' iterations')
		plt.ylabel('Cross Entropy (Loss)')

		plt.figure(2)
		plt.plot(range(epochs), testArr)
		plt.xlabel(str(epochs) + ' iterations')
		plt.ylabel('Test Loss')

		plt.show()

		ind = 5
		exampleX = trainX[ind,:].reshape([-1,210])
		exampleY = logits.eval(feed_dict = {x:exampleX})
		print("exampleY shape",exampleY.shape)
		plt.figure()
		plt.gray()
		exampleY = exampleY.reshape(21, 5)
		plt.axis('off')
		plt.imshow(exampleY)
		plt.savefig('./prediction/' + str(SAMPLE_KEEP_PROB) + '.png')


if __name__ == '__main__':
  main()
