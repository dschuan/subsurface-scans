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
from processJSON import getTruthArray
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

modelpath = "./model/testmodel.ckpt"

saver = tf.train.import_meta_graph(modelpath + '.meta')
fignum = 0
def plot_three_d(plot_object,plotname,threshold = 0.1):
    global fignum
    figure = plt.figure(fignum)
    fignum = fignum + 1
    figure.suptitle( plotname)
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
    print('ravel_x',ravel_x.shape)
    for index,item in enumerate(plot_object.ravel()):
        if item > threshold:
            filtered_x.append(ravel_x[index])
            filtered_y.append(ravel_y[index])
            filtered_z.append(ravel_z[index])
            filtered_colour.append(item)
    if len(filtered_colour) > 0:
        scatter = ax.scatter(filtered_x,filtered_y,filtered_z,c=filtered_colour,cmap=plt.get_cmap('coolwarm'))
        figure.colorbar(scatter)
    else:
        print('empty array received')

    figure = plt.figure(fignum)
    fignum = fignum + 1
    figure.suptitle('weight distribution ' + plotname)
    plt.hist(filtered_colour)


def check_backpropraw_file():

    path = './backprop/*'
    all_files = glob.glob(path)
    print('all_files',all_files)
    for file in all_files:
        filename, file_extension = os.path.splitext(file)
        filename = filename.split('\\')[-1]
        print(filename)
        truth = getTruthArray('../new_res/'+filename,threedim = True)

        backprop =  np.load(file)
        plot_three_d(backprop,'backprop',threshold = 0.7)
        plot_three_d(truth,'truth',threshold = 0.4)
        plt.show()

#
# with tf.Session() as sess:
#     print('session started')
#
#     print('using saved model at',modelpath)
#     saver.restore(sess, modelpath)
#
#     vars = [v for v in tf.trainable_variables()]
#     for var in vars:
#         print(var)
#
#     print(var)
#
#     weights_var = tf.get_default_graph().get_tensor_by_name('conv1/backproj_weights_1:0').eval()
#
#
#     print('weights_var type',type(weights_var))
#     print('weights_var shape',weights_var.shape)
#
#
#     for i in range(weights_var.shape[4]):
#         weights = weights_var[:,:,:,:,i]
#         print('weights shape',weights.shape)
#         weights = np.reshape(weights,(weights.shape[0],weights.shape[1],weights.shape[2]))
#         mask = tf.get_default_graph().get_tensor_by_name("conv1/mask_1:0").eval()
#         print('mask shape',mask.shape)
#         mask = np.reshape(mask[:,:,:,:,i],(mask.shape[0],mask.shape[1],mask.shape[2]))
#         masked_weights = mask * weights
#         plot_three_d(masked_weights,'masked_weights' + str(i))
#
#     # filterbase = create_filter(filter_z_len = 20,filter_x_len = 20,filter_y_len = 20,distance_to_top = 14,visualise = False)
#     # plot_three_d(filterbase,'filterbase')
#     # z,x,y = plot_object.nonzero()
#     # fig = plt.figure(0)
#     # fig.suptitle('1')
#     # ax = fig.add_subplot(111, projection='3d')
#     # axes = plt.gca()
#     # axes.set_xlim([0,plot_object.shape[1]])
#     # axes.set_ylim([0,plot_object.shape[2]])
#     # axes.set_zlim([0,plot_object.shape[0]])
#     #
#     # ax.scatter(x, y, z,c=plot_object.ravel())
#
#     plt.show()
