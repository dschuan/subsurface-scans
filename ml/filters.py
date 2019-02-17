from processJSON import loadIntoArray, getTruthArray
from scipy import fft
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from ast import literal_eval as make_tuple
import glob
import os
from mpl_toolkits.mplot3d import Axes3D
import math



def create_filter(filter_z_len = 70,filter_x_len = 40,filter_y_len = 40,distance_to_top = 14,visualise = False):





    filter = np.zeros((filter_x_len,filter_y_len,filter_z_len))

    point_x = int(filter_x_len/2)
    point_y = int(filter_y_len/2)
    point_z = int(filter_z_len/2)




    for x_pos in range(filter_x_len):
        for y_pos in range(filter_y_len):


            index = math.sqrt( (distance_to_top*distance_to_top) + (x_pos-point_x)*(x_pos-point_x) + (y_pos-point_y)*(y_pos-point_y))
            index = index - distance_to_top + point_z
            filter[x_pos][y_pos][int(index)] = 1

    if visualise:
        x,y,z = filter.nonzero()
        fig = plt.figure(1)
        fig.suptitle('filter at cm ' + str(distance_to_top))
        ax = fig.add_subplot(111, projection='3d')
        axes = plt.gca()
        axes.set_xlim([0,filter_x_len])
        axes.set_ylim([0,filter_y_len])
        axes.set_zlim([0,filter_z_len])

        ax.scatter(x, y, z, zdir='z',c= 'blue')
        plt.show()

    filter = np.transpose(filter, (2, 0, 1))

    return filter

if __name__ == "__main__":
    create_filter()
