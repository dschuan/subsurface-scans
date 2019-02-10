from processJSON import loadIntoArray
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

NUM_POINTS = 4096
def processBackground(debug = False,reload = False):

    save_dest = './np_save/background_signal'
    if reload:
        backgrounddata = loadIntoArray('../otherresults/07_02No_Target')
        numscans = backgrounddata.shape[0] * backgrounddata.shape[1]
        background_signal = np.sum(backgrounddata, axis = (0,1)) / numscans
        np.save(save_dest, background_signal)

    else:
        background_signal =  np.load(save_dest+'.npy')



    if debug: print('backgrounddata.shape',backgrounddata.shape)



    return background_signal

def cleanTarget(input_signal,background_signal):
    cleaned_signal = input_signal - background_signal[None,None,:]

    return cleaned_signal

def re_init():
    processBackground(debug = True,reload = True)

def convert_to_complex(cleaned_signal,plot = False):


    cleaned_signal_fft = np.fft.fft(cleaned_signal)
    print('cleaned_signal_fft shape',cleaned_signal_fft.shape)

    half_cleaned_signal_fft = np.copy(cleaned_signal_fft)
    half_cleaned_signal_fft[:,:,int(NUM_POINTS/2):] = 0 + 0j

    complex_clean_signal = np.fft.ifft(half_cleaned_signal_fft)


    if plot:
        plt.figure(1)
        plt.plot(range(4096),original[0][0])
        plt.plot(range(4096),cleaned_signal[0][0])

        plt.legend(['original', 'cleaned_signal'])

        plt.figure(2)
        plt.plot(range(4096),background)
        plt.legend(['background'])

        plt.figure(3)
        plt.plot(range(4096),half_cleaned_signal_fft[0][0])
        plt.legend(['half_cleaned_signal_fft'])

        plt.figure(4)
        plt.plot(range(4096),complex_clean_signal[0][0])
        plt.legend(['complex_clean_signal'])


        plt.show()

    return complex_clean_signal


def back_projection(complex_clean_signal,reload = False):
    offset = 71
    delta_distance = 0.0015 # timestep * speed_light/2
    resolution = 0.01

    x_range = np.arange(0.0, 0.2, resolution)
    y_range = np.arange(0.0, 0.21, resolution)
    z_range = np.arange(-0.4, 0, resolution)

    distance_r_to_t = 0.02
    x_radar_range = np.arange(0.0,0.2,resolution)
    y_radar_range = np.arange(0.0,0.21,resolution)

    image = np.zeros((len(x_range),len(y_range),len(z_range)))

    save_dest = './np_save/backproj_image'
    if reload:
        for index,x_pos in enumerate(x_range):
            print('backprojection loading',index/len(x_range))
            for y_pos in y_range:
                for z_pos in z_range:
                    accumulator = 0
                    for x_radar_pos in x_radar_range:
                        for y_radar_pos in y_radar_range:
                            point = np.array((x_pos ,y_pos, z_pos))
                            radarT = np.array((x_radar_pos + distance_r_to_t/2, y_radar_pos, 0))
                            radarR = np.array((x_radar_pos - distance_r_to_t/2, y_radar_pos, 0))
                            distance = np.linalg.norm(radarT-point) + np.linalg.norm(radarR-point)
                            distance = distance/2
                            scan_index = int(distance/delta_distance)
                            contribution = complex_clean_signal[int(x_radar_pos*100)][int(y_radar_pos*100)][scan_index]
                            accumulator += contribution
                    image[int(x_pos*100)][int(y_pos*100)][int(z_pos*100)] = abs(accumulator)
        np.save(save_dest, image)


    else:
        image =  np.load(save_dest+'.npy')

    print('image shape',image.shape)

    image_min = image.min( keepdims=True)
    image_max = image.max( keepdims=True)

    norm_image = (image - image_min)/(image_max-image_min)

    x,y,z = (norm_image>0.6).nonzero()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    axes = plt.gca()
    axes.set_xlim([0,20])
    axes.set_ylim([0,19])
    axes.set_zlim([0,40])

    ax.scatter(x, y, z, zdir='z',c= 'red')
    # figure = plt.figure(7)
    # figure.suptitle('image')
    # ax = figure.add_subplot(111, projection='3d')
    # axes = plt.gca()
    # x = np.arange(image.shape[0])[:, None, None]
    # y = np.arange(image.shape[1])[None, :, None]
    # z = np.arange(image.shape[2])[None, None, :]
    # x, y, z = np.broadcast_arrays(x, y, z)
    # c = np.tile(image.ravel()[:, None], [1, 3])
    # ax.scatter(x.ravel(),
    #    y.ravel(),
    #    z.ravel(),
    #    c=image.ravel(),
    #    cmap=plt.get_cmap('Reds'))
    plt.show()

if __name__ == "__main__":
    re_init() #does backprop and save it, takes a wwhile but only need to run once
    background = processBackground()
    original = loadIntoArray('../otherresults/07_02Aluminum')
    cleaned_signal = cleanTarget(original,background)
    complex_clean_signal = convert_to_complex(cleaned_signal,plot = False)
    back_projection(complex_clean_signal,reload = False)
