from processJSON import loadIntoArray, getTruthArray, getFiles
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
PRINT_INFO = True

def processArray(file,use_backproj = True,use_preprocessed = False):

	truth = getTruthArray(file,True)
	filename = file.split('\\')[-1]
	if PRINT_INFO:
		print('processArray truth from processJSON truthshape',truth.shape)
		print('processArray looking at', filename)


	if not use_backproj:
		print("getting clean signal data")
		return (get_data(file,use_backproj = False),truth)

	elif use_preprocessed:
		print("Using preprocessed backproj data")
		return (np.load('./backprojraw/' + filename + '.npy'),truth)

	else:
		print("Reloading backproj data")
		cfm = 'no'
		while cfm != 'yes':
			cfm = input("long process to init, press yes to confirm")
		return (get_data(file),truth)





def processBackground(debug = False,reload = False):

	save_dest = './np_save/background_signal'
	if reload:
		backgrounddata = loadIntoArray('../otherresults/07_02No_Target')
		numscans = backgrounddata.shape[0] * backgrounddata.shape[1]
		background_signal = np.sum(backgrounddata, axis = (0,1)) / numscans
		np.save(save_dest, background_signal)

	else:
		background_signal =  np.load(save_dest+'.npy')



	if debug: print('processArray backgrounddata.shape',backgrounddata.shape)



	return background_signal

def cleanTarget(input_signal,background_signal):
	cleaned_signal = input_signal - background_signal[None,None,:]

	return cleaned_signal

def re_init():
	processBackground(debug = True,reload = True)

def convert_to_complex(cleaned_signal,plot = False):


	cleaned_signal_fft = np.fft.fft(cleaned_signal)
	print('processArray cleaned_signal_fft shape',cleaned_signal_fft.shape)

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

def sample_clean_signal(complex_clean_signal,offset = 71, sampling_step = 7,num_samples = 40):
	#(20, 21, 4096)
	sampled_signal = complex_clean_signal[:,:,offset::sampling_step].copy()
	sampled_signal = sampled_signal[:,:,:num_samples].copy()

	if PRINT_INFO:
		print("processArray sampled_signal shape",sampled_signal.shape)
	return sampled_signal

def back_projection(complex_clean_signal,name,max_plane_offset = -1,reload = False,plot_images = False):
	offset = 75
	delta_distance = 9.765625e-12 * 3e8 / 2 # timestep * speed_light/2
	resolution = 0.01

	x_range = np.arange(0.0, 0.2, resolution)
	y_range = np.arange(0.0, 0.21, resolution)
	z_range = np.arange(0.0, 0.4, resolution)

	distance_r_to_t = 0.02
	x_radar_range = np.arange(0.0,0.2,resolution)
	y_radar_range = np.arange(0.0,0.21,resolution)

	image = np.zeros((len(x_range),len(y_range),len(z_range)))
	filename, file_extension = os.path.splitext(name)
	filename = filename.split('\\')[-1]
	save_dest = './backprojraw/' + filename #+ str(max_plane_offset)
	print('processArray backproj save_dest',save_dest)
	first = True

	if reload:
		points_considered = 0
		iters = 0
		max_point_considered = 0
		for index,x_pos in enumerate(x_range):

			print('processArray backprojection loading',index/len(x_range))
			for y_pos in y_range:
				for z_pos in z_range:
					accumulator = 0


					iters = iters+1
					iter_points_considered = 0
					for x_radar_pos in x_radar_range:
						for y_radar_pos in y_radar_range:
							plane_offset =  np.linalg.norm( np.array((x_pos ,y_pos))- np.array((x_radar_pos , y_radar_pos)))
							if max_plane_offset != -1 and plane_offset*100 > max_plane_offset:
								continue
							iter_points_considered = iter_points_considered + 1
							points_considered = points_considered + 1
							point = np.array((x_pos ,y_pos, z_pos))
							radarT = np.array((x_radar_pos + distance_r_to_t/2, y_radar_pos, 0))
							radarR = np.array((x_radar_pos - distance_r_to_t/2, y_radar_pos, 0))
							distance = np.linalg.norm(radarT-point) + np.linalg.norm(radarR-point)
							distance = distance/2
							scan_index = int(distance/delta_distance) + offset
							contribution = complex_clean_signal[int(x_radar_pos*100)][int(y_radar_pos*100)][scan_index]
							accumulator += contribution
					if first:
						first = False
						print('accumulator',accumulator)
						print('abs accumulator',abs(accumulator))
					max_point_considered = max(max_point_considered,iter_points_considered)


					image[int(x_pos*100)][int(y_pos*100)][int(z_pos*100)] = abs(accumulator)

		print('saving at',save_dest)
		np.save(save_dest, image)
		print('avg_points_considered',points_considered/iters)
		print('max_point_considered',abs(max_point_considered))
		print('iters',iters)
	else:
		image =  np.load(save_dest+'.npy')

	print('processArray backproj image shape',image.shape)

	image_min = image.min( keepdims=True)
	image_max = image.max( keepdims=True)

	norm_image = (image - image_min)/(image_max-image_min)

	if plot_images:

		x,y,z = (norm_image>0.5).nonzero()

		fig = plt.figure(1)
		fig.suptitle('backproj 3d with filter radius' + str(max_plane_offset))
		ax = fig.add_subplot(111, projection='3d')
		axes = plt.gca()
		axes.set_xlim([0,20])
		axes.set_ylim([0,19])
		axes.set_zlim([0,40])

		ax.scatter(x, y, z, zdir='z',c= 'red')


		fig = plt.figure(2)
		fig.suptitle('backproj slice')
		axes = plt.gca()
		axes.set_xlim([0,20])
		axes.set_ylim([0,19])
		plt.imshow(norm_image[:,:,14])

		fig = plt.figure(3)
		fig.suptitle('naive max')
		axes = plt.gca()
		axes.set_xlim([0,20])
		axes.set_ylim([0,19])
		scan_index = int(0.14/delta_distance) + offset
		slice = complex_clean_signal[:,:,scan_index].copy()
		print('slice shape',slice.shape)
		plt.imshow(abs(slice))


		trutharray = getTruthArray( name)
		fig = plt.figure(4)
		fig.suptitle('truth')
		axes = plt.gca()
		axes.set_xlim([0,20])
		axes.set_ylim([0,19])
		plt.imshow(trutharray.sum(axis = (2)))
		plt.show()
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
	else:
		return norm_image


def get_data(file,use_backproj = True):
	background = processBackground()
	original = loadIntoArray(file)
	cleaned_signal = cleanTarget(original,background)
	complex_clean_signal = convert_to_complex(cleaned_signal,plot = False)
	#
	#
	# sample_reshape = np.transpose(sample, (2, 0, 1))

	if use_backproj:
		backproj = back_projection(complex_clean_signal,name = file,max_plane_offset = -1,reload = True)
		return backproj
	else:
		return sample_clean_signal(cleaned_signal)

if __name__ == "__main__":
	path ="../new_res_temp/*.json"
	files = getFiles(path)
	i = 1
	print(files)
	for file in files:
		data = get_data(file)
		# name = './backprop/' + file.split('\\')[-1]
		# print('saving at',name)
		# np.save(name,data)
	#
	# for file in files:
	# 	i = i+1
	# 	print('looking at',file)
	# 	scanArray,truthArray = processArray(file)
	#
	# 	print('truthArray shape',truthArray.shape)
	# 	print('scanArray shape',scanArray.shape)
	#
	# 	figure = plt.figure(i)
	# 	figure.suptitle('truth' + str(file))
	# 	z,x,y = truthArray.nonzero()
	# 	ax = figure.add_subplot(111, projection='3d')
	# 	axes = plt.gca()
	# 	axes.set_xlim([0,20])
	# 	axes.set_ylim([0,19])
	# 	axes.set_zlim([0,40])
	# 	ax.scatter(x, y, z)
	# plt.show()
