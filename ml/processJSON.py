from scipy import fft
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from ast import literal_eval as make_tuple
import glob
import os
from mpl_toolkits.mplot3d import Axes3D

NUM_MATERIALS = 4
DEPTH = 60
PRINT_INFO = True
MATERIALS = {
	'pvc' : {
		'size' : 2,
		'shape' : 'cylinder',
		'layer' : 'bottom',
		'value': 1
	},

	'wood' : {
		'size' : 2,
		'shape' : 'rectangular',
		'layer' : 'bottom',
		'value': 2
	},

	'metal' : {
		'size' : 1.3,
		'shape' : 'cylinder',
		'layer' : 'bottom',
		'value': 3
	},

	'aluminum' : {
		'size' : 2.5,
		'shape' : 'flat',
		'layer' : 'top',
		'value': 4
	}
	# ,
	#
	# 'wire' : {
	# 	'size' : 0.7,
	# 	'shape' : 'cylinder',
	# 	'layer' : 'bottom',
	# 	'value': 5
	# }

}

SCAN_ATTRIBUTES = {
	'num_points': 4096
}

def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n))


def findMin(amplitude,num_sample_layers):

	splits = list(split(amplitude, num_sample_layers))
	db = lambda x: 10 * np.log10(abs(x) + 1)
	splits = [db(max(split)) for split in splits]


	# convertDb = np.vectorize(db)
	# amp_arr = np.asarray(amplitude)
	# amp_arr = convertDb(amp_arr)
	# np.amin(amp_arr)

	return splits

def processGroundTruth(array,point_pairs,radius,value,threedim = False):

	object_distance = 14

	if threedim:

		for point_pair in point_pairs:
			line_point1,line_point2 = point_pair
			line_point1 = np.asarray(line_point1)
			line_point2 = np.asarray(line_point2)
			for x in range(0, array.shape[1]):
				for y in range(0,  array.shape[2]):
					point = np.asarray((x,y))

					dist = np.abs(np.cross(line_point2-line_point1, line_point1-point)) / np.linalg.norm(line_point2-line_point1)

					if dist < radius:
						array[object_distance-1][x][y] = 1
		if PRINT_INFO:
			print('processGroundTruth truthshape',array.shape)
		return array

	else:

		for point_pair in point_pairs:
			line_point1,line_point2 = point_pair
			line_point1 = np.asarray(line_point1)
			line_point2 = np.asarray(line_point2)
			for x in range(0, array.shape[0]):
				for y in range(0,  array.shape[1]):
					point = np.asarray((x,y))

					dist = np.abs(np.cross(line_point2-line_point1, line_point1-point)) / np.linalg.norm(line_point2-line_point1)

					if dist < radius:
						array[x][y][value-1] = 1

		return array



def loadIntoArray(file):
	with open(file + '.json') as f:
		data = json.load(f)

	results = data['scan']
	y_val = []
	x_val = []
	for result in results:
		coord = make_tuple(result['coord'])
		coord = (int(coord[0]),int(coord[1]))
		y_val.append(coord[1])
		x_val.append(coord[0])


	X_LENGTH = max(x_val) - min(x_val) + 1
	Y_LENGTH = max(y_val) - min(y_val) + 1
	Y_OFFSET =  min(y_val)
	X_OFFSET =  min(x_val)

	#print( " X_LENGTH ",X_LENGTH," Y_LENGTH ",Y_LENGTH," Y_OFFSET ",Y_OFFSET," X_OFFSET ",X_OFFSET)
	scanArray = np.zeros(shape=(X_LENGTH,Y_LENGTH,SCAN_ATTRIBUTES['num_points']))
	for result in results:
		coord = make_tuple(result['coord'])
		coord = (int(coord[0]-X_OFFSET),int(coord[1]-Y_OFFSET))

		body = result['body']
		amp = body[0]['amplitude']
		for index,item in enumerate(amp):
			scanArray[coord[0]][coord[1]][index] = item

	#first row has anomalous data
	scanArray = np.delete(scanArray, (0), axis=0)

	return scanArray


def processJSON(file,num_sample_layers = -1 ):


	#print( " X_LENGTH ",X_LENGTH," Y_LENGTH ",Y_LENGTH," Y_OFFSET ",Y_OFFSET," X_OFFSET ",X_OFFSET)

	with open(file + '.json') as f:
		data = json.load(f)
	results = data['scan']



	y_val = []
	x_val = []
	for result in results:
		coord = make_tuple(result['coord'])
		coord = (int(coord[0]),int(coord[1]))
		y_val.append(coord[1])
		x_val.append(coord[0])


	X_LENGTH = max(x_val) - min(x_val) + 1
	Y_LENGTH = max(y_val) - min(y_val) + 1
	Y_OFFSET =  min(y_val)
	X_OFFSET =  min(x_val)

	scanArray = np.zeros(shape=(X_LENGTH,Y_LENGTH,num_sample_layers))
	for result in results:
		coord = make_tuple(result['coord'])
		coord = (int(coord[0]-X_OFFSET),int(coord[1]-Y_OFFSET))

		body = result['body']
		amp = body[0]['amplitude']
		# print(coord, findMin(amp))
		samples = findMin(amp,num_sample_layers)
		for index,item in enumerate(samples):
			scanArray[coord[0]][coord[1]][index] = item

	scanArray = np.delete(scanArray, (0), axis=0)
	##cross section
	# plt.plot(scanArray[:,int(X_LENGTH/2 - X_OFFSET)])
	# plt.show()

	return (scanArray,getTruthArray(file))

def getTruthArray(file,threedim = False):
	grid_size = 21
	if threedim:
		if PRINT_INFO:
			print("using threedim truth")
		array = np.zeros((DEPTH,grid_size,grid_size))
	else:
		if PRINT_INFO:
			print("using numchannels truth")
		array = np.zeros((grid_size,grid_size,NUM_MATERIALS))

	with open(file + '_desc.json') as f:
		data = json.load(f)

		for item in data['truth']:
			start = make_tuple(item['start'])
			start = (start[0]-data['startX'],start[1]-data['startY'])

			end = make_tuple(item['end'])
			end = (end[0]-data['startX'],end[1]-data['startY'])

			line = [(start,end)]

			radius = MATERIALS[item['material']]['size']
			value = MATERIALS[item['material']]['value']

			#print('plotting',item['material'],'from',start,'to',end)
			array = processGroundTruth(array,line,radius,value,threedim)
			if PRINT_INFO:
				print('getTruthArray truthshape',array.shape)
		array = np.delete(array, (0), axis=1)
	return array

def getFiles(path):
	all_files = glob.glob(path)

	scans = []
	for file in all_files:
		if "desc" not in file:
			filename, file_extension = os.path.splitext(file)
			scans.append(filename)

	return scans



if __name__ == "__main__":



	print('test')
