from scipy import fft
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from ast import literal_eval as make_tuple
import glob
import os

MATERIALS = {
	'pvc' : {
		'size' : 2,
		'shape' : 'cylinder',
		'layer' : 'bottom',
		'value': 1
	},

	'wood' : {
		'size' : 2,
		'shape' : 'cylinder',
		'layer' : 'bottom',
		'value': 2
	},

	'metal' : {
		'size' : 1.3,
		'shape' : 'rectangular',
		'layer' : 'bottom',
		'value': 3
	},

	'aluminum' : {
		'size' : 2.5,
		'shape' : 'flat',
		'layer' : 'top',
		'value': 4
	},

	'wire' : {
		'size' : 0.7,
		'shape' : 'cylinder',
		'layer' : 'bottom',
		'value': 5
	}

}
def findMin(amplitude):
	db = lambda x: -10 * np.log10(abs(x))
	convertDb = np.vectorize(db)
	amp_arr = np.asarray(amplitude)
	amp_arr = convertDb(amp_arr)
	return np.amin(amp_arr)

def processGroundTruth(point_pairs,array,radius,value):



	for point_pair in point_pairs:
		line_point1,line_point2 = point_pair
		line_point1 = np.asarray(line_point1)
		line_point2 = np.asarray(line_point2)

		for x in range(0, array.shape[0]):
			for y in range(0,  array.shape[1]):
				point = np.asarray((x,y))

				dist = np.abs(np.cross(line_point2-line_point1, line_point1-point)) / np.linalg.norm(line_point2-line_point1)

				if dist < radius:
					array[x][y] = max(value,array[x][y])

	return array




def processJSON(file):
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

	print( " X_LENGTH ",X_LENGTH," Y_LENGTH ",Y_LENGTH," Y_OFFSET ",Y_OFFSET," X_OFFSET ",X_OFFSET)

	scanArray = np.zeros(shape=(X_LENGTH,Y_LENGTH))
	for result in results:
		coord = make_tuple(result['coord'])
		coord = (int(coord[0]-X_OFFSET),int(coord[1]-Y_OFFSET))

		body = result['body']
		amp = body[0]['amplitude']
		# print(coord, findMin(amp))
		scanArray[coord[0]][coord[1]] = findMin(amp)

	scanArray = np.delete(scanArray, (0), axis=0)
	##cross section
	# plt.plot(scanArray[:,int(X_LENGTH/2 - X_OFFSET)])
	# plt.show()

	return (scanArray,getTruthArray(file))

def getTruthArray(file):
	grid_size = 20
	array = np.zeros((grid_size,grid_size))
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

			print('plotting',item['material'],'from',start,'to',end)
			array = processGroundTruth(line,array,radius,value)
			array = np.delete(array, (0), axis=0)
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



		# line = [((0,10),(10,20))]
		# value = 1
		# radius = 1
		# array = processGroundTruth(line,array,radius,value)
	path ="../results/*.json"
	files = getFiles(path)

	for file in files:

		print('looking at',file)
		scanArray,truthArray = processJSON(file)
		plt.figure(1)
		plt.imshow(scanArray)
		plt.figure(2)
		plt.imshow(truthArray)
		plt.show()
