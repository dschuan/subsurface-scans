from scipy import fft
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert
from ast import literal_eval as make_tuple


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




def processJSON(data):
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
		coord = (int(coord[0]),int(coord[1]))

		body = result['body']
		amp = body[0]['amplitude']

		scanArray[coord[0]-X_OFFSET][coord[1]-Y_OFFSET] = findMin(amp)

	#
	# plt.imshow(scanArray)
	# plt.show()
	#
	# plt.plot(scanArray[:,int(X_LENGTH/2 - X_OFFSET)])
	# plt.show()

	truthArray = np.zeros_like(scanArray)
	truthArray[13,:] = np.ones(Y_LENGTH)

	# plt.imshow(truthArray)
	# plt.show()
	return (scanArray,truthArray)

if __name__ == "__main__":
	# grid_size = 20
	# array = np.zeros((grid_size,grid_size))
	#
	# line = [((0,10),(10,20))]
	# value = 1
	# radius = 1
	# array = processGroundTruth(line,array,radius,value)
	#
	# line = [((0,3),(20,8))]
	# value = 2
	# radius = 3
	# array = processGroundTruth(line,array,radius,value)
	#
	# line = [((15,0),(1,20))]
	# value = 3
	# radius = 0.5
	# array = processGroundTruth(line,array,radius,value)
	#
	# plt.imshow(array)
	# plt.show()

	with open('../results/metal_ruler.json') as f:
		data = json.load(f)
		scanArray,truthArray = processJSON(data)

		plt.imshow(scanArray)
		plt.show()
