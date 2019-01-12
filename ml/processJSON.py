from scipy import fft
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert



def findMin(amplitude):
	db = lambda x: -10 * np.log10(abs(x))
	convertDb = np.vectorize(db)
	amp_arr = np.asarray(amplitude)
	amp_arr = convertDb(amp_arr)
	return np.amin(amp_arr)

def processJSON(data):
	keys = data.values()
	y_val = []
	x_val = []
	for key in keys:
		coord = key['coord']
		y_val.append(coord[1])
		x_val.append(coord[0])


	X_LENGTH = max(x_val) - min(x_val) + 1
	Y_LENGTH = max(y_val) - min(y_val) + 1
	Y_OFFSET =  min(y_val)
	X_OFFSET =  min(x_val)

	print( " X_LENGTH ",X_LENGTH," Y_LENGTH ",Y_LENGTH," Y_OFFSET ",Y_OFFSET," X_OFFSET ",X_OFFSET)

	keys = data.values()
	scanArray = np.zeros(shape=(X_LENGTH,Y_LENGTH))
	for key in keys:
		amp = key['amplitude']
		coord = key['coord']
		scanArray[coord[0]-X_OFFSET][coord[1]-Y_OFFSET] = findMin(amp)


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
