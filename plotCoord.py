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


def simpleMinPlot(data):
	keys = data.values()
	y_val = []
	x_val = []
	for key in keys:
		amp = key['amplitude']
		y_val.append(findMin(amp))
		coord = key['coord']
		x_val.append(coord[0])
	plt.xticks(x_val)
	plt.plot(x_val, y_val)
	plt.show()
	print(x_val)

def twoDimPlot(data):
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

	print(scanArray.shape)
	print(scanArray[0])
	plt.imshow(scanArray)
	plt.show()

	plt.plot(scanArray[:,int(X_LENGTH/2 - X_OFFSET)])
	plt.show()

	# truthArray = np.zeros_like(scanArray)
	# print(truthArray.shape)
	# truthArray[13,:] = np.ones(Y_LENGTH)
	#
	# plt.imshow(truthArray)
	# plt.show()

def arrayPlot(data):
	keys = data.keys()
	y_val = []
	x_val = []
	for key in keys:
		print("key",key)
		scans = data[key]



	tx_val = []
	rx_val = []
	scankeys = scans.keys()

	count = 0
	for scankey in scankeys:
		count = count + 1
		coord = make_tuple(scankey)
		tx_val.append(coord[1])
		rx_val.append(coord[0])

	print("Number of pairs: ",count)

	RX_LENGTH = max(rx_val) - min(rx_val) + 1
	TX_LENGTH = max(tx_val) - min(tx_val) + 1
	TX_OFFSET =  min(tx_val)
	RX_OFFSET =  min(rx_val)

	print( " RX_LENGTH ",RX_LENGTH," TX_LENGTH ",TX_LENGTH," TX_OFFSET ",TX_OFFSET," RX_OFFSET ",RX_OFFSET)


	scanArray = np.zeros(shape=(RX_LENGTH + 1,TX_LENGTH + 1))
	for scankey in scankeys:
		scan = scans[scankey]
		amp = scan['amplitude']
		pair = make_tuple(scankey)
		scanArray[pair[0]][pair[1]] = findMin(amp)

	print(scanArray.shape)
	print(scanArray[0])
	plt.imshow(scanArray)
	plt.colorbar()
	plt.show()




with open('./results/array_scan_B1.json') as f:
	data = json.load(f)
	arrayPlot(data)
	# keys = list(data.values())
	# key = keys[10]
	# signal = np.asarray(key['amplitude'])
	# t = np.asarray(key['time'])
	# analytic_signal = hilbert(signal)
	# real_signal = np.real(analytic_signal)
	# imag_signal = np.imag(analytic_signal)
	# print(len(real_signal), len(imag_signal))
	# amplitude_envelope = np.abs(analytic_signal)
	# instantaneous_phase = np.unwrap(np.angle(analytic_signal))
	#
	# fig = plt.figure()
	# ax0 = fig.add_subplot(211)
	# ax0.plot(t, signal, label='original')
	#
	# ax0.plot(t, imag_signal, label='imaginary')
	# ax0.set_xlabel("time in seconds")
	# ax0.legend()
	# ax1 = fig.add_subplot(212)
	# ax1.plot(t, signal, label='original')
	# ax1.plot(t, real_signal, label='real')
	#
	# ax1.set_xlabel("time in seconds")
	# ax1.legend()
	# plt.show()
