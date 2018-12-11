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

with open('./results/one_line_read_openbox.json') as f:
	data = json.load(f)
	keys = list(data.values())
	key = keys[10]
	signal = np.asarray(key['amplitude'])
	t = np.asarray(key['time'])
	analytic_signal = hilbert(signal)
	real_signal = np.real(analytic_signal)
	imag_signal = np.imag(analytic_signal)
	print(len(real_signal), len(imag_signal))
	amplitude_envelope = np.abs(analytic_signal)
	instantaneous_phase = np.unwrap(np.angle(analytic_signal))

	fig = plt.figure()
	ax0 = fig.add_subplot(211)
	ax0.plot(t, signal, label='original')

	ax0.plot(t, imag_signal, label='imaginary')
	ax0.set_xlabel("time in seconds")
	ax0.legend()
	ax1 = fig.add_subplot(212)
	ax1.plot(t, signal, label='original')
	ax1.plot(t, real_signal, label='real')

	ax1.set_xlabel("time in seconds")
	ax1.legend()
	plt.show()
