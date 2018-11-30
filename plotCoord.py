from scipy import fft
import json
import numpy as np
import matplotlib.pyplot as plt

def findMin(amplitude):
	db = lambda x: -10 * np.log10(abs(x))
	convertDb = np.vectorize(db)
	amp_arr = np.asarray(amplitude)
	amp_arr = convertDb(amp_arr)
	return np.amin(amp_arr)

with open('./results/one_line_read_openbox.json') as f:
	data = json.load(f)

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
