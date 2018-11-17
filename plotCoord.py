from scipy import fft
import json
import numpy as np
import matplotlib.pyplot as plt


with open('./results/cover_box_line_reflector_db.json') as f:
	data = json.load(f)

	plt.figure()

	oldcoord = [0,0]
	for item in data.keys():
		coord = data[str(item)]["coord"]
		if coord[1] == oldcoord[1]:
			continue
		oldcoord = coord
		time = data[str(item)]["time"]
		amplitude = data[str(item)]["amplitude"]

		plt.plot(amplitude,time)
		plt.xlabel('time')
		plt.ylabel('amplitude')
		plt.ylim((0, 40))
		plt.title('dB at' + str(coord))
		plt.show(block = False)
		#print(rasterImage.shape)
		plt.pause(0.5)
		plt.gcf().clear()
