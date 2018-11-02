from __future__ import print_function # WalabotAPI works on both Python 2 an 3.
from sys import platform
from os import system
from imp import load_source
import time
from os.path import join
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np

from scipy import ndimage as ndi
from skimage import feature
from skimage.filters import roberts, sobel, scharr, prewitt

from mpl_toolkits.mplot3d import Axes3D


APP_STATUS = ['STATUS_CLEAN',
'STATUS_INITIALIZED',
'STATUS_CONNECTED',
'STATUS_CONFIGURED',
'STATUS_SCANNING',
'STATUS_CALIBRATING',
'STATUS_CALIBRATING_NO_MOVEMENT']

if platform == 'win32':
	modulePath = join('C:/', 'Program Files', 'Walabot', 'WalabotSDK',
		'python', 'WalabotAPI.py')
elif platform.startswith('linux'):
	modulePath = join('/usr', 'share', 'walabot', 'python', 'WalabotAPI.py')

wlbt = load_source('WalabotAPI', modulePath)
wlbt.Init()

def PrintSensorTargets(targets):
	system('cls' if platform == 'win32' else 'clear')
	if targets:
		for i, target in enumerate(targets):

			print(('Target #{}:\ntype: {}\nangleDeg: {}\nx: {}\ny: {}\nz: {}'+
				'\nwidth: {}\namplitude: {}\n').format(i + 1, target.type,
				target.angleDeg, target.xPosCm, target.yPosCm, target.zPosCm,
				target.widthCm, target.amplitude))
	else:
		print('No Target Detected')

def InWallApp():
	# wlbt.SetArenaX - input parameters
	xArenaMin, xArenaMax, xArenaRes = -5, 5, 0.3
	# wlbt.SetArenaY - input parameters
	yArenaMin, yArenaMax, yArenaRes = -10, 10, 0.3
	# wlbt.SetArenaZ - input parameters
	zArenaMin, zArenaMax, zArenaRes = 3, 8, 0.3
	# Initializes walabot lib
	wlbt.Initialize()
	# 1) Connects: Establish communication with walabot.
	wlbt.ConnectAny()
	# 2) Configure: Set scan profile and arena
	# Set Profile - to Short-range.
	wlbt.SetProfile(wlbt.PROF_SHORT_RANGE_IMAGING)
	# Set arena by Cartesian coordinates, with arena resolution
	wlbt.SetArenaX(xArenaMin, xArenaMax, xArenaRes)
	wlbt.SetArenaY(yArenaMin, yArenaMax, yArenaRes)
	wlbt.SetArenaZ(zArenaMin, zArenaMax, zArenaRes)
	# Walabot filtering disable
	wlbt.SetDynamicImageFilter(wlbt.FILTER_TYPE_MTI)

	# 3) Start: Start the system in preparation for scanning.
	wlbt.Start()
	# calibrates scanning to ignore or reduce the signals
	wlbt.StartCalibration()
	appStatus = -1
	while appStatus != 4:
		appStatus, calibrationProcess = wlbt.GetStatus()
		print("Starting up ", APP_STATUS[appStatus], "percentage", calibrationProcess )
		for _ in range(10):
			wlbt.Trigger()

	while wlbt.GetStatus()[0] == wlbt.STATUS_CALIBRATING:
		wlbt.Trigger()

	fig = plt.figure()
	while True:

		#appStatus, calibrationProcess = wlbt.GetStatus()
		# 5) Trigger: Scan (sense) according to profile and record signals
		# to be available for processing and retrieval.
		wlbt.Trigger()
		# 6) Get action: retrieve the last completed triggered recording
		#targets = wlbt.GetImagingTargets()
		#_, _, _, sliceDepth, power = wlbt.GetRawImageSlice()
		rasterImage, _, _, _, power = wlbt.GetRawImage()
		rasterImage = np.array(rasterImage)

		xArray = [0]
		yArray = [0]
		zArray = [0]

		print(rasterImage)
		print(power)
		for (x,y,z), value in np.ndenumerate(rasterImage):
			if value > 100 and power > 1:
				xArray.append(x)
				yArray.append(y)
				zArray.append(z)

		ax = fig.add_subplot(111, projection='3d')

		ax.scatter(xArray,zArray,yArray)

		# PrintSensorTargets(targets)
		#PrintSensorTargets(targets)



		ax.set_xlim(0, 67)
		ax.set_ylim(0, 17)
		ax.set_zlim(0, 37)
		#plt.imshow(rasterImage)
		plt.show(block = False)
		#print(rasterImage.shape)
		plt.pause(0.2)
		plt.gcf().clear()



	# 7) Stop and Disconnect.
	wlbt.Stop()
	wlbt.Disconnect()
	wlbt.Clean()
	print('Terminate successfully')

if __name__ == '__main__':
	InWallApp()
