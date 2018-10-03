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
	while True:
		wlbt.SetThreshold(80)

		appStatus, calibrationProcess = wlbt.GetStatus()
		# 5) Trigger: Scan (sense) according to profile and record signals
		# to be available for processing and retrieval.
		wlbt.Trigger()
		# 6) Get action: retrieve the last completed triggered recording
		targets = wlbt.GetImagingTargets()
		_, _, _, sliceDepth, power = wlbt.GetRawImageSlice()
		rasterImage, _, _, _, _ = wlbt.GetRawImage()
		rasterImage = np.array(rasterImage)
		# PrintSensorTargets(targets)
		PrintSensorTargets(targets)
		print(rasterImage.shape)
		element = int((sliceDepth - zArenaMin) / zArenaRes)

		if element >= 16:
			element = 15
		print(element)
		print ("Length:", len(rasterImage), "\n SliceDepth: ", sliceDepth, "\n power: ", power)
		slice = rasterImage[:, :, element]
		slice1 = rasterImage[:, :, element + 2]
		slice2 = rasterImage[:, :, element + 1]
		plt.plot(slice1[0], slice1[1], 0)
		plt.show()
		stop = input("stop")
		#plt.xlim(-10, 34)
		#plt.ylim(-10, 44)
		#plt.imshow(rasterImage)
		#plt.show(block = False)
		#print(rasterImage.shape)
		#plt.pause(0.001)
		#plt.gcf().clear()



	# 7) Stop and Disconnect.
	wlbt.Stop()
	wlbt.Disconnect()
	wlbt.Clean()
	print('Terminate successfully')

if __name__ == '__main__':
	InWallApp()
