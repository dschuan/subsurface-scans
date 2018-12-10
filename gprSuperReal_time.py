from __future__ import print_function # WalabotAPI works on both Python 2 and 3.
from sys import platform
import time as tm
from os import system, path
import json
from imp import load_source
import winsound
import matplotlib.pyplot as plt
import numpy as np


APP_STATUS = ['STATUS_CLEAN',
'STATUS_INITIALIZED',
'STATUS_CONNECTED',
'STATUS_CONFIGURED',
'STATUS_SCANNING',
'STATUS_CALIBRATING',
'STATUS_CALIBRATING_NO_MOVEMENT']

WalabotAPI = load_source('WalabotAPI',
    'C:/Program Files/Walabot/WalabotSDK/python/WalabotAPI.py')
WalabotAPI.Init("C:/Program Files/Walabot/WalabotSDK/bin/WalabotAPI.dll")

TX_ANTENNA_NUM = 14
RX_ANTENNA_NUM = 15
def plotDB(time, amplitude):
    db = lambda x: -10 * np.log10(abs(x))
    convertDb = np.vectorize(db)
    time_arr = np.asarray(time)
    amp_arr = np.asarray(amplitude)
    amp_arr = convertDb(amp_arr)
    plt.ylim(None, 30)
    print("Min amplitude value ", np.min(amp_arr))
    plt.plot(time_arr.tolist(), amp_arr.tolist())

    plt.show(block = False)
    plt.pause(0.2)
    plt.gcf().clear()

def GetSignal(targets, counter, filename, x_pos, y_pos):
    # jsonRes = {}
    # print("target length", len(targets))
    # jsonRes['coord'] = [x_pos, y_pos]
    # jsonRes['time'] = targets[1]
    # jsonRes['amplitude'] = targets[0]
    # with open(filename) as f:
    #     data = json.load(f)
    #
    # data.update({counter: jsonRes})
    #
    # with open(filename, 'w') as f:
    #     json.dump(data, f)
    plotDB(targets[1], targets[0])

def PrintSensorTargets(targets, counter, filename, rasterImage, power, x_pos, y_pos):
    system('cls' if platform == 'win32' else 'clear')
    jsonRes = {}
    jsonRes['rasterImage'] = rasterImage
    jsonRes['power'] = power
    if targets:
        for i,target in enumerate(targets):
            res = ('Target #{}:\ntype: {}\nangleDeg: {}\nx: {}\ny: {}\nz: {}' + '\nwidth: {}\namplitude: {}\n').format(i + 1, target.type, target.angleDeg, target.xPosCm, target.yPosCm, target.zPosCm, target.widthCm, target.amplitude)
            jsonRes['Coordinates'] = [x_pos, y_pos]

            jsonRes['targetNum'] = i + 1
            jsonRes['type'] = target.type
            jsonRes['angleDeg'] = target.angleDeg
            jsonRes['xPosCm'] = target.xPosCm
            jsonRes['yPosCm'] = target.yPosCm
            jsonRes['zPosCm'] = target.zPosCm
            jsonRes['widthCm'] = target.widthCm
            jsonRes['amplitude'] = target.amplitude
            jsonRes['rasterImage'] = rasterImage
            jsonRes['power'] = power

            with open(filename) as f:
                data = json.load(f)

            data.update({counter: jsonRes})

            with open(filename, 'w') as f:
                json.dump(data, f)
    else:
        with open(filename) as f:
            data = json.load(f)

        data.update({counter: jsonRes})

        with open(filename, 'w') as f:
            json.dump(data, f)

def InWallApp(filename):
    rawSignalFile = filename+'.json'
    imageFile = filename+'_simple.json'
    # WalabotAPI.SetArenaX - input parameters
    xArenaMin, xArenaMax, xArenaRes = -3, 4, 0.5
    # WalabotAPI.SetArenaY - input parameters
    yArenaMin, yArenaMax, yArenaRes = -6, 4, 0.5
    # WalabotAPI.SetArenaZ - input parameters
    zArenaMin, zArenaMax, zArenaRes = 3, 15, 0.5
    # Configure Walabot database install location (for windows)
    WalabotAPI.SetSettingsFolder()
    # 1) Connect: Establish communication with walabot.
    WalabotAPI.ConnectAny()
    # 2) Configure: Set scan profile and arena
    # Set Profile - to Short-range.
    WalabotAPI.SetProfile(WalabotAPI.PROF_SHORT_RANGE_IMAGING)

    # Walabot filtering disable
    WalabotAPI.SetDynamicImageFilter(WalabotAPI.FILTER_TYPE_NONE)
    # 3) Start: Start the system in preparation for scanning.
    WalabotAPI.Start()
    # calibrates scanning to ignore or reduce the signals
    WalabotAPI.StartCalibration()
    appStatus = -1
    while appStatus != 4:
        appStatus, calibrationProcess = WalabotAPI.GetStatus()

        print("Starting up ", APP_STATUS[appStatus], "percentage", calibrationProcess )
        for _ in range(10):
            WalabotAPI.Trigger()
    pairs = WalabotAPI.GetAntennaPairs();
    counter = 1
    stopper = input("Begin")
    freq = 250

    x_pos = 10
    increment = 2
    y_pos = 0
    while True:
        if x_pos > 30:
            y_pos += increment
            x_pos = 10
        # 5) Trigger: Scan (sense) according to profile and record signals
        # to be available for processing and retrieval.

        WalabotAPI.Trigger()
        # 6) Get action: retrieve the last completed triggered recording
        print("Scanning ",counter, " using ", TX_ANTENNA_NUM, " transmitter and ", RX_ANTENNA_NUM, " receiver")
        pairs = WalabotAPI.GetAntennaPairs();

        for pair in pairs:
            if pair.txAntenna == TX_ANTENNA_NUM and pair.rxAntenna == RX_ANTENNA_NUM:
                chosenPair = pair
        targets = [];
        print("Coordinates: ", x_pos, " ,", y_pos)


        targets= WalabotAPI.GetSignal(chosenPair);
        GetSignal(targets, counter, rawSignalFile, x_pos, y_pos)
        print("Obtained ", counter, "raw signal")


        #targets = WalabotAPI.GetImagingTargets()
        #rasterImage, _, _, _, power = WalabotAPI.GetRawImage()

        #PrintSensorTargets(targets,counter,imageFile,rasterImage,power, x_pos, y_pos)
        #print("Obtained ", counter, "target image")

        #alarm when done

        freq += 1;

        counter += 1
        x_pos += increment


    #rasterImage, _, _, sliceDepth, power = WalabotAPI.GetRawImageSlice()

    # print targets found
    #PrintSensorTargets(targets)
    # 7) Stop and Disconnect.
    WalabotAPI.Stop()
    WalabotAPI.Disconnect()
    print ('Terminate successfully')
if __name__ == '__main__':
    filename = input("Key in experiment name: ")
    filename = './results/' + filename
    paths = [filename+'.json', filename+'_simple.json']
    for pth in paths:
        if not path.exists(pth):
            with open(pth, 'w+') as f:
                json.dump({}, f)

    InWallApp(filename)
