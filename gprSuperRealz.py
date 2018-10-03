from __future__ import print_function # WalabotAPI works on both Python 2 and 3.
from sys import platform
import time
from os import system, path
import json
from imp import load_source
import winsound
import matplotlib.pyplot as plt

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

def GetSignals(targets, counter, filename):
    jsonRes = {}
    jsonRes['time'] = targets[0]
    jsonRes['amplitude'] = targets[1]

    with open(filename) as f:
        data = json.load(f)

    data.update({counter: jsonRes})

    with open(filename, 'w') as f:
        json.dump(data, f)

def PrintSensorTargets(targets, counter, filename):
    system('cls' if platform == 'win32' else 'clear')
    if targets:
        for i,target in enumerate(targets):
            res = ('Target #{}:\ntype: {}\nangleDeg: {}\nx: {}\ny: {}\nz: {}' + '\nwidth: {}\namplitude: {}\n').format(i + 1, target.type, target.angleDeg, target.xPosCm, target.yPosCm, target.zPosCm, target.widthCm, target.amplitude)

            jsonRes = {}
            jsonRes['targetNum'] = i + 1
            jsonRes['type'] = target.type
            jsonRes['angleDeg'] = target.angleDeg
            jsonRes['xPosCm'] = target.xPosCm
            jsonRes['yPosCm'] = target.yPosCm
            jsonRes['zPosCm'] = target.zPosCm
            jsonRes['widthCm'] = target.widthCm
            jsonRes['amplitude'] = target.amplitude

            with open(filename) as f:
                data = json.load(f)

            data.update({counter: jsonRes})

            with open(filename, 'w') as f:
                json.dump(data, f)
    else:
        with open(filename) as f:
            data = json.load(f)

        data.update({counter: None})

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
    # Set arena by radial coordinates, with arena resolution

    # Set Spherical Coordinates
    #WalabotAPI.SetArenaR(30, 200, 5)

    # Set polar and resolution of arena
    # WalabotAPI.SetArenaTheta(-15, 15, 5);

    # Set azimuth range
    # WalabotAPI.SetArenaPhi(-60, 60, 5);


    #WalabotAPI.SetArenaX(xArenaMin, xArenaMax, xArenaRes)
    #WalabotAPI.SetArenaY(yArenaMin, yArenaMax, yArenaRes)
    #WalabotAPI.SetArenaZ(zArenaMin, zArenaMax, zArenaRes)
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

    counter = 1
    dog = input("Begin")
    while True:
        # 5) Trigger: Scan (sense) according to profile and record signals
        # to be available for processing and retrieval.
        WalabotAPI.Trigger()
        # 6) Get action: retrieve the last completed triggered recording
        print("Scanning ",counter)
        pairs = WalabotAPI.GetAntennaPairs();
        targets = [];
        targets= WalabotAPI.GetSignal(pairs[0]);
        GetSignal(targets, counter, rawSignalFile)
        print("Obtained ", counter, "raw signal")



        #plt.xlim(0, 1e-8)
        #plt.ylim(-0.5, 0.5)
        # plt.plot(targets[1], targets[0])
        # plt.show(block = False)
        # plt.pause(0.5)
        # plt.gcf().clear()


        targets = WalabotAPI.GetImagingTargets()
        PrintSensorTargets(targets,counter,imageFile)
        print("Obtained ", counter, "target")

        #alarm when done
        duration = 500
        freq = 440
        winsound.Beep(freq, duration)

        stop = input("Press key to scan")


        counter += 1


    #rasterImage, _, _, sliceDepth, power = WalabotAPI.GetRawImageSlice()

    # print targets found
    #PrintSensorTargets(targets)
    # 7) Stop and Disconnect.
    WalabotAPI.Stop()
    WalabotAPI.Disconnect()
    print ('Terminate successfully')
if __name__ == '__main__':
    filename = input("Key in experiment name: ")
    paths = [filename+'.json', filename+'_simple.json']
    for pth in paths:
        if not path.exists(pth):
            with open(pth, 'w+') as f:
                json.dump({}, f)

    InWallApp(filename)
