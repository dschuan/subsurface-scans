from __future__ import print_function # WalabotAPI works on both Python 2 and 3.
from sys import platform
import time
from os import system, path
import json
from imp import load_source
import winsound
import matplotlib.pyplot as plt
import numpy as np
import serial

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

    start = input("scan now?")

    WalabotAPI.Trigger()
    pairs = WalabotAPI.GetAntennaPairs();

    xpos = 0
    ypos = 0
    result = {}
    coord = (xpos,ypos)
    result[str(coord)] = {}

    for pair in pairs:
        coordResult = {}
        targets= WalabotAPI.GetSignal(pair);



        pairResult = {}
        pairResult['time'] = targets[1]
        pairResult['amplitude'] = targets[0]

        coordResult[str((pair.txAntenna,pair.rxAntenna))] = pairResult
        result[str(coord)].update(coordResult)





    #
    # with open(filename) as f:
    #     data = json.load(f)
    #
    # data.update({counter: jsonRes})

    with open(filename, 'w') as f:
        json.dump(result, f)

    WalabotAPI.Stop()
    WalabotAPI.Disconnect()
    print ('Terminate successfully')



if __name__ == '__main__':
    filename = input("Key in experiment name: ")
    filename = './results/' + filename
    with open(filename, 'w+') as f:
        json.dump({}, f)

    InWallApp(filename)
