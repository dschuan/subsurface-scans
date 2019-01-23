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
from xytable.pydrive import xytable
from collections import defaultdict

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


startY =10
endy =30
startX = 5
endx = 25
lineSpacing = 1
def getPosition(startY,startX,endy,endx,lineSpacing):

    # define the lower and upper limits for x and y

    xmove = endx - startX
    ymove= endy - startY
    minX, maxX, minY, maxY = startX, startX+xmove, startY, startY+ymove
    # create one-dimensional arrays for x and y

    x = np.linspace(minX, maxX, (maxX - minX)/lineSpacing + 1)
    y = np.linspace(minY, maxY, (maxY - minY)/lineSpacing + 1)
    # create the mesh based on these arrays
    X, Y = np.meshgrid(x, y)
    X = X.reshape((np.prod(X.shape),))
    Y = Y.reshape((np.prod(Y.shape),))
    coords = list(zip(X, Y))

    # positionIndexMod = positionIndex % (len(coords))
    # position = coords[positionIndexMod]
    #
    # return position

    return coords

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

    XYtable = xytable('COM7')
    XYtable.open()

    pairs = WalabotAPI.GetAntennaPairs();

    # start = input("scan now?")

    WalabotAPI.Trigger()
    pairs = WalabotAPI.GetAntennaPairs();

    coords = getPosition(startY,startX,endy,endx,lineSpacing)

    result = defaultdict(lambda:defaultdict(int))
    description = defaultdict(lambda:defaultdict(int))

    description['startY'] = startY
    description['startX'] = startX
    description['endy'] = endy
    description['endx'] = endx
    description['lineSpacing'] = lineSpacing
    description['description'] = input('describe experiment')

    TX_ANTENNA_NUM = 14
    RX_ANTENNA_NUM = 15
    for xpos, ypos in coords:
        print('setting xytable to ',xpos,' ',ypos)
        XYtable.set_position(xpos,ypos)
        time.sleep(0.1)
        print('getting signal from walabot')
        WalabotAPI.Trigger()
        print('received signal from walabot')
        for pair in pairs:
            if pair.txAntenna == TX_ANTENNA_NUM and pair.rxAntenna == RX_ANTENNA_NUM:
                coordResult = {}

                targets= WalabotAPI.GetSignal(pair);



                pairResult = {}
                pairResult['time'] = targets[1]
                pairResult['amplitude'] = targets[0]

                coordResult[str((pair.txAntenna,pair.rxAntenna))] = pairResult
                result[str((xpos,ypos))].update(coordResult)

    print(result.keys())
    with open(filename, 'w') as f:
        json.dump(result, f)

    with open(filename+'_desc', 'w') as f:
        json.dump(description, f)

    WalabotAPI.Stop()
    WalabotAPI.Disconnect()
    XYtable.close()
    print ('Terminate successfully')



if __name__ == '__main__':
    filename = input("Key in experiment name: ")
    filename = './results/' + filename
    with open(filename, 'w+') as f:
        json.dump({}, f)

    InWallApp(filename)
