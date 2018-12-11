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

TX_ANTENNA_NUM = 14
RX_ANTENNA_NUM = 15

def plotDB(time, amplitude):
    plt.gcf().clear()

    db = lambda x: 10 * np.log10(abs(x))
    tome = lambda t: t * 3 * (10**8) /2
    convertDb = np.vectorize(db)
    convertTime = np.vectorize(tome)
    time_arr = np.asarray(time)
    time_arr = convertTime(time_arr)
    amp_arr = np.asarray(amplitude)
    #amp_arr = convertDb(amp_arr)
    print("Max amplitude value ", np.max(amp_arr))


def GetSignal(targets, counter, filename, x_pos, y_pos):
    jsonRes = {}
    print("target length", len(targets))
    jsonRes['coord'] = [x_pos, y_pos]
    jsonRes['time'] = targets[1]
    jsonRes['amplitude'] = targets[0]
    with open(filename) as f:
        data = json.load(f)

    data.update({counter: jsonRes})

    with open(filename, 'w') as f:
        json.dump(data, f)
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
    freq = 450
    #original x_pos = 10, y_pos = 0
    x_pos = 10
    increment = 1
    y_pos = 0
    positionIndex = 0
    while True:
        time.sleep(1)
        x,y = getPosition(positionIndex)
        positionIndex += 1
        x,y = str(x), str(y)
        x_pos, y_pos = str(x), str(y)

        #send position to walabot
        msg = ",".join((x,y)).encode('utf-8')
        print("Sending input to arduino ", msg)
        ser.write(msg)
        ser.flush()
        print("sent")

        #wait for movemen finish
        msg = b""
        while(True):
            time.sleep(1)
            newInput = ser.read(ser.inWaiting()) # read all characters in buffer
            msg += newInput
            if not newInput == b"":
                print ("Arduino: ",newInput)

            if b"Movement Complete" in msg:
                break

            if b"Out of bounds" in msg:
                break
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
        # duration = 500
        # winsound.Beep(freq, duration)
        # stop = input("Press enter to continue")
        #
        # freq += 5;
        #
        # counter += 1



    #rasterImage, _, _, sliceDepth, power = WalabotAPI.GetRawImageSlice()

    # print targets found
    #PrintSensorTargets(targets)
    # 7) Stop and Disconnect.
    WalabotAPI.Stop()
    WalabotAPI.Disconnect()
    print ('Terminate successfully')
def initArduino(ser):
    msg = b""
    while(True):
        newInput = ser.read(ser.inWaiting()) # read all characters in buffer
        msg += newInput

        if b'init done' in msg:
            print ("Arduino: ",newInput)
            break




def getPosition(positionIndex):

    # define the lower and upper limits for x and y
    startY = 67.5
    startX = 61
    minX, maxX, minY, maxY = startX-20, startX+20, startY-10, startY+10
    # create one-dimensional arrays for x and y
    lineSpacing = 10
    x = np.linspace(minX, maxX, (maxX - minX)/lineSpacing + 1)
    y = np.linspace(minY, maxY, (maxY - minY)/lineSpacing + 1)
    # create the mesh based on these arrays
    X, Y = np.meshgrid(x, y)
    X = X.reshape((np.prod(X.shape),))
    Y = Y.reshape((np.prod(Y.shape),))
    coords = list(zip(X, Y))

    positionIndexMod = positionIndex % (len(coords))
    position = coords[positionIndexMod]

    return position



if __name__ == '__main__':
    ser = serial.Serial('COM3', 9600, timeout=1)
    initArduino(ser)
    filename = input("Key in experiment name: ")
    filename = './results/' + filename
    paths = [filename+'.json', filename+'_simple.json']
    for pth in paths:
        if not path.exists(pth):
            with open(pth, 'w+') as f:
                json.dump({}, f)

    InWallApp(filename)
