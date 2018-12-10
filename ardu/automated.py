import serial
import time
import numpy as np



def getPosition(positionIndex):

    # define the lower and upper limits for x and y
    minX, maxX, minY, maxY = 40, 80, 30, 70
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

ser = serial.Serial('COM5', 9600, timeout=1)

msg = b""



while(True):
    newInput = ser.read(ser.inWaiting()) # read all characters in buffer
    msg += newInput

    if b'init done' in msg:
        print ("Arduino: ",newInput)
        break

positionIndex = 0

while(True):


    time.sleep(1)
    x,y = getPosition(positionIndex)
    x,y = str(x), str(y)
    positionIndex += 1
    # x = input("Enter x: ")
    # y = input("Enter y: ")
    msg = ",".join((x,y)).encode('utf-8')

    print("Sending input to arduino ", msg)
    ser.write(msg)
    ser.flush()
    print("sent")


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
