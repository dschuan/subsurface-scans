import serial
import time

ser = serial.Serial('COM5', 9600, timeout=1)

msg = b""
while(True):
    newInput = ser.read(ser.inWaiting()) # read all characters in buffer
    msg += newInput

    if b'init done' in msg:
        print ("Arduino: ",newInput)
        break

isDown = True
while(True):


    time.sleep(1)
    if(isDown):
        x = "50"
        y = "60"
        isDown = False
    else:
        x = "50"
        y = "50"
        isDown = True
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
