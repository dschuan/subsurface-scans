import serial
import time
import math
y_axis = '1SP' #y-axis
x_axis = '2SP' #x-axis

y_axis_position = '1la-' #Position header for Y-axis
x_axis_position = '2la' #Position header for X-axis

speed_y = '1200'
speed_x = '500'

y_initiate_move = '1M'
x_initiate_move = '2M'
y_increment = '6000'
x_increment = '3000'

class xytable():
    def __init__(self, port = 'COM7'):
      self.x_position = '0'
      self.y_position = '0'
      self.ser = serial.Serial(port, timeout=1)




    def open(self):
        print('serial at ' + self.ser.name)
        self.ser.set_buffer_size(rx_size=8388608)
        #initialise
        self.write_bytes('JMP2')
        input('am i back?')


        #set speed
        self.write_bytes(y_axis + speed_y)
        time.sleep(1)
        self.write_bytes(x_axis + speed_x)
        time.sleep(1)




    def write_bytes(self,string):
        # print('writing ' + string+' with ' + self.ser.name)
        message = string + "\n"
        if not self.ser.is_open:
             print('WARNING:serial not open')
        self.ser.write(message.encode('utf8'))


    def add_string(self,a,b):
        value = int(a) + int(b)
        return str(value)

    def minus_string(self,a,b):
        value = int(a) - int(b)
        return str(value)

    def y_move(self):
        global y_axis_position
        self.write_bytes(y_axis_position + self.y_position)
        time.sleep(0.1)
        self.write_bytes(y_initiate_move)
        time.sleep(0.1)

    def x_move(self):
        global x_axis_position
        self.write_bytes(x_axis_position + self.x_position)
        time.sleep(0.1)
        self.write_bytes(x_initiate_move)
        time.sleep(0.1)

    def set_position(self,x,y):
        global x_increment
        global y_increment

        distance = math.hypot(float(self.x_position)/int(x_increment)-x, float(self.y_position)/int(y_increment)-y)
        self.x_position = str(x* int(x_increment))
        self.y_position = str(y* int(y_increment))
        self.y_move()
        self.x_move()
        print('distance',distance)
        time.sleep(1 + distance/10)

    def close(self):
        #initialise
        self.write_bytes('JMP2')
        time.sleep(2)
        self.ser.close()
        print('DONE')

if __name__ == "__main__":

    xytable = xytable('COM7')
    xytable.open()


    xytable.set_position(5,5)


    # xytable.close()
