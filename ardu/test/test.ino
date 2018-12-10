

#include <Stepper.h>

//Stepper motor pins
const int stepPin1 = 2; 
const int dirPin1 = 3; 
const int stepPin2 = 4; 
const int dirPin2 = 5;

//Robot scanning area
const float W = 120;
const float H = 120 ;

//item dimensions
const float w = 0;
const float h = 0;


float curr_l1;
float curr_l2;

//coordinate scaling
const int grid = 1;

//spool diameter
const float d = 2;
const int original_l = 170;
const int stepno = 200;
const float steplength = (d*PI)/stepno;

float curr_x = 0;
float curr_y = 0;

//Initializing constants
int cali_x = 60;
int cali_y = 60; 


void setup() 
{
  // put your setup code here, to run once:
  pinMode(stepPin1,OUTPUT); 
  pinMode(dirPin1,OUTPUT);
  pinMode(stepPin2,OUTPUT); 
  pinMode(dirPin2,OUTPUT);
  Serial.begin(9600);
  init_cali();
}
void loop() //This will be executed over and over
{ 
    if (Serial.available() > 0) {

        if(Serial.read() == '1') {   

        } else if(Serial.read() == '0') {

        }
    }
}
