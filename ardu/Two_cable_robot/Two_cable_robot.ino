
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


float curr_l1 = 75.8;
float curr_l2 = 78.5;

//coordinate scaling
const int grid = 1;

//spool diameter
const float d = 2;
const int original_l = 170;
const int stepno = 200;
const float steplength = 0.035;

float curr_x = 50;
float curr_y = 50;

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
  //init_cali();
  Serial.print("init done");
}

void loop() 
{
    if (Serial.available()>0)
    {
      Serial.print("Current position:(");
      Serial.print(curr_x);
      Serial.print(",");
      Serial.print(curr_y);
      Serial.print(")\n");

      String first = Serial.readStringUntil(',');
      Serial.read();
      String second = Serial.readStringUntil('\0');
      int new_x = first.toInt();
      int new_y = second.toInt(); 
      moveTo(new_x, new_y);
      

    }
}

//Move current length to coordinate (50,50) in scanning area. 
void init_cali()
{
  curr_l1 = original_l;
  curr_l2 = original_l;
  
  Serial.print("Calibrating length");
  float cali_l1 = sqrt(sq(cali_x) + sq(cali_y));
  float cali_l2 = sqrt(sq(cali_x) + sq(cali_y));
  float init_delt_l1 = curr_l1 - cali_l1;
  float init_delt_l2 = curr_l2 - cali_l2;
  int init_step_l1 = (abs(init_delt_l1)/ steplength);
  int init_step_l2 = (abs(init_delt_l2)/ steplength);

  digitalWrite(dirPin1,LOW);
  digitalWrite(dirPin2,HIGH);

  //Move both motors
  //int counter1 = 200;
  //int counter2 = 200;
  int counter1 = init_step_l1;
  int counter2 = init_step_l2;

  while (counter1 != 0 || counter2 != 0)
  {
    if(counter1 != 0)
    {
      digitalWrite(stepPin1,HIGH);
      delayMicroseconds(20);
      digitalWrite(stepPin1,LOW);
      delayMicroseconds(20);
      Serial.print(counter1);
      Serial.print(")\n");
      counter1 = counter1 - 1;
      
    }
       
    if (counter2 != 0)
    {
      digitalWrite(stepPin2,HIGH);
      delayMicroseconds(20);
      digitalWrite(stepPin2,LOW);
      delayMicroseconds(20);
      counter2 = counter2 - 1;
    }
    delay(20); 
  }
  curr_l1 = cali_l1;
  curr_l2 = cali_l2;
  curr_x = 50;
  curr_y = 50;
  Serial.print("Calibration complete. Current coordinates: (50,50)\n");
  Serial.print("Length of string:");
  Serial.print(curr_l1);
  Serial.print("\n");
}


void moveTo(int x, int y)
{
  //Moving to coordinate (x,y)
  Serial.print("Moving to (");
  Serial.print(x);
  Serial.print(",");
  Serial.print(y);
  Serial.print(")\n"); 

  //Make sure that the coordinates is not out of bounds
  if (x > W || y >H)
  {
    Serial.print("Out of bounds! Try again.");
    return;
  }
  x = x + 10;
  y = y + 10;

  Serial.print("Actually at: (");
  Serial.print(x);
  Serial.print(",");
  Serial.print(y);
  Serial.print(")\n"); 

  //Calculate new length for each spool needed for the new coordinate
  float l1 = sqrt(sq(x - (w/2)) + sq(y-(h/2)));
  float l2 = sqrt(sq(W - (x + (w/2))) + sq(y - (h/2)));

  Serial.print("l1:");
  Serial.print(l1);
  Serial.print("\n\n");
  Serial.print("l2 :");
  Serial.print(l2);
  Serial.print("\n\n");

  //How much the length of each string has to change  from current length
  float delta_l1 = curr_l1 - l1;
  float delta_l2 = curr_l2 - l2;

  Serial.print("l1 change needed:");
  Serial.print(delta_l1);
  Serial.print("\n");
  Serial.print("l2 change needed:");
  Serial.print(delta_l2);
  Serial.print("\n");
  
  //Set direction of both Motor 1 and 2
  if (delta_l1 < 0)
  {
    digitalWrite(dirPin1,HIGH);
  }
  else
  {
     digitalWrite(dirPin1,LOW);
  }

  if (delta_l2 < 0)
  {
    digitalWrite(dirPin2,LOW);
  }
  else
  {
    digitalWrite(dirPin2,HIGH);
  }
 
  //Step count of each motor required
  int step_l1 = (abs(delta_l1)/ steplength);
  int step_l2 = (abs(delta_l2)/ steplength);
  
  //Move both motors
  int counter1 = step_l1;
  int counter2 = step_l2;

  //l1 offset
  //y = 1.030925x - 2.19174


  while (counter1 != 0 || counter2 != 0)
  {
    if(counter1 != 0)
    {
      digitalWrite(stepPin1,HIGH);
      delayMicroseconds(20);
      digitalWrite(stepPin1,LOW);
      counter1 = counter1 - 1;
    }
       
    if (counter2 != 0)
    {
      digitalWrite(stepPin2,HIGH);
      delayMicroseconds(20);
      digitalWrite(stepPin2,LOW);
      counter2 = counter2 - 1;
    }
    delay(20); 
  }

  Serial.print("Motor 1 Steps:");
  Serial.print(step_l1);
  Serial.print("\n");
  Serial.print("Motor 2 Steps:");
  Serial.print(step_l2);
  Serial.print("\n");
  Serial.print("Movement Complete\n\n");

    

  //update current values
  curr_l1 = l1;
  curr_l2 = l2; 
  curr_x = x - 10;
  curr_y = y - 10;
  
}
