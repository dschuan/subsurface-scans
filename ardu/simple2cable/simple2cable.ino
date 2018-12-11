#include <Stepper.h>

//Stepper motor pins
const int stepPin1 = 2; 
const int dirPin1 = 3; 
const int stepPin2 = 4; 
const int dirPin2 = 5;

const float W = 120;

const float w = 7;

float curr_x = 61;
float curr_y = 67.5;


float curr_l1 = sqrt(sq(curr_x - w/2)+sq(curr_y)) ;
float curr_l2 = sqrt(sq(W - (curr_x + w/2))+sq(curr_y));
//float curr_l2 = 1.058342 * raw_l2 - 8.409091;
const float steplength = 0.03675;




void setup() 
{
  // put your setup code here, to run once:
  pinMode(stepPin1,OUTPUT); 
  pinMode(dirPin1,OUTPUT);
  pinMode(stepPin2,OUTPUT); 
  pinMode(dirPin2,OUTPUT);
  Serial.begin(9600);
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
      float new_x = first.toFloat();
      float new_y = second.toFloat(); 
      moveTo(new_x, new_y);
      

    }
}

void moveTo(float x, float y)
{
  //Moving to coordinate (x,y)
  Serial.print("Moving to (");
  Serial.print(x);
  Serial.print(",");
  Serial.print(y);
  Serial.print(")\n"); 

  //Calculate new length for each spool needed for the new coordinate
  float l1 = sqrt(sq(x - (w/2)) + sq(y));
  float l2 = sqrt(sq(W - (x + (w/2))) + sq(y));
  //l2 = 1.058342 * l2 - 8.409091;
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

  curr_x = x;
  curr_y = y;
  
}
