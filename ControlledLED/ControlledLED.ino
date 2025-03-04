int ledPin1 = 10;
int ledPin2 = 11;
int ledPin3 = 12;
int ledPin4 = 13;

void setup() {
  Serial.begin(9600);  
  pinMode(ledPin1, OUTPUT);  
  pinMode(ledPin2, OUTPUT);  
  pinMode(ledPin3, OUTPUT);
  pinMode(ledPin4, OUTPUT);
}

void loop() {
  if (Serial.available()) {
    int incomingByte = Serial.read();
    // '0' ==> empty
    if (incomingByte == '0') {
      analogWrite(ledPin1, 255);  // Turn LED off
      analogWrite(ledPin2, 255);  // Turn LED off
      analogWrite(ledPin3, 255);  // Turn LED off
      analogWrite(ledPin4, 0);  // Turn LED off
    } else if (incomingByte == '1') { //'1' ==> when present blink => reg or finding
      int brightness = random(0, 256);
      analogWrite(ledPin1, brightness);  // Turn LED off
      brightness = random(0, 256);
      analogWrite(ledPin2, brightness);  // Turn LED off
      brightness = random(0, 256);
      analogWrite(ledPin3, brightness);  // Turn LED off
      analogWrite(ledPin4, 0);  // Turn LED off
    } else if (incomingByte == '2') { //'2' ==> detected
      int brightness = random(0, 256);
      analogWrite(ledPin1, brightness);  // Turn LED off
      brightness = random(0, 256);
      analogWrite(ledPin2, brightness);  // Turn LED off
      brightness = random(0, 256);
      analogWrite(ledPin3, brightness);  // Turn LED off
      analogWrite(ledPin4, 255);  // Turn LED off
    }
  }
}
