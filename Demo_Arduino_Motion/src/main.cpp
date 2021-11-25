// Please use the tensorflow, which has the same version number as the tflite micro library (2.4.0) used in the project, to train and generate NN 
#include <Arduino.h>
#include <Arduino_LSM9DS1.h>
#include "TFLITE_Wrapper.h"
#include <ArduinoBLE.h>
#include "BLE_transfer.h"
#include "myUtils.h"
// #include "setting.h"

TFLITE_Wrapper::TinyML::TFLite<
    NUMBER_OF_INPUTS,
    NUMBER_OF_OUTPUTS,
    TENSOR_ARENA_SIZE> nn;

// uint8_t *model;

char event[60];

void setup() {
  pinMode(LED_BUILTIN, OUTPUT);
  // put your setup code here, to run once:
  Serial.begin(9600);
  // Wait for serial monitor to connect
  while (!Serial);

  // Initialize IMU sensors
  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU!");
    while (1);
  }

  // Print out the samples rates of the IMUs
  Serial.print("Accelerometer sample rate: ");
  Serial.print(IMU.accelerationSampleRate());
  Serial.println(" Hz");
  Serial.print("Gyroscope sample rate: ");
  Serial.print(IMU.gyroscopeSampleRate());
  Serial.println(" Hz");

  Serial.println();
  setupBLEFileTransfer();
}

void loop() {
  // BLE transmission in the loop
    updateBLEFileTransfer();
}

void onBLEFileReceived(uint8_t* file_data, int file_length) {
    // Examine the model
    Serial.println("Model Received");
    
    Serial.print("Model size is: ");
    Serial.println(file_length);
    Serial.println();
    print_memory_info(event, 60);
    Serial.println();

    // Load the model
    if (!nn.begin(file_data)) {
        Serial.println("Cannot inialize model");
        Serial.println(nn.errorMessage());
        delay(60000);
    }
    else {
        Serial.println("Model loaded, starting inference");
    }

    float input[NUMBER_OF_INPUTS] = {0};
    float output[NUMBER_OF_OUTPUTS] = {0};
    // Run inference
    while(true){
      
      if(preprocessing(input)){
        nn.predict(input, output);
        postprocessing(output);
      }    
    }
}