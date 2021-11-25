#include <Arduino.h>
#include "mbed.h"
#include "mbed_mem_trace.h"
#include <Arduino_LSM9DS1.h>

#define NUMBER_OF_INPUTS 300
#define NUMBER_OF_OUTPUTS 2
#define TENSOR_ARENA_SIZE 8*1024

// Application dependent variables
// Values from Tiny Motion Trainer
#define MOTION_THRESHOLD 0.2
#define CAPTURE_DELAY 200 // This is now in milliseconds
#define NUM_SAMPLES 50
bool isCapturing = false;
int numSamplesRead = 0;
// Array to map gesture index to a name
const char *GESTURES[] = {
    "Gesture 0", "Gesture 1"
};
// Variables to hold IMU data
float aX, aY, aZ, gX, gY, gZ;

void print_memory_info(char* printEvent, int iSize) {
    // allocate enough room for every thread's stack statistics
    int cnt = osThreadGetCount();
    mbed_stats_stack_t *stats = (mbed_stats_stack_t*) malloc(cnt * sizeof(mbed_stats_stack_t));
 
    cnt = mbed_stats_stack_get_each(stats, cnt);
    for (int i = 0; i < cnt; i++) {
        snprintf_P(printEvent, iSize, "Thread: 0x%lX, Stack size: %lu / %lu\r\n", stats[i].thread_id, stats[i].max_size, stats[i].reserved_size);
        Serial.println(printEvent);
    }
    free(stats);
 
    // Grab the heap statistics
    mbed_stats_heap_t heap_stats;
    mbed_stats_heap_get(&heap_stats);
    snprintf_P(printEvent, iSize, "Heap size: %lu / %lu bytes\r\n", heap_stats.current_size, heap_stats.reserved_size);
    Serial.println(printEvent);
}

bool preprocessing(float* input){
    while (!isCapturing) {
        if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) {
        
          IMU.readAcceleration(aX, aY, aZ);
          IMU.readGyroscope(gX, gY, gZ);

          // Sum absolute values
          float average = fabs(aX / 4.0) + fabs(aY / 4.0) + fabs(aZ / 4.0) + fabs(gX / 2000.0) + fabs(gY / 2000.0) + fabs(gZ / 2000.0);
          average /= 6.;

          // Above the threshold?
          if (average >= MOTION_THRESHOLD) {
            isCapturing = true;
            numSamplesRead = 0;
            break;
          }
        }
    }  

    while (isCapturing) {

        // Check if both acceleration and gyroscope data is available
        if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) {

            // read the acceleration and gyroscope data
            IMU.readAcceleration(aX, aY, aZ);
            IMU.readGyroscope(gX, gY, gZ);

            // Normalize the IMU data between -1 to 1 and store in the model's
            // input tensor. Accelerometer data ranges between -4 and 4,
            // gyroscope data ranges between -2000 and 2000
            input[numSamplesRead * 6 + 0] = aX / 4.0;
            input[numSamplesRead * 6 + 1] = aY / 4.0;
            input[numSamplesRead * 6 + 2] = aZ / 4.0;
            input[numSamplesRead * 6 + 3] = gX / 2000.0;
            input[numSamplesRead * 6 + 4] = gY / 2000.0;
            input[numSamplesRead * 6 + 5] = gZ / 2000.0;
            
            numSamplesRead++;

            // Do we have the samples we need?
            if (numSamplesRead == NUM_SAMPLES) {
            
            // Stop capturing
            isCapturing = false;
            return true;
            }
        }
    }
    return false;
}

void postprocessing(float* output){
    // Loop through the output tensor values from the model
    int maxIndex = 0;
    float maxValue = 0;
    for (int i = 0; i < NUMBER_OF_OUTPUTS; i++) {
        float _value = output[i];
        if(_value > maxValue){
            maxValue = _value;
            maxIndex = i;
        }
        // Serial.print(GESTURES[i]);
        // Serial.print(": ");
        // Serial.println(output[i], 6);
    }
    
    if(output[maxIndex] > 0.9){
        Serial.print("Winner: ");
        Serial.print(GESTURES[maxIndex]);
        Serial.print(" Prob: ");
        Serial.println(output[maxIndex], 6);
    }

    // Add delay to not double trigger
    delay(CAPTURE_DELAY);
}