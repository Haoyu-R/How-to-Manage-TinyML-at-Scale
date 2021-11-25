// Based on EloquentTinyMLnoIO.h - a wrapper class from EloquentTinyML: TensorFlow lite for ESP32, from https://github.com/eloquentarduino/EloquentTinyML
#pragma once
#ifndef _TFLITE_WRAPPER
#define _TFLITE_WRAPPER

#include <math.h>
#include "TensorFlowLite.h"
#include "tensorflow/lite/version.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
// We use all ops resolver as it is expected to receive the model OTA that means user might not know beforehand which operators are in the model.
#include "tensorflow/lite/micro/all_ops_resolver.h"
// #include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

namespace TFLITE_Wrapper{
    namespace TinyML{

        enum TfLiteError {
            OK,
            VERSION_MISMATCH,
            CANNOT_ALLOCATE_TENSORS,
            NOT_INITIALIZED,
            INVOKE_ERROR
        };

        /**
         * Interface to Tensorflow Lite for Microcontrollers
         *
         * @tparam inputSize
         * @tparam outputSize
         * @tparam tensorArenaSize how much memory to allocate to the tensors
         */
        template<size_t inputSize, size_t outputSize, size_t tensorArenaSize>
        class TFLite{
            public:
                /**
                 * Contructor
                 * @param modelData a binary tflite model as exported 
                 */

                TFLite() : failed(false) {}

                ~TFLite(){
                    delete interpreter;
                    delete model;
                }

                /**
                 * Inizialize NN
                 *
                 * @param modelData
                 * @return
                 */
                bool begin(const unsigned char *modelData) {

                    this->model = tflite::GetModel(modelData);

                    // assert model version and runtime version match
                    if (this->model->version() != TFLITE_SCHEMA_VERSION) {
                        this->failed = true;
                        this->error = VERSION_MISMATCH;
                        Serial.print("Version: ");
                        Serial.println(this->model->version());
                        return false;
                    }

                    // This pulls in all the operation implementations we need. 
                    // NOLINTNEXTLINE(runtime-global-variables)
                    static tflite::AllOpsResolver resolver;

                    static tflite::MicroInterpreter static_interpreter(this->model, resolver, this->tensorArena, tensorArenaSize, &this->micro_error_reporter);
                    this->interpreter = &static_interpreter;

                    if (this->interpreter->AllocateTensors() != kTfLiteOk) {
                        this->failed = true;
                        this->error = CANNOT_ALLOCATE_TENSORS;
                        return false;
                    }

                    this->input = this->interpreter->input(0);
                    this->output = this->interpreter->output(0);
                    this->error = OK;
                    failed = false;
                    return true;
                }

                bool initialized() {
                    return !failed;
                }

                float predict(float *input, float *output) {
                    // Abort if initialization failed
                    if( !initialized() ) {
                        this->error = NOT_INITIALIZED;
                        return sqrt(-1);
                    }

                    // Copy input with scaling
                    for(size_t i = 0; i < inputSize; ++i)
                        this->input->data.f[i] = input[i];

                    // run inference
                    if( this->interpreter->Invoke() != kTfLiteOk ) {
                        this->error = INVOKE_ERROR;
                        return sqrt(-1);
                    }

                    // Copy output
                    for(size_t i = 0; i < outputSize; ++i) {
                        output[i] = (float) this->output->data.f[i];
                    }
                    
                    return output[0];
                }
                
                uint8_t predictClass(float *input, int inSize, float *output, int outSize) {
                    predict(input, inSize, output, outSize);

                    uint8_t classIdx = 0;
                    float maxProba = output[0];

                    for (size_t i = 1; i < outSize; i++) {
                        if (output[i] > maxProba) {
                            classIdx = i;
                            maxProba = output[i];
                        }
                    }
                    return classIdx;
                }
                /**
                 * Get error msg
                */
                const char* errorMessage() {
                    switch (this->error) {
                        case OK:
                            return "No error";
                        case VERSION_MISMATCH:
                            return "Version mismatch";
                        case CANNOT_ALLOCATE_TENSORS:
                            return "Cannot allocate tensors";
                        case NOT_INITIALIZED:
                            return "Interpreter has not been initialized";
                        case INVOKE_ERROR:
                            return "Interpreter invoke() returned an error";
                        default:
                            return "Unknown error";
                    }
                }

                

            protected:
                bool failed;
                TfLiteError error;
                tflite::ErrorReporter* error_reporter = nullptr;
                tflite::MicroInterpreter* interpreter = nullptr;
                tflite::MicroErrorReporter micro_error_reporter;
                TfLiteTensor* input = nullptr;
                TfLiteTensor* output = nullptr;
                
                const tflite::Model* model = nullptr;
                float* outputResult;

                // Create an area of memory to use for input, output, and intermediate arrays.
                // Minimum arena size, at the time of writing. After allocating tensors
                // you can retrieve this value by invoking interpreter.arena_used_bytes().
                // // Extra headroom for model + alignment + future interpreter changes.
                uint8_t tensorArena[tensorArenaSize] __attribute__((aligned(16)));
        };



    }



}

#endif