# Deploy Tensorflow Lite Micro Model on the Fly (Arduino)

This project contains Arduino sketch code for loading and deploying TFLite micro model to an [Arduino Nano 33 BLE Sense board](https://store-usa.arduino.cc/products/arduino-nano-33-ble-sense) on the fly via BLE.

## Project Structure
* [lib](https://code.siemens.com/haoyu.ren/tinyml-research-symposium-2022-haoyu/-/tree/main/Demo_Arduino_Motion/lib): the dependency libraries for the project, for example, BLE, IMU supporting library.
* [models](https://code.siemens.com/haoyu.ren/tinyml-research-symposium-2022-haoyu/-/tree/main/Demo_Arduino_Motion/models): example models for demonstration
* [src](https://code.siemens.com/haoyu.ren/tinyml-research-symposium-2022-haoyu/-/tree/main/Demo_Arduino_Motion/src): Arduino sketch code

## Use
1. Connect the board to a development PC
2. Download and run our Arduino sketch (in Plaftform IO or Arduino IDE). 
3. Use Chrome to open the [Website](https://haoyu-r.github.io/BLE_Transfer/website/index.html) for tflite model BLE transmitting which is based on the [WebBLE example](https://github.com/petewarden/ble_file_transfer)
4. **Connect** your device via BLE
5. Open the serial port
6. **Choose File** in the [models](https://code.siemens.com/haoyu.ren/tinyml-research-symposium-2022-haoyu/-/tree/main/Demo_Arduino_Motion/models) folder
7. **Transfer File** to deploy the model to the board via BLE
8. Have fun