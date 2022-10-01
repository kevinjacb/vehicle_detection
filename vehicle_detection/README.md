# Vehicle detection and distance estimation using opencv, yoloV5 and python

## Features:
-> Model input size = 256 X 256__
-> Model is int8 quantized for faster inference__
-> Model inference and nms is implemented with the help of opencv dnn__

## Limitations:
-> Due to the small input size, distant detections cannot be made accurately due to significant loss of details__
-> Raspberry pi doesn't have much computation power and therefore, inferences are much slower than that of a smartphone__
-> Faster inference can be obtained with the help of tflite models__

## Usage:
    
By default the program uses a webcam at source 0. To change the source execute:__
    ```python detect.py --source 0/1/2```__
    To switch to a video file do:__
    ```python detect.py --source <file path>```__

By default the program doesn't store the output. To store the output do:__
        ```python detect.py --destination <filename>.avi```__

By default the program displays live output. To remove that execute:__
    ```python detect.py --noshow --destination <filename.avi>```__
    ```--noshow``` **should be used with** ```--destination```__

The program also supports picamera! Use:__
    ```python detect.py --picamera```__
    to switch to picamera.__

The program has the option to switch to a different yolo-model like so:__
    ```python detect.py --model "model_path" --inp_size width height```__
