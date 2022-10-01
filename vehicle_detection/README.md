# Vehicle detection and distance estimation using opencv, yoloV5 and python

## Features:
-> Model input size = 256 X 256  
-> Model is int8 quantized for faster inference  
-> Model inference and nms is implemented with the help of opencv dnn  

## Limitations:
-> Due to the small input size, distant detections cannot be made accurately due to significant loss of details  
-> Raspberry pi doesn't have much computation power and therefore, inferences are much slower than that of a smartphone  
-> Faster inference can be obtained with the help of tflite models  

## Usage:
    
By default the program uses a webcam at source 0. To change the source execute:  
    ```python detect.py --source 0/1/2```  
    To switch to a video file do:  
    ```python detect.py --source <file path>```  

By default the program doesn't store the output. To store the output do:  
        ```python detect.py --destination <filename>.avi```  

By default the program displays live output. To remove that execute:  
    ```python detect.py --noshow --destination <filename.avi>```  
    ```--noshow``` **should be used with** ```--destination```  

The program also supports picamera! Use:  
    ```python detect.py --picamera```  
    to switch to picamera.  

The program has the option to switch to a different yolo-model like so:  
    ```python detect.py --model "model_path" --inp_size width height```  
