import cv2 as cv
import argparse
import numpy as np
import math
import time


labels = dict({0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'})

parser = argparse.ArgumentParser(description='Detection settings')
parser.add_argument('--model',default='yolov5n-updated.onnx')
parser.add_argument('--inp_size',default=[256,256],nargs=2)
parser.add_argument('--picamera',default=False,action='store_true')
parser.add_argument('--source',default=0)
parser.add_argument('--destination',default='')
parser.add_argument('--noshow',default=True,action='store_false')
parser.add_argument('--conf_thresh',default=0.3,)
parser.add_argument('--dist_thresh',default=[0.5],nargs='+')
parser.add_argument('--showfps',default=False,action='store_true')
parser.add_argument('--fps',default=30)

args = parser.parse_args()

class PiCam:
    def __init__(self):
        #creates an instance of Picamera2
        self.piCam = Picamera2()    
        #initializes the camera with a preview configuration
        self.piCam.configure(self.piCam.create_preview_configuration())   
        self.piCam.start()  #starts the camera

    def read(self):
        #reads frames from the camera as an array of type uint8
        frame = np.array(self.piCam.capture_array()[...,:3],dtype=np.uint8)
        #converts the rgb image to bgr
        frame = cv.cvtColor(frame,cv.COLOR_RGB2BGR)
        return True,frame


#if picamera is set true, use that instead
if args.picamera:
    from picamera2 import Picamera2
    source = PiCam() 
else:
    #initializes source with a video capture object (stream from webcam)
    source = cv.VideoCapture(int(args.source) if isinstance(args.source,int) else args.source)

#extract arguments from the command line
dest = args.destination
show = args.noshow
model = args.model
inp_size = args.inp_size
conf_thresh = float(args.conf_thresh)
dist_thresh = args.dist_thresh
dist_thresh.append(1)
show_fps = args.showfps
fps = args.fps

if not show and len(dest) == 0:
    exit('Destination not specified.')

print('Starting ...')
net = cv.dnn.readNetFromONNX(model) #create a net object

#read sample frame to get width and height
ret,sample_frame = source.read() 
if not ret:
    exit("Couldn't initialize camera. Exiting")
height,width = sample_frame.shape[:2]
frame_area = height * width

#predefining a set of colors in BGR
#colors : green, yellow, orange, red
bb_colors = [[0,255,0],
            [153,255,255],
            [0,128,255],
            [0,0,255]]

#determining the number of colors to skip depending on the number of distance thresholds
color_jump = math.ceil(len(bb_colors)/(len(dist_thresh)-1))

if len(dest) > 0:
    #create a writer to write to file if destination is provided
    writer = cv.VideoWriter(dest,cv.VideoWriter_fourcc(*'XVID'),fps,(width,height)) 


# print(height,width)

while True:
    start = time.time()
    ret, frame = source.read() #read next frame
    if not ret:
        break
    
     #change frame from bgr to rgb
    frame_rgb = cv.cvtColor(frame,cv.COLOR_BGR2RGB)

    #resize and normalize the input image
    inp_blob = cv.dnn.blobFromImage(frame_rgb.astype(np.float32),size=inp_size,scalefactor=1./255.) 
    
    net.setInput(inp_blob)
    
    #output format = boxes(0-3),scores(4),classes(5-84)
    output = net.forward()[0]

    boxes = output[:,:4]
    scores = output[:,4]
    classes = output[:,5:]

    boxes = boxes.astype(np.float32)/inp_size[0] #boxes are normalized to 0-1 range

    indices = cv.dnn.NMSBoxes(boxes,scores,conf_thresh,0.4) #apply nms to output got get the correct bounding boxes

    for index in indices:
        x,y,w,h = boxes[index] #coordinates in the form (x_center,y_center,width,height)

        #calculating start and end points of the rectangle
        x = int(max(0,(x-w/2)*width))
        y = int(min(width,(y-h/2)*height))
        x1 = int(max(0,x+w*width))
        y1 = int(min(height,y+h*height))

        #calculating the area of the bounding box
        area = abs(x-x1)*abs(y-y1)

        #determing the label from the highest class score
        label = labels[np.argmax(classes[index])]

        color = bb_colors[0]
        if x < 30 or y < 30 or abs(x1 - width) < 30 or abs(y1 - height) < 30:
            color = bb_colors[-1]
        elif (x > 30 and x < 100)  or (y > 30 and y < 100) or (abs(x1 - width) > 30 and abs(x1-width) < 100) or (abs(y1 - height) > 30 and abs(y1-height) < 100):
            color = bb_colors[-2]
        else:
            for i,thresh in enumerate(dist_thresh):
                if area/frame_area < float(thresh):
                    color = bb_colors[i*color_jump if i*color_jump < len(bb_colors) else -1]
                    break
        cv.rectangle(frame,(x,y),(x1,y1),color,3)
        cv.putText(frame,f'{label} {int(scores[index]*100)}%',(x,max(y-20,0)),cv.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),3)
        if show_fps:
            cv.putText(frame,'fps: {fps:0.2f}'.format(fps = 1/(time.time() - start)) ,(30,30),cv.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)


    if show:
        #displays all the frames on the screen live
        cv.imshow('frame',frame)
        if cv.waitKey(10) & 0xFF == 27:
            break

    if len(dest) > 0:
        #write to destination if provided
        writer.write(frame)

if len(dest) > 0:
    writer.release()
if not args.picamera:    
    source.release()

cv.destroyAllWindows()
print("Done!")