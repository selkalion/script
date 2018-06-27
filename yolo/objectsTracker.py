import cv2
import numpy as np
import sys
import os.path
from darkflow.net.build import TFNet
 
# Create a VideoCapture object and read from input file
video_path = sys.argv[1]
cap = cv2.VideoCapture(video_path)
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
count = 11
objs = None
name = []
colorMap = []

options = {"model": "yolov2.cfg", "load": "yolov2.weights", "threshold": 0.4, "gpu": 0.6}

tfnet = TFNet(options)

def onChange(trackbarValue):
    cap.set(cv2.CAP_PROP_POS_FRAMES, trackbarValue)
    err,img = cap.read()
    cv2.imshow("ObjectDetect", img)
    pass

cv2.namedWindow('ObjectDetect')
cv2.createTrackbar( 'seek', 'ObjectDetect', 0, length, onChange )

onChange(0)
seek = cv2.getTrackbarPos('seek','ObjectDetect')
cap.set(cv2.CAP_PROP_POS_FRAMES,seek)
 
# Check if camera opened successfully
if (cap.isOpened()== False): 
    print("Error opening video stream or file")

text_file = open("labels.txt", "r")
labels = text_file.read().split('\n')
print(labels)
colorMap = tfnet.meta['colors']
print(colorMap)

# Read until video is completed
while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
        #resize if video wider than 1280
        if cap.get(3) > 1280:
            frame = cv2.resize(frame, (1280, 720))
        count += 1
        if count == 12:
            count = 0
            name = []
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)

            objs = tfnet.return_predict(frame)
            print(objs)
            for res in objs:
                left = res['topleft']['x']
                top = res['topleft']['y']
                right = res['bottomright']['x']
                bottom = res['bottomright']['y'] 
                label = res['label']
                colorIndex = labels.index(label)
                color = colorMap[colorIndex]
                confidence = res['confidence']
                imgHeight, imgWidth, _ = frame.shape
                thick = int((imgHeight + imgWidth) // 300)

                cv2.rectangle(frame,(left, top), (right, bottom), color, thick)
                cv2.putText(frame, label, (left, top - 12), 0, 1e-3 * imgHeight,
                    color, thick//3)
        else:
            for res in objs:
                left = res['topleft']['x']
                top = res['topleft']['y']
                right = res['bottomright']['x']
                bottom = res['bottomright']['y'] 
                label = res['label']
                colorIndex = labels.index(label)
                color = colorMap[colorIndex]
                confidence = res['confidence']
                imgHeight, imgWidth, _ = frame.shape
                thick = int((imgHeight + imgWidth) // 300)

                cv2.rectangle(frame,(left, top), (right, bottom), color, thick)
                cv2.putText(frame, label, (left, top - 12), 0, 1e-3 * imgHeight,
                    color, thick//3)          
        # Display the resulting frame 
        cv2.imshow('ObjectDetect',frame)
 
        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
         
    # Break the loop
    else: 
        break

# When everything done, release the video capture object
cap.release()
 
# Closes all the frames
cv2.destroyAllWindows()
