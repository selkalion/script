import cv2
import numpy as np
import sys
import os.path
from darkflow.net.build import TFNet
 
# Create a VideoCapture object and read from input file
video_path = sys.argv[1]
cap = cv2.VideoCapture(video_path)
cascade_file = "lbpcascade_animeface.xml"
cascade = cv2.CascadeClassifier(cascade_file)
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
count = 11
faces = None
name = []

options = {"pbLoad": "yolo-voc-afed.pb", "metaLoad": "yolo-voc-afed.meta", "threshold": 0.5, "gpu": 0.6}

tfnet = TFNet(options)

def onChange(trackbarValue):
    cap.set(cv2.CAP_PROP_POS_FRAMES, trackbarValue)
    err,img = cap.read()
    cv2.imshow("AnimeFaceDetect", img)
    pass

cv2.namedWindow('AnimeFaceDetect')
cv2.createTrackbar( 'seek', 'AnimeFaceDetect', 0, length, onChange )

onChange(0)
seek = cv2.getTrackbarPos('seek','AnimeFaceDetect')
cap.set(cv2.CAP_PROP_POS_FRAMES,seek)
 
# Check if camera opened successfully
if (cap.isOpened()== False): 
    print("Error opening video stream or file")
 
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

            faces = tfnet.return_predict(frame)
            for res in faces:
                left = res['topleft']['x']
                top = res['topleft']['y']
                right = res['bottomright']['x']
                bottom = res['bottomright']['y']
                label = res['label']
                confidence = res['confidence']

                if label == 'face':
                    cv2.rectangle(frame,(left, top), (right, bottom), (0, 0, 255), 2)
        else:
            for res in faces:
                left = res['topleft']['x']
                top = res['topleft']['y']
                right = res['bottomright']['x']
                bottom = res['bottomright']['y']
                label = res['label']
                confidence = res['confidence']
                
                if label == 'face':
                    cv2.rectangle(frame,(left, top), (right, bottom), (0, 0, 255), 2)               
        # Display the resulting frame 
        cv2.imshow('AnimeFaceDetect',frame)
 
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
