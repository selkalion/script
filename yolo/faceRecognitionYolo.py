import cv2
import numpy as np
import sys
import os.path
import tensorflow as tf
from darkflow.net.build import TFNet
 
# Create a VideoCapture object and read from input file
video_path = sys.argv[1]
cap = cv2.VideoCapture(video_path)
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
count = 11
faces = None
name = []
sess = tf.Session()

options = {"pbLoad": "yolo-voc-afed.pb", "metaLoad": "yolo-voc-afed.meta", "threshold": 0.4, "gpu": 0.6}

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

# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line
    in tf.gfile.GFile("../transfer-learning-anime/retrained_labels.txt")]

# Unpersists graph from file
with tf.gfile.FastGFile("../transfer-learning-anime/retrained_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
sess.run(softmax_tensor)
 
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
                label = res['label']
                if label == 'face':
                    left = res['topleft']['x']
                    top = res['topleft']['y']
                    right = res['bottomright']['x']
                    bottom = res['bottomright']['y']
                    confidence = res['confidence']

                    crop_img = frame[top:bottom, left:right]
                    cv2.imwrite(
                        "tmp/tmp.jpg",
                        crop_img
                    )
                    cv2.rectangle(frame,(left, top), (right, bottom), (0, 0, 255), 2)
                    image_data = tf.gfile.FastGFile("tmp/tmp.jpg", 'rb').read()  
                    # Feed the image_data as input to the graph and get first prediction
                    predictions = sess.run(softmax_tensor, 
                    {'DecodeJpeg/contents:0': image_data})
                    # Sort to show labels of first prediction in order of confidence
                    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
                    first = True
                    for node_id in top_k:
                        human_string = label_lines[node_id]
                        score = predictions[0][node_id]
                        print('%s (score = %.5f)' % (human_string, score))
                        if first:
                            cv2.putText(frame, human_string, (left, top - 10), cv2.FONT_HERSHEY_PLAIN, 2,(0,0,255),2,cv2.LINE_AA)
                            name.append(human_string)
                            first = False
        else:
            p = 0
            for res in faces:
                label = res['label']
                if label == 'face':
                    left = res['topleft']['x']
                    top = res['topleft']['y']
                    right = res['bottomright']['x']
                    bottom = res['bottomright']['y']
                    confidence = res['confidence']
                    cv2.rectangle(frame,(left, top), (right, bottom), (0, 0, 255), 2)
                    cv2.putText(frame, name[p], (left, top - 10), cv2.FONT_HERSHEY_PLAIN, 2,(0,0,255),2,cv2.LINE_AA)
                    p += 1               
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
