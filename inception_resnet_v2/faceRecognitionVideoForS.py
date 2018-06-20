import tensorflow as tf
import cv2
import numpy as np
import sys
import os.path
 
# Create a VideoCapture object and read from input file
video_path = sys.argv[1]
cap = cv2.VideoCapture(video_path)
cascade_file = "lbpcascade_animeface.xml"
cascade = cv2.CascadeClassifier(cascade_file)
sess = tf.Session()
length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
count = 23
faces = None
name = []
resultCount = 0

def onChange(trackbarValue):
    cap.set(cv2.CAP_PROP_POS_FRAMES,trackbarValue)
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
    in tf.gfile.GFile("./data/animeface/labels.txt")]

# Unpersists graph from file
with tf.gfile.FastGFile("./minimal_graph.proto", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')
 
# Check if camera opened successfully
if (cap.isOpened()== False): 
    print("Error opening video stream or file")

# Read until video is completed
while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret == True:
        count += 1
        if count == 24:
            count = 0
            name = []
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
        
            faces = cascade.detectMultiScale(gray,
                                         # detector options
                                         scaleFactor = 1.1,
                                         minNeighbors = 5,
                                         minSize = (24, 24))
            for (x, y, w, h) in faces:
                crop_img = frame[y:y+h, x:x+w]
                if h>96:
                    crop_img = cv2.resize(crop_img, (96, 96), interpolation=cv2.INTER_AREA)
                cv2.imwrite(
                    "tmp/tmp.jpg",
                    crop_img
                )
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                image_data = tf.gfile.FastGFile("tmp/tmp.jpg", 'rb').read()  
                # Feed the image_data as input to the graph and get first prediction
                softmax_tensor = sess.graph.get_tensor_by_name('InceptionResnetV2/Logits/Predictions:0')
                predictions = sess.run(softmax_tensor, 
                {'input_image:0': image_data})
                # Sort to show labels of first prediction in order of confidence
                top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
                first = True
                for node_id in top_k:
                    human_string = label_lines[node_id]
                    score = predictions[0][node_id]
                    print('%s (score = %.5f)' % (human_string, score))
                    if first:
                        cv2.putText(frame, human_string, (x-10, y-10), cv2.FONT_HERSHEY_PLAIN, 2,(0,0,255),2,cv2.LINE_AA)
                        name.append(human_string)
                        first = False
            cv2.imwrite("result/rs_" + str(resultCount) + ".jpg", frame)
            resultCount = resultCount + 1;
        else:
            p = 0
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, name[p], (x-10, y-10), cv2.FONT_HERSHEY_PLAIN, 2,(0,0,255),2,cv2.LINE_AA)
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
