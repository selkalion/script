import tensorflow as tf
import cv2
import sys
import os.path

image_path = sys.argv[1]
cascade_file = "lbpcascade_animeface.xml"
sess = tf.Session()

# Loads label file, strips off carriage return
label_lines = [line.rstrip() for line
    in tf.gfile.GFile("./retrained_labels.txt")]

# Unpersists graph from file
with tf.gfile.FastGFile("./retrained_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
#new
cascade = cv2.CascadeClassifier(cascade_file)
image = cv2.imread(image_path, cv2.IMREAD_COLOR)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.equalizeHist(gray)
    
faces = cascade.detectMultiScale(gray,
        # detector options
        scaleFactor = 1.1,
        minNeighbors = 5,
        minSize = (24, 24))
for (x, y, w, h) in faces:
    crop_img = image[y:y+h, x:x+w]
    cv2.imwrite(
                "tmp/tmp.jpg",
                crop_img
            )
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
    image_data = tf.gfile.FastGFile("tmp/tmp.jpg", 'rb').read()  
    # Feed the image_data as input to the graph and get first prediction
    predictions = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image_data})
    # Sort to show labels of first prediction in order of confidence
    top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
    first = True
    for node_id in top_k:
        human_string = label_lines[node_id]
        score = predictions[0][node_id]
        print('%s (score = %.5f)' % (human_string, score))
        if first:
            cv2.putText(image, human_string, (x-10, y-10), cv2.FONT_HERSHEY_PLAIN, 2,(0,0,255),2,cv2.LINE_AA)
            first = False

cv2.imshow("AnimeFaceDetect", image)
cv2.waitKey(0)
cv2.imwrite("out.png", image)

    

