from darkflow.net.build import TFNet
import cv2

options = {"pbLoad": "yolo-voc-afed.pb", "metaLoad": "yolo-voc-afed.meta", "threshold": 0.5,"gpu": 0.6}

tfnet = TFNet(options)

imgcv = cv2.imread("r.jpg")
results = tfnet.return_predict(imgcv)
print(results)
colorMap = tfnet.meta['colors']
for res in results:
    left = res['topleft']['x']
    top = res['topleft']['y']
    right = res['bottomright']['x']
    bottom = res['bottomright']['y']
    #colorIndex = res['coloridx']
    #color = colorMap[colorIndex]
    label = res['label']
    confidence = res['confidence']
    imgHeight, imgWidth, _ = imgcv.shape
    thick = int((imgHeight + imgWidth) // 300)

    cv2.rectangle(imgcv,(left, top), (right, bottom), (0, 0, 255), thick)
    cv2.putText(imgcv, label, (left, top - 12), 0, 1e-3 * imgHeight,
        (0, 0, 255), thick//3)
cv2.imwrite('./output.jpg', imgcv)
