from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from vision.utils.misc import Timer
import cv2
import sys
import numpy as np

model_path = './models/mobilenet-v1-ssd-mp-0_675.pth'
label_path = './models/voc-model-labels.txt'

if len(sys.argv) >= 5:
    cap = cv2.VideoCapture(sys.argv[4])  # capture from file
else:
    cap = cv2.VideoCapture(0)   # capture from camera
    cap.set(3, 1920)
    cap.set(4, 1080)

class_names = [name.strip() for name in open(label_path).readlines()]
num_classes = len(class_names)

net = create_mobilenetv1_ssd(len(class_names), is_test=True)
net.load(model_path)

predictor = create_mobilenetv1_ssd_predictor(net, candidate_size=200)

timer = Timer()

base_path = 'image'
n = 0
ext = 'png'

while True:
    ret, orig_image = cap.read()
    if orig_image is None:
        continue
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    timer.start()
    boxes, labels, probs = predictor.predict(image, 10, 0.4)
    interval = timer.end()
    # print('Time: {:.2f}s, Detect Objects: {:d}.'.format(interval, labels.size(0)))

    labels_np = labels.numpy()
    if np.any(labels_np == 15) and np.any(labels_np == 8):
        print("cat and person!!!!")
        cv2.imwrite('{}_{}.{}'.format(base_path, n, ext), orig_image)
        n += 1
    
    for i in range(boxes.size(0)):
        box = boxes[i, :]
        label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
        # print(label)
        cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)

        cv2.putText(orig_image, label,
                    (box[0]+20, box[1]+40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,  # font scale
                    (255, 0, 255),
                    2)  # line type
    cv2.imshow('annotated', orig_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
