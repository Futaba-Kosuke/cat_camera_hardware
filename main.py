import os
import argparse
import cv2
import numpy as np
import sys
import time
from threading import Thread
import importlib.util
import firebase_admin
from firebase_admin import credentials, firestore, storage
import datetime
import platform
import base64
import requests
import json

class VideoStream:
    
    def __init__(self, resolution=(640, 480), framerate=30):
        self.stream = cv2.VideoCapture(0)

        self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.stream.set(3, resolution[0])
        self.stream.set(4, resolution[1])

        self.grabbed, self.frame = self.stream.read()

        self.stopped = False

    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                self.stream.release()
                return

            self.grabbed, self.frame = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True

class Firebase:

    def __init__(self, cred_path='./firebase/cred.json', bucket_name='cat-camera-cafe8.appspot.com'):
        cred = credentials.Certificate(cred_path)
        firebase_admin.initialize_app(cred, {'storageBucket': bucket_name})

        self.bucket_name = bucket_name
        self.bucket = storage.bucket()

        self.db = firestore.client()

    def upload_file(self, file_path='./firebase/image.jpeg', content_type='image/jpeg', meta_data={'from': platform.system()}):
        blob = self.bucket.blob(file_path.split('/')[-1])
        with open(file_path, 'rb') as f:
            blob.upload_from_file(f, content_type=content_type)
        blob.metadata = meta_data
        blob.patch()

        blob.make_public()
        url = blob.public_url
        self.db.collection("images").add({
            'isFavorite': False,
            'url': url,
            'postTime': datetime.datetime.now()
        })

        os.remove(file_path)

def numpy_to_base64(img_np):
    # numpyをbase64に変換
    _, temp = cv2.imencode('.jpeg', img_np)
    img_base64 = base64.b64encode(temp)
    return img_base64

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--modeldir', help='model\'s path', default='models')
    parser.add_argument('--display', help='flag\'s window', default=True)
    parser.add_argument('-w', '--wait', help='waiting time [sec]', default=5)
    parser.add_argument('-s', '--server', default=True)

    args = parser.parse_args()
    
    MODEL_NAME = args.modeldir
    IS_DISPLAY = args.display
    WAITING_SEC = float(args.wait)
    IS_SERVER = args.server

    GRAPH_NAME = 'detect.tflite'
    LABELMAP_NAME = 'labelmap.txt'
    min_conf_threshold = float(0.5)
    imgW, imgH = 1280, 720

    print('WAITING_SEC: {}'.format(WAITING_SEC))

    pkg = importlib.util.find_spec('tflite_runtime')
    if pkg:
        from tflite_runtime.interpreter import Interpreter
        print('use tflite')
    
    else:
        from tensorflow.lite.python.interpreter import Interpreter
        print('use tf')
    
    CWD_PATH = os.getcwd()
    PATH_TO_CKPT = os.path.join(CWD_PATH, MODEL_NAME, GRAPH_NAME)
    PATH_TO_LABELS = os.path.join(CWD_PATH, MODEL_NAME, LABELMAP_NAME)
    
    with open(PATH_TO_LABELS, 'r') as f:
        labels = [line.strip() for line in f.readlines()]
    
    if labels[0] == '???':
        del(labels[0])
    
    labels = np.array(labels)
    
    interpreter = Interpreter(model_path=PATH_TO_CKPT)
    
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]
    
    floating_model = (input_details[0]['dtype'] == np.float32)
    
    input_mean = 127.5
    input_std = 127.5
    
    frame_rate_calc = 1
    freq = cv2.getTickFrequency()
    
    videostream = VideoStream(resolution=(imgW, imgH), framerate=30).start()
    time.sleep(1)

    firebase = Firebase()
    
    # 時間の初期値
    prev_time = 0
    while True:
        t1 = cv2.getTickCount()
    
        frame = videostream.read()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (width, height))
        input_data = np.expand_dims(frame_resized, axis=0)
    
        if floating_model:
            input_data = (np.float32(input_data) - input_mean) / input_std
    
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()

        boxes = interpreter.get_tensor(output_details[0]['index'])[0]
        classes = interpreter.get_tensor(output_details[1]['index'])[0]
        scores = interpreter.get_tensor(output_details[2]['index'])[0]
    
        labels_valid = labels[np.int64(classes[(scores > min_conf_threshold) & (scores <= 1.0)])]
        # 現在の時間
        now_time = time.time()
        # 猫／人間が含まれている ∧ 前回の撮影からWAITING_SECだけ時間が経過している
        if ('cat' in labels_valid) and ('person' in labels_valid) and (now_time - prev_time > WAITING_SEC):
            print('cat_and_person')
            file_name = datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')

            if IS_SERVER:
                frame_base64 = numpy_to_base64(frame)
                Thread(target=requests.post, args=('http://127.0.0.1:5000/upload', { 'img_base64': frame_base64 })).start()
            else:
                file_path = './firebase/{}.jpeg'.format(file_name)
                cv2.imwrite(file_path, frame)
                Thread(target=firebase.upload_file, kwargs={ 'file_path': './firebase/{}.jpeg'.format(file_name) }).start()

            # 時間の更新
            prev_time = now_time
    
        if IS_DISPLAY:
            for i in range(len(scores)):
                if (scores[i] > min_conf_threshold) and (scores[i] <= 1.0):
                    x_min, x_max = int(max(1, (boxes[i][1] * imgW))), int(min(imgW, (boxes[i][3] * imgW)))
                    y_min, y_max = int(max(1, (boxes[i][0] * imgH))), int(min(imgH, (boxes[i][2] * imgH)))
    
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (10, 255, 0), 2)
    
                    object_name = labels[int(classes[i])]
                    label = '%s: %d%%' % (object_name, int(scores[i]*100))
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    label_ymin = max(y_min, labelSize[1] + 10)
                    cv2.rectangle(frame, (x_min, label_ymin-labelSize[1]-10), (x_min+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED)
                    cv2.putText(frame, label, (x_min, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
            cv2.putText(frame, 'FPS: {0:.2f}'.format(frame_rate_calc), (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow('Object detector', frame)
    
            t2 = cv2.getTickCount()
            time1 = (t2 - t1) / freq
            frame_rate_calc = 1 / time1
    
            if cv2.waitKey(1) == ord('q'):
                break
    
    cv2.destroyAllWindows()
    videostream.stop()
