import os, sys
import argparse
import time
import datetime
import platform
import base64
import json
import importlib.util
from threading import Thread

import numpy as np
import cv2
import requests

from video.video import VideoStream
from firebase.firebase import Firebase

min_conf_threshold = float(0.5)
FRAME_SIZE = (1280, 720)
INPUT_MEAN = 255 / 2
INPUT_STD = 255 / 2

def numpy_to_base64(img_np):
    # numpyをbase64に変換
    _, temp = cv2.imencode('.jpeg', img_np)
    img_base64 = base64.b64encode(temp)
    return img_base64

def main():
    # コマンドライン引数を展開
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', help='model path', default='./models/detect.tflite')
    parser.add_argument('-l', '--label', help='labels path', default='./models/labelmap.txt')
    parser.add_argument('-s', '--server', help='flag server', action='store_false')
    parser.add_argument('-d', '--display', help='flag window', action='store_false')
    parser.add_argument('-w', '--wait', help='waiting time [sec]', default=5)

    args = parser.parse_args()
    MODEL_PATH = os.path.abspath(args.model)
    LABEL_PATH = os.path.abspath(args.label)
    IS_SERVER = args.server
    IS_DISPLAY = args.display
    WAITING_SEC = float(args.wait)

    print('MODEL_PATH: {}'.format(MODEL_PATH))
    print('LABEL_PATH: {}'.format(LABEL_PATH))
    print('IS_SERVER: {}'.format(IS_SERVER))
    print('IS_DISPLAY: {}'.format(IS_DISPLAY))
    print('WAITING_SEC: {}'.format(WAITING_SEC))

    # TensorFlowのインポート
    # 環境に合ったものを選択
    pkg = importlib.util.find_spec('tflite_runtime')
    if pkg:
        from tflite_runtime.interpreter import Interpreter
        print('use tflite')
    else:
        from tensorflow.lite.python.interpreter import Interpreter
        print('use tf')

    # labelの読み込み
    with open(LABEL_PATH, 'r') as f:
        labels = [line.strip() for line in f.readlines()]
    
    if labels[0] == '???':
        del(labels[0])

    labels = np.array(labels)
    
    # 学習モデルの読み込み
    interpreter = Interpreter(model_path=MODEL_PATH)
    # メモリの確保
    interpreter.allocate_tensors()
    # 各種モデル情報の取得
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_size = (input_details[0]['shape'][1], input_details[0]['shape'][2])
    is_floating_model = (input_details[0]['dtype'] == np.float32)
    
    frame_rate_calc = 1
    freq = cv2.getTickFrequency()
    
    # 各クラスのインスタンス生成
    videostream = VideoStream(resolution=FRAME_SIZE, framerate=30)
    firebase = Firebase()
    
    # 撮影スタート
    videostream.start()
    # 別スレッド処理なのでセットアップまで待つ
    time.sleep(1)
    
    # 時間の初期値
    prev_time = 0
    while True:
        t1 = cv2.getTickCount()
    
        # フレームの取得〜入力データへの整形
        frame = videostream.read()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, input_size)
        input_data = np.expand_dims(frame_resized, axis=0)
        
        if is_floating_model:
            input_data = (np.float32(input_data) - INPUT_MEAN) / INPUT_STD

        # 予測の実行
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
 
        # 予測結果の取得
        boxes = interpreter.get_tensor(output_details[0]['index'])[0]
        classes = interpreter.get_tensor(output_details[1]['index'])[0]
        scores = interpreter.get_tensor(output_details[2]['index'])[0]
 
        labels_valid = labels[np.int64(classes[(scores > min_conf_threshold) & (scores <= 1.0)])]

        # 検知結果の検証〜Firebaseへのアップロード
        # 現在の時間
        now_time = time.time()
        # 猫／人間が含まれている ∧ 前回の撮影からWAITING_SECだけ時間が経過している
        if ('cat' in labels_valid) and ('person' in labels_valid) and (now_time - prev_time > WAITING_SEC):
            print('cat_and_person')

            file_name = datetime.datetime.now().strftime('%Y-%m-%dT%H:%M:%S')
            if IS_SERVER:
                frame_base64 = numpy_to_base64(frame).decode('utf-8')

                cat_idx = [i for i, x in enumerate(labels_valid) if x == 'cat']
                cat_boxes = [{}] * len(cat_idx)
                for i, x in enumerate(cat_idx):
                    x_min, x_max = int(max(1, (boxes[x][1] * FRAME_SIZE[0]))), int(min(FRAME_SIZE[0], (boxes[x][3] * FRAME_SIZE[0])))
                    y_min, y_max = int(max(1, (boxes[x][0] * FRAME_SIZE[1]))), int(min(FRAME_SIZE[1], (boxes[x][2] * FRAME_SIZE[1])))
                    box = {
                        'x_min': str(x_min),
                        'x_max': str(x_max),
                        'y_min': str(y_min),
                        'y_max': str(y_max)
                    }
                    cat_boxes[i] = box;

                payload = {
                    'img_base64': frame_base64,
                    'cat_boxes': cat_boxes
                }
                print(payload)
                payload = json.dumps(payload).encode('utf-8')

                headers = { 'Content-Type': 'application/json' }

                Thread(target=requests.post, args=('http://127.0.0.1:5000/upload', ), kwargs={ 'headers': headers, 'data': payload }).start()
            else:
                file_path = './firebase/{}.jpeg'.format(file_name)
                cv2.imwrite(file_path, frame)
                Thread(target=firebase.upload_file, kwargs={ 'file_path': './firebase/{}.jpeg'.format(file_name) }).start()

            # 時間の更新
            prev_time = now_time

        # 映像の表示処理
        if IS_DISPLAY:
            for i in range(len(scores)):
                if (scores[i] > min_conf_threshold) and (scores[i] <= 1.0):
                    x_min, x_max = int(max(1, (boxes[i][1] * FRAME_SIZE[0]))), int(min(FRAME_SIZE[0], (boxes[i][3] * FRAME_SIZE[0])))
                    y_min, y_max = int(max(1, (boxes[i][0] * FRAME_SIZE[1]))), int(min(FRAME_SIZE[1], (boxes[i][2] * FRAME_SIZE[1])))
    
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

if __name__ == '__main__':
    main()
