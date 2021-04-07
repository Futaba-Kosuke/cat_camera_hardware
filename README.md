# cat_camera_hardware

RaspberryPiのカメラで飼い猫と飼い主を認識し、ツーショットを自動撮影する監視カメラです。  
撮影した写真はアプリからの閲覧が可能な他、複数の猫の分類を実装しています。  

九州アプリチャレンジキャラバン: 🏆福岡ビジネスデジタルコンテンツ賞  
福岡ビジネスデジタルコンテンツ: 🏆ヤング賞  

<img width="500" alt="スクリーンショット 2021-04-08 0 07 03" src="https://user-images.githubusercontent.com/49780545/113890630-4c732000-97ff-11eb-9e5d-a4123457ca4f.png">
<img width="500" alt="スクリーンショット 2021-04-08 0 07 30" src="https://user-images.githubusercontent.com/49780545/113890668-5563f180-97ff-11eb-9ae6-7fd24a0b55e1.png">

## Requirement

- python 3.7.3
- pip 20.2.3
 
## Installation

1. clone repository
```sh
git clone git@github.com:Futaba-Kosuke/cat_camera_hardware.git
cd cat_camera_hardware
```

2. install modules
```sh
python3 -m venv .env
source .env/bin/activate
python -m pip install -r requirements.txt
```
```sh
# MacOS / python3.7.x
python -m pip install https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp37-cp37m-macosx_10_14_x86_64.whl

# raspberry_pi
python -m pip install tensorflow
```

3. add firebase credential json file 
```sh
mv <cred.json path> ./firebase/cred.json
```
 
## Usage
 
```bash
# Start the server
source .env/bin/activate
python main.py
```

## Used
- RaspberryPi
- Tensorflow
- Firestore
- Cloud storage for firebase