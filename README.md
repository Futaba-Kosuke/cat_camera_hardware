# cat_camera_hardware

## How to setup your environment

- python 3.7.3
- pip 20.2.3

1. clone repository
```sh
git clone git@github.com:Futaba-Kosuke/cat_camera_hardware.git
cd cat_camera_hardware
```

2. install modules
```sh
python3 -m venv .env
source .env/bin/activate
.env/bin/python3 -m pip install -r requirements.txt
```
```sh
# MacOS / python3.7.x
.env/bin/python3 -m pip install https://dl.google.com/coral/python/tflite_runtime-2.1.0.post1-cp37-cp37m-macosx_10_14_x86_64.whl

# raspberry_pi
.env/bin/python3 -m pip install tensorflow
```

3. add firebase credential json file 
```sh
mv <cred.json path> ./firebase/cred.json
```

4. Run
```sh
.env/bin/python3 main.py
```
