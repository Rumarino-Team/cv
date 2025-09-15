
### Create Virtual Enviroment
 We have run this module with the Python 3.12 version.
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt 
```

### Activate workspace
```bash
mkdir -p ros2_ws/src
ln hydrus_cv ros2_ws/src/hydrus_cv
colcon build
```
### Running Bencharks
```bash
cd ../ # We are going to the script as a python module and for that we need to outside
python -m cv.pyCV.benchmark
```


### Running  the Ros2 Computer Vision Node
 TODO:
