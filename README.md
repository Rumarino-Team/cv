
### Download repo with third_party
Just download all the extra dependencies for running everything
```bash
git clone --recursive https://github.com/Rumarino-Team/cv.git
```
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
python -m src.pyCV.benchmark
```

### Install Ros2 and dependencies
```bash
sudo apt update
sudo apt install ros-jazzy-vision-msgs
sudo apt install ros-jazzy-sensor-msgs
sudo apt install ros-jazzy-geometry-msgs
sudo apt install ros-jazzy-rviz2
sudo apt install libogre-1.12-dev

```
### Running  the Ros2 Computer Vision Node

### Using RVIZ2

### Using USBCamera in ROS2
install package
```bash
sudo apt install ros-jazzy-usb-cam
sudo apt install python3-pydantic
```
Run the camera node
```bash
cd ros2_ws
source install/setup.bash
ros2 launch usb_cam camera.launch.py

``` 

### Optional ORB_SLAM3
#IN PROGRESS
link to the ORB_SLAM ROS2 app
https://github.com/Cruiz102/ORB_SLAM3.git

### hydrus_cv packages services and launches

#TODO



 



