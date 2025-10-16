
### Download repo with third_party
Just download all the extra dependencies for running everything
```bash
git clone --recursive https://github.com/Rumarino-Team/cv.git
cd cv
```
### Create Virtual Enviroment
 We have run this module with the Python 3.12 version.
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt 
```

### Create workspace
```bash
mkdir -p ros2_ws/src
ln hydrus_cv ros2_ws/src/hydrus_cv
```

### Running Bencharks
```bash
cd ros2_ws
python -m src.hydrus_cv.hydrus_cv.benchmark
```

### Install Ros2 and dependencies
```bash
sudo apt update
sudo apt install ros-jazzy-vision-msgs
sudo apt install ros-jazzy-sensor-msgs
sudo apt install ros-jazzy-geometry-msgs
sudo apt install ros-jazzy-rviz2
sudo apt install libogre-1.12-dev
sudo apt install ros-jazzy-usb-cam

#Orb SLAM dependencies
sudo apt install libopencv-dev libeigen3-dev libboost-all-dev libssl-dev
sudo apt install ros-jazzy-pangolin
```

### Building Orb Slam
```bash
cd third_party
wget https://github.com/UZ-SLAMLab/ORB_SLAM3/raw/refs/heads/master/Vocabulary/ORBvoc.txt.tar.gz
tar -xf ORBvoc.txt.tar.gz
git clone https://github.com/Cruiz102/ORB_SLAM3.git
#build everything with a single command
cd ORB_SLAM3
chmod +x build.sh
sudo ./build.sh
```


### Building Ros Packages
```bash
# Build
./build.sh
source /opt/ros/jazzy/setup.bash
source install/setup.bash

```



#### Individual Components

**Full HydrusCV Pipeline (without SLAM):**
```bash
ros2 launch hydrus_cv full_pipeline.launch.py
```

**ORB-SLAM3 only:**
```bash
ros2 launch orb_slam mono_webcam.launch.py
```

**Depth estimation only:**
```bash
ros2 launch hydrus_cv depth_publisher.launch.py
```





 



