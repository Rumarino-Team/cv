
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

### Install Orb Slam

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


### Building Ros2 Orb Slam 
```bash
# Build
cd ~/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select orb_slam3_ros2
source install/setup.bash

# Run with webcam
ros2 launch orb_slam3_ros2 mono_webcam.launch.py
```

```



### Ros2 Launch commands

#### Combined SLAM + Computer Vision Pipeline
Run both ORB-SLAM3 and HydrusCV together for complete localization and object detection:
```bash
cd ros2_ws
source install/setup.bash
ros2 launch hydrus_cv slam_cv_pipeline.launch.py
```

With custom parameters:
```bash
ros2 launch hydrus_cv slam_cv_pipeline.launch.py \
    camera_device:=/dev/video2 \
    image_width:=1280 \
    image_height:=720 \
    use_orb_viewer:=false
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





 



