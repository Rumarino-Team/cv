# Array of detections from YOLO inference
# Includes timestamp for synchronization

std_msgs/Header header

# Array of all detections in this frame
Detection[] detections

# Image dimensions the detections correspond to
uint32 image_width
uint32 image_height

# Model information
string model_name
string model_version
