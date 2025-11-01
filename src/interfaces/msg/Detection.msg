# Single YOLO detection
# Represents one detected object with bounding box and classification

# Class ID (0-N)
int32 class_id

# Class name (e.g., "gate", "buoy", "shark")
string class_name

# Confidence score (0.0 - 1.0)
float32 confidence

# 2D Bounding box in image coordinates (pixels)
float32 x_min  # Left edge
float32 y_min  # Top edge
float32 x_max  # Right edge
float32 y_max  # Bottom edge

# Center point of the bounding box
float32 center_x
float32 center_y

# Width and height of the bounding box
float32 width
float32 height
