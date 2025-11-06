from collections import deque, defaultdict
import rclpy
from rclpy.node import Node
from interfaces.msg import Map, Detection, DetectionArray
from interfaces.msg import MapObject as RosMapObject
from vision_msgs.msg import BoundingBox3D
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Pose, PoseStamped
from message_filters import Subscriber, ApproximateTimeSynchronizer
import numpy as np
import os
import sys
from cv_bridge import CvBridge

# Scripts from hydrus_cv reads from a thirdparty folder  that is supposed to be in the repo_root/thirdparty
# and scripts are supposed to be in a ros2 workspace
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if src_path not in sys.path:
    sys.path.insert(0, src_path)
from hydrus_cv.detection_core import map_objects
from hydrus_cv.custom_types import CameraIntrinsics, Point3D, Rotation3D
from hydrus_cv.custom_types import MapObject
from hydrus_cv.custom_types import MapState
from hydrus_cv.custom_types import Detection as CustomDetection
from remote_pdb import RemotePdb
class ComputerVisionPublisher(Node):
    def __init__(self):
        super().__init__("ComputerVisionPublisher")

        self.declare_parameter("rgb_topic", "/webcam_rgb")
        self.declare_parameter("depth_topic", "/webcam_depth")
        self.declare_parameter("camera_info_topic", "/webcam_intrinsics")
        self.declare_parameter("imu_topic", "/imu")
        self.declare_parameter("annotated_image_topic", "annotated_image")
        rgb_topic = self.get_parameter("rgb_topic").get_parameter_value().string_value
        depth_topic = (
            self.get_parameter("depth_topic").get_parameter_value().string_value
        )
        cam_intrinsic_topic = (
            self.get_parameter("camera_info_topic").get_parameter_value().string_value
        )
        imu_topic = self.get_parameter("imu_topic").get_parameter_value().string_value
        
        # Log topic configuration
        self.get_logger().info("=" * 60)
        self.get_logger().info("CV Publisher Topic Configuration:")
        self.get_logger().info(f"  RGB Topic:        {rgb_topic}")
        self.get_logger().info(f"  Depth Topic:      {depth_topic}")
        self.get_logger().info(f"  Camera Info:      {cam_intrinsic_topic}")
        self.get_logger().info(f"  IMU/Pose Topic:   {imu_topic}")
        self.get_logger().info("=" * 60)
        self.rgb_sub = self.create_subscription(Image, rgb_topic, self.rgb_callback, 10)
        self.info_sub = self.create_subscription(CameraInfo, cam_intrinsic_topic, self.info_callback, 10)
        self.depth_sub = self.create_subscription(Image, depth_topic, self.depth_callback, 10)
        self.imu_sub = self.create_subscription(PoseStamped, imu_topic, self.imu_callback, 10)

        
        self.get_logger().info("Using individual callbacks for debugging (synchronizer disabled)")

        # Subscribe to YOLO detections
        self.declare_parameter("yolo_detections_topic", "yolo_detections")
        detections_topic = self.get_parameter("yolo_detections_topic").get_parameter_value().string_value
        self.detections_sub = self.create_subscription(
            DetectionArray, detections_topic, self.detections_callback, 10
        )
        self.get_logger().info(f"  YOLO Detections:  {detections_topic}")

        self.publisher = self.create_publisher(Map, "map_objects", 10)
        self.cv_timer = self.create_timer(0.5, self.ros_map_objects)
        
        self.last_rgb: np.ndarray | None = None
        self.last_depth: np.ndarray | None = None
        self.last_cam_intrinsic: CameraIntrinsics | None = None
        self.last_position: Point3D | None = None
        self.last_rotation: Rotation3D | None = None
        self.last_detections: list[CustomDetection] = []  # Store latest YOLO detections
        self.map_state: MapState = MapState()
        self.bridge = CvBridge()
        self.detection_history = defaultdict(deque)  # Dictionary of deques indexed by track_id
        # 3D bounding box dimensions for each class (width, height, depth in meters)
        # Format: "class_name": (width, height, depth)
        self.box_map = {
            # Custom classes
            "gate": (2.0, 3.0, 0.5),
            "buoy": (0.3, 0.3, 0.3),
            "shark": (2.0, 0.5, 0.3),
            "swordfish": (1.0, 0.2, 0.2),
            
            # Common COCO classes
            "person": (0.6, 1.8, 0.3),
            "bicycle": (1.8, 1.2, 0.4),
            "car": (4.5, 1.8, 1.8),
            "motorcycle": (2.0, 1.2, 0.8),
            "boat": (5.0, 2.0, 1.5),
            "bottle": (0.08, 0.25, 0.08),
            "cup": (0.08, 0.12, 0.08),
            "chair": (0.5, 0.9, 0.5),
            "couch": (2.0, 0.8, 0.9),
            "potted plant": (0.3, 0.5, 0.3),
            "bed": (2.0, 0.6, 1.8),
            "dining table": (1.5, 0.75, 1.0),
            "tv": (1.0, 0.6, 0.1),
            "laptop": (0.35, 0.02, 0.25),
            "cell phone": (0.08, 0.15, 0.01),
            "book": (0.15, 0.22, 0.03),
            "clock": (0.3, 0.3, 0.1),
            "vase": (0.15, 0.3, 0.15),
            "backpack": (0.3, 0.4, 0.2),
            "handbag": (0.3, 0.3, 0.15),
            "suitcase": (0.5, 0.7, 0.25),
        }
        
        # Expected frequencies: maximum number of objects per class to track
        # If more objects are detected, keep only the most frequently updated ones
        self.expected_frequencies = {
            
            # Common objects - adjust based on your use case
            "bottle": 1,
            # Add more classes as needed
        }
        
        self.get_logger().info("CV Publisher initialized - waiting for sensor data and detections...")

    def ros_img_to_cv2(self, msg, encoding="bgr8") -> np.ndarray:
        """
        Convert a ROS sensor_msgs/Image message to an OpenCV numpy array.
        :param msg: ROS Image message
        :param encoding: Desired encoding ("bgr8", "mono8", "mono16", "32FC1")
        :return: OpenCV numpy array
        """
        if encoding not in ["bgr8", "mono8", "mono16", "32FC1"]:
            raise ValueError(f"Unsupported encoding: {encoding}")

        dtype_map = {
            "bgr8": np.uint8,
            "mono8": np.uint8,
            "mono16": np.uint16,
            "32FC1": np.float32,
        }

        dtype = dtype_map[encoding]

        # Calculate expected array size based on encoding and dimensions
        channels = 3 if encoding == "bgr8" else 1
        expected_size = msg.height * msg.width * channels
        actual_size = len(msg.data)

        if actual_size != expected_size:
            # Try to infer correct dimensions based on actual data size
            if encoding == "bgr8" and actual_size % 3 == 0:
                total_pixels = actual_size // 3
                # Try to determine if width is correct but height is wrong
                if total_pixels % msg.width == 0:
                    corrected_height = total_pixels // msg.width
                    img_array = np.frombuffer(msg.data, dtype=dtype).reshape(
                        (corrected_height, msg.width, 3)
                    )
                    return img_array

        # Convert the byte data to a NumPy array
        img_array = np.frombuffer(msg.data, dtype=dtype)

        try:
            # Reshape based on image dimensions and encoding
            if encoding == "bgr8":
                # Use step value if available for proper alignment
                if msg.step > 0 and msg.step >= msg.width * 3:
                    img_array = img_array.reshape((msg.height, msg.width, 3))
                else:
                    img_array = np.reshape(img_array, (msg.height, msg.width, 3))
            else:
                # Single-channel
                img_array = np.reshape(img_array, (msg.height, msg.width))

            return img_array

        except Exception as e:
            raise ValueError(
                f"Reshape failed: {e}. Image info: height={msg.height}, width={msg.width}, step={msg.step}, encoding={msg.encoding}"
            )

    # TEMPORARY DEBUG CALLBACKS - Print message info as they arrive
    def rgb_callback(self, msg: Image):
        """Callback for RGB images"""
        self.get_logger().info(
            f"📷 RGB: {msg.width}x{msg.height}, encoding={msg.encoding}, "
            f"timestamp={msg.header.stamp.sec}.{msg.header.stamp.nanosec:09d}",
            throttle_duration_sec=2.0
        )
        self.last_rgb = self.ros_img_to_cv2(msg)
    
    def info_callback(self, msg: CameraInfo):
        """Callback for camera info"""
        # Extract camera intrinsics from K matrix: [fx, 0, cx, 0, fy, cy, 0, 0, 1]
        fx, fy, cx, cy = msg.k[0], msg.k[4], msg.k[2], msg.k[5]
        self.get_logger().info(
            f"📐 Camera Info: {msg.width}x{msg.height}, "
            f"fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}, "
            f"timestamp={msg.header.stamp.sec}.{msg.header.stamp.nanosec:09d}",
            throttle_duration_sec=2.0
        )
        self.last_cam_intrinsic = (fx, fy, cx, cy)
    
    def depth_callback(self, msg: Image):
        """Callback for depth images"""
        self.get_logger().info(
            f"🌊 Depth: {msg.width}x{msg.height}, encoding={msg.encoding}, "
            f"timestamp={msg.header.stamp.sec}.{msg.header.stamp.nanosec:09d}",
            throttle_duration_sec=2.0
        )
        self.last_depth = self.ros_img_to_cv2(msg)
    
    def imu_callback(self, msg: PoseStamped):
        """Callback for IMU/Pose"""
        self.get_logger().info(
            f"🧭 Pose: pos=({msg.pose.position.x:.2f}, {msg.pose.position.y:.2f}, {msg.pose.position.z:.2f}), "
            f"timestamp={msg.header.stamp.sec}.{msg.header.stamp.nanosec:09d}",
            throttle_duration_sec=2.0
        )
        self.last_position = Point3D(msg.pose.position.x, msg.pose.position.y, msg.pose.position.z)
        self.last_rotation = Rotation3D(
            msg.pose.orientation.x, msg.pose.orientation.y, 
            msg.pose.orientation.z, msg.pose.orientation.w
        )
    
    def detections_callback(self, msg: DetectionArray):
        """Callback for YOLO detections"""
        self.get_logger().info(
            f"🎯 Received {len(msg.detections)} detections from YOLO",
            throttle_duration_sec=2.0
        )
        # Convert ROS Detection messages to custom Detection format for compatibility
        self.last_detections = []
        for det in msg.detections:
            custom_det = CustomDetection(
                _x1=int(det.x_min),
                _y1=int(det.y_min),
                _x2=int(det.x_max),
                _y2=int(det.y_max),
                _cls=det.class_id,
                _conf=det.confidence,
                _distance=None,  # Will be calculated from depth
                _point=None,  # Will be calculated from depth
                _bbox_3d=None  # Will be calculated from depth
            )
            self.last_detections.append(custom_det)
    
    def ros_map_objects(self):
        # Check if we have all required data before processing
        if (
            self.last_rgb is None
            or self.last_cam_intrinsic is None
            or self.last_depth is None
            or self.last_position is None
            or self.last_rotation is None
            or len(self.last_detections) == 0
        ):
            self.get_logger().debug(
                f"Waiting for sensor inputs - "
                f"RGB: {self.last_rgb is not None}, "
                f"Intrinsic: {self.last_cam_intrinsic is not None}, "
                f"Depth: {self.last_depth is not None}, "
                f"Pose: {self.last_position is not None}, "
                f"Detections: {len(self.last_detections)}",
                throttle_duration_sec=5.0
            )
            return

        # COCO class names (80 classes) - must match YOLO publisher
        class_names = [
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
            "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
            "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
            "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
            "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
            "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
            "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
            "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
            "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
        ]
        map_objects(
            self.map_state,
            class_names,
            self.box_map,
            self.last_detections,
            self.last_cam_intrinsic,
            self.last_depth,
            self.last_position,
            self.last_rotation,
            self.detection_history,
            self.expected_frequencies  # Add expected frequencies parameter
        )
        # Create Map message with all detected objects
        map_msg = Map()
        
        # Create empty map bounds (can be calculated if needed)
        map_msg.map_bounds = BoundingBox3D()
        
        # Convert internal MapObject to ROS MapObject messages
        for map_object in self.map_state.objects:
            # Skip objects without 3D bounding boxes (class not in box_map)
            if map_object.bbox_3d is None:
                self.get_logger().warn(
                    f"Skipping object with class {map_object.cls} - no 3D bounding box defined",
                    throttle_duration_sec=5.0
                )
                continue
            
            ros_object = RosMapObject()
            ros_object.cls = map_object.cls
            bbox = BoundingBox3D()
            # Ensure quaternion values are floats for ROS message compatibility
            bbox.center.orientation.x = float(map_object.bbox_3d.rotation.x)
            bbox.center.orientation.y = float(map_object.bbox_3d.rotation.y)
            bbox.center.orientation.z = float(map_object.bbox_3d.rotation.z)
            bbox.center.orientation.w = float(map_object.bbox_3d.rotation.w)
            # Ensure position values are floats
            bbox.center.position.x = float(map_object.point.x)
            bbox.center.position.y = float(map_object.point.y)
            bbox.center.position.z = float(map_object.point.z)
            # Ensure size values are floats
            bbox.size.x = float(map_object.bbox_3d.height)
            bbox.size.y = float(map_object.bbox_3d.width)
            bbox.size.z = float(map_object.bbox_3d.length)
            ros_object.bbox = bbox
            map_msg.objects.append(ros_object)

        # Publish the Map message
        self.publisher.publish(map_msg)
        
        self.get_logger().info(
            f"📍 Published map with {len(map_msg.objects)} objects",
            throttle_duration_sec=2.0
        )


def main(args=None):
    rclpy.init(args=args)
    node = ComputerVisionPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
