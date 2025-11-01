#!/usr/bin/env python3
"""
YOLO Detection Publisher Node

This node subscribes to RGB images, runs YOLO inference, and publishes:
1. DetectionArray messages with all detected objects
2. Annotated images with bounding boxes drawn

This separates object detection from the mapping/tracking logic.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from interfaces.msg import Detection, DetectionArray
from cv_bridge import CvBridge
import numpy as np
import cv2
import os
import sys

# Add thirdparty path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from hydrus_cv.detection_core import YOLOModelManager


class YOLOPublisher(Node):
    def __init__(self):
        super().__init__("yolo_publisher")

        # Declare parameters
        self.declare_parameter("rgb_topic", "/camera1/image_raw")
        self.declare_parameter("yolo_model_path", "../../weights/yolov8.pt")
        self.declare_parameter("detections_topic", "yolo_detections")
        self.declare_parameter("annotated_image_topic", "annotated_image")
        self.declare_parameter("confidence_threshold", 0.25)
        self.declare_parameter("publish_annotated", True)

        # Get parameters
        rgb_topic = self.get_parameter("rgb_topic").get_parameter_value().string_value
        yolo_model_path = self.get_parameter("yolo_model_path").get_parameter_value().string_value
        detections_topic = self.get_parameter("detections_topic").get_parameter_value().string_value
        annotated_topic = self.get_parameter("annotated_image_topic").get_parameter_value().string_value
        self.confidence_threshold = self.get_parameter("confidence_threshold").get_parameter_value().double_value
        self.publish_annotated = self.get_parameter("publish_annotated").get_parameter_value().bool_value

        # Log configuration
        self.get_logger().info("=" * 60)
        self.get_logger().info("YOLO Publisher Configuration:")
        self.get_logger().info(f"  RGB Topic:           {rgb_topic}")
        self.get_logger().info(f"  Model Path:          {yolo_model_path}")
        self.get_logger().info(f"  Detections Topic:    {detections_topic}")
        self.get_logger().info(f"  Annotated Topic:     {annotated_topic}")
        self.get_logger().info(f"  Confidence Thresh:   {self.confidence_threshold}")
        self.get_logger().info(f"  Publish Annotated:   {self.publish_annotated}")
        self.get_logger().info("=" * 60)

        # Initialize YOLO model
        self.yolo_manager = YOLOModelManager(yolo_model_path)
        self.bridge = CvBridge()

        # COCO class names (80 classes) - YOLO models use standard COCO dataset indices
        # Full list to ensure class IDs match correctly (class 39 = bottle, class 0 = person, etc.)
        self.class_names = [
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

        # Colors for visualization (BGR format)
        self.colors = {
            0: (0, 0, 255),      # gate - Red
            1: (0, 255, 255),    # buoy - Yellow
            2: (255, 0, 255),    # shark - Magenta
            3: (255, 255, 0),    # swordfish - Cyan
            4: (0, 255, 0),      # person - Green
            5: (255, 0, 0),      # bottle - Blue
            6: (128, 0, 128),    # boat - Purple
        }

        # Subscribers and publishers
        self.rgb_sub = self.create_subscription(
            Image, rgb_topic, self.image_callback, 10
        )
        self.detections_pub = self.create_publisher(
            DetectionArray, detections_topic, 10
        )
        if self.publish_annotated:
            self.annotated_pub = self.create_publisher(
                Image, annotated_topic, 10
            )

        self.get_logger().info("YOLO Publisher initialized and ready!")

    def image_callback(self, msg: Image):
        """Process incoming RGB image and publish detections"""
        try:
            # Convert ROS image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            
            # Run YOLO inference
            results = self.yolo_manager.model(cv_image, verbose=False)
            
            # Create DetectionArray message
            detection_array = DetectionArray()
            detection_array.header = msg.header
            detection_array.image_width = msg.width
            detection_array.image_height = msg.height
            detection_array.model_name = "YOLOv8"
            detection_array.model_version = "1.0"
            
            # Process each detection
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    confidence = float(box.conf[0].cpu().numpy())
                    
                    # Filter by confidence threshold
                    if confidence < self.confidence_threshold:
                        continue
                    
                    # Extract box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    # Create Detection message
                    detection = Detection()
                    detection.class_id = class_id
                    detection.class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
                    detection.confidence = confidence
                    detection.x_min = float(x1)
                    detection.y_min = float(y1)
                    detection.x_max = float(x2)
                    detection.y_max = float(y2)
                    detection.center_x = float((x1 + x2) / 2)
                    detection.center_y = float((y1 + y2) / 2)
                    detection.width = float(x2 - x1)
                    detection.height = float(y2 - y1)
                    
                    detection_array.detections.append(detection)
            
            # Publish detections
            self.detections_pub.publish(detection_array)
            
            # Publish annotated image if enabled
            if self.publish_annotated:
                annotated_img = self._draw_detections(cv_image, detection_array.detections)
                annotated_msg = self.bridge.cv2_to_imgmsg(annotated_img, encoding="bgr8")
                annotated_msg.header = msg.header
                self.annotated_pub.publish(annotated_msg)
            
            # Log detection count periodically
            if len(detection_array.detections) > 0:
                self.get_logger().info(
                    f"🎯 Detected {len(detection_array.detections)} objects",
                    throttle_duration_sec=2.0
                )
        
        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")

    def _draw_detections(self, image: np.ndarray, detections: list) -> np.ndarray:
        """Draw bounding boxes and labels on image"""
        annotated = image.copy()
        
        for detection in detections:
            # Get coordinates
            x1 = int(detection.x_min)
            y1 = int(detection.y_min)
            x2 = int(detection.x_max)
            y2 = int(detection.y_max)
            
            # Get color for this class
            color = self.colors.get(detection.class_id, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Create label
            label = f"{detection.class_name}: {detection.confidence:.2f}"
            
            # Get text size for background
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            
            # Draw background rectangle for text
            cv2.rectangle(
                annotated,
                (x1, y1 - text_height - baseline - 5),
                (x1 + text_width, y1),
                color,
                -1,  # Filled
            )
            
            # Draw text
            cv2.putText(
                annotated,
                label,
                (x1, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),  # Black text
                1,
                cv2.LINE_AA,
            )
        
        return annotated


def main(args=None):
    rclpy.init(args=args)
    node = YOLOPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
