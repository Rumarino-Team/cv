import rclpy
import cv2
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import numpy as np
import os


class CameraWrapper(Node):
    def __init__(self):
        super().__init__("CameraWrapperNode")

        # Declare parameters
        self.declare_parameter("source_type", "camera")  # 'camera' or 'video'
        self.declare_parameter("camera_index", 0)  # Camera device index
        self.declare_parameter("video_path", "")  # Path to MP4 file
        self.declare_parameter("loop_video", True)  # Loop video when it ends
        self.declare_parameter("rgb_topic", "/webcam_rgb")
        self.declare_parameter("camera_info_topic", "/webcam_intrinsics")
        self.declare_parameter("frame_rate", 30.0)  # Publishing rate in Hz
        self.declare_parameter("width", 640)
        self.declare_parameter("height", 480)

        # Get parameters
        self.source_type = (
            self.get_parameter("source_type").get_parameter_value().string_value
        )
        self.camera_index = (
            self.get_parameter("camera_index").get_parameter_value().integer_value
        )
        self.video_path = (
            self.get_parameter("video_path").get_parameter_value().string_value
        )
        self.loop_video = (
            self.get_parameter("loop_video").get_parameter_value().bool_value
        )
        rgb_topic = self.get_parameter("rgb_topic").get_parameter_value().string_value
        camera_info_topic = (
            self.get_parameter("camera_info_topic").get_parameter_value().string_value
        )
        self.frame_rate = (
            self.get_parameter("frame_rate").get_parameter_value().double_value
        )
        self.width = self.get_parameter("width").get_parameter_value().integer_value
        self.height = self.get_parameter("height").get_parameter_value().integer_value

        # Create publishers
        self.rgb_publisher = self.create_publisher(Image, rgb_topic, 10)
        self.camera_info_publisher = self.create_publisher(
            CameraInfo, camera_info_topic, 10
        )

        # Initialize CV Bridge
        self.bridge = CvBridge()

        # Initialize video capture
        self.cap = None
        self._initialize_capture()

        if self.cap is None or not self.cap.isOpened():
            self.get_logger().error("Failed to open video source")
            return

        # Create timer for publishing frames
        timer_period = 1.0 / self.frame_rate
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.get_logger().info(f"Video2ROS node started with {self.source_type} source")
        if self.source_type == "camera":
            self.get_logger().info(f"Using camera index: {self.camera_index}")
        else:
            self.get_logger().info(f"Using video file: {self.video_path}")

    def _initialize_capture(self):
        """Initialize video capture based on source type"""
        try:
            if self.source_type == "camera":
                self.cap = cv2.VideoCapture(self.camera_index)
                if self.cap.isOpened():
                    # Set camera resolution
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                    # Get actual resolution
                    actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    self.get_logger().info(
                        f"Camera resolution: {actual_width}x{actual_height}"
                    )

            elif self.source_type == "video":
                if not os.path.exists(self.video_path):
                    self.get_logger().error(f"Video file not found: {self.video_path}")
                    return

                self.cap = cv2.VideoCapture(self.video_path)
                if self.cap.isOpened():
                    video_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    video_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    video_fps = self.cap.get(cv2.CAP_PROP_FPS)
                    frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    self.get_logger().info(
                        f"Video resolution: {video_width}x{video_height}"
                    )
                    self.get_logger().info(
                        f"Video FPS: {video_fps}, Total frames: {frame_count}"
                    )

            else:
                self.get_logger().error(
                    f'Invalid source_type: {self.source_type}. Use "camera" or "video"'
                )

        except Exception as e:
            self.get_logger().error(f"Error initializing capture: {str(e)}")

    def timer_callback(self):
        """Read and publish frames"""
        if self.cap is None or not self.cap.isOpened():
            self.get_logger().warn("Video capture is not opened")
            return

        ret, frame = self.cap.read()

        if not ret:
            if self.source_type == "video":
                if self.loop_video:
                    self.get_logger().info("Video ended, looping back to start")
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self.cap.read()
                    if not ret:
                        self.get_logger().error("Failed to read frame after reset")
                        return
                else:
                    self.get_logger().info(
                        "Video ended. Looping disabled, stopping node."
                    )
                    self.destroy_node()
                    rclpy.shutdown()
                    return
            else:
                self.get_logger().error("Failed to read frame from camera")
                return

        try:
            img_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            img_msg.header.stamp = self.get_clock().now().to_msg()
            img_msg.header.frame_id = "camera_frame"

            # Publish image
            self.rgb_publisher.publish(img_msg)

            # Create and publish camera info
            camera_info_msg = self._create_camera_info(frame.shape[1], frame.shape[0])
            camera_info_msg.header.stamp = img_msg.header.stamp
            camera_info_msg.header.frame_id = img_msg.header.frame_id
            self.camera_info_publisher.publish(camera_info_msg)

        except Exception as e:
            self.get_logger().error(f"Error publishing frame: {str(e)}")

    def _create_camera_info(self, width, height):
        """Create a CameraInfo message with basic intrinsics"""
        camera_info = CameraInfo()
        camera_info.width = width
        camera_info.height = height

        # Estimate focal length (rough approximation)
        focal_length = width * 0.8
        cx = width / 2.0
        cy = height / 2.0

        # Camera matrix K
        camera_info.k = [focal_length, 0.0, cx, 0.0, focal_length, cy, 0.0, 0.0, 1.0]

        # Distortion coefficients (assuming no distortion)
        camera_info.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        camera_info.distortion_model = "plumb_bob"

        # Rectification matrix (identity)
        camera_info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]

        # Projection matrix
        camera_info.p = [
            focal_length,
            0.0,
            cx,
            0.0,
            0.0,
            focal_length,
            cy,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
        ]

        return camera_info

    def destroy_node(self):
        """Cleanup when node is destroyed"""
        if self.cap is not None:
            self.cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)

    try:
        node = CameraWrapper()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
