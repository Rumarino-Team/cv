import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import numpy as np
import glob
import os
import sys
import cv2


from .detection_core import DepthAnythingManager


class DepthPublisher(Node):
    """
    ROS2 Node that subscribes to RGB images and publishes depth maps
    using the DepthAnything V2 model.
    """

    def __init__(self):
        super().__init__("depth_publisher")

        # Declare parameters
        self.declare_parameter("enabled", True)
        self.declare_parameter("rgb_topic", "/webcam_rgb")
        self.declare_parameter("depth_output_topic", "/depth_anything/depth")
        self.declare_parameter("depth_visualization_topic", "/depth_anything/depth_viz")
        
        # Use absolute path to weights directory - using dav2_s.pt (small model) which matches the 'vits' encoder
        default_weights_path = os.path.join(
            os.path.expanduser("~"), "Projects", "auv", "weights", "dav2_s.pt"
        )
        self.declare_parameter("depth_model_path", default_weights_path)
        self.declare_parameter("publish_rate", 10.0)  # Hz
        
        # Depth normalization parameters
        self.declare_parameter("depth_min", 0.3)  # Minimum depth in meters
        self.declare_parameter("depth_max", 10.0)  # Maximum depth in meters
        self.declare_parameter("use_adaptive_range", False)  # Auto-adjust min/max per frame
        self.declare_parameter("publish_visualization", True)  # Publish colored depth map

        # Get parameters
        self.enabled = self.get_parameter("enabled").get_parameter_value().bool_value
        rgb_topic = self.get_parameter("rgb_topic").get_parameter_value().string_value
        depth_output_topic = (
            self.get_parameter("depth_output_topic").get_parameter_value().string_value
        )
        depth_viz_topic = (
            self.get_parameter("depth_visualization_topic").get_parameter_value().string_value
        )
        depth_model_path = (
            self.get_parameter("depth_model_path").get_parameter_value().string_value
        )
        publish_rate = (
            self.get_parameter("publish_rate").get_parameter_value().double_value
        )
        
        # Depth range parameters
        self.depth_min = self.get_parameter("depth_min").get_parameter_value().double_value
        self.depth_max = self.get_parameter("depth_max").get_parameter_value().double_value
        self.use_adaptive_range = self.get_parameter("use_adaptive_range").get_parameter_value().bool_value
        self.publish_visualization = self.get_parameter("publish_visualization").get_parameter_value().bool_value


        # Initialize CV Bridge for ROS-OpenCV conversion
        self.bridge = CvBridge()

        # Initialize DepthAnything model
        self.get_logger().info(f"Loading DepthAnything model from: {depth_model_path}")

        if not os.path.exists(depth_model_path):
            # Try to find available depth models in the project weights directory
            weights_dir = os.path.join(
                os.path.expanduser("~"), "Projects", "auv", "weights"
            )
            depth_anything_paths = glob.glob(os.path.join(weights_dir, "dav*.pt"))
            exception_message = f"The path {depth_model_path} was not found.\n"
            exception_message += "Available depth models:\n"
            for idx, path in enumerate(depth_anything_paths):
                exception_message += f"{idx}) {path}\n"
            raise Exception(exception_message)

        self.depth_model = DepthAnythingManager(depth_model_path)
        self.get_logger().info("DepthAnything model loaded successfully")

        # Subscribe to RGB images
        self.rgb_sub = self.create_subscription(Image, rgb_topic, self.rgb_callback, 10)

        # Publisher for depth images (raw 32-bit float)
        self.depth_pub = self.create_publisher(Image, depth_output_topic, 10)
        
        # Publisher for visualization (8-bit or colored)
        if self.publish_visualization:
            self.depth_viz_pub = self.create_publisher(Image, depth_viz_topic, 10)

        # Store last received RGB image
        self.last_rgb_image = None
        self.last_rgb_header = None

        # Create timer for processing and publishing
        timer_period = 1.0 / publish_rate
        self.timer = self.create_timer(timer_period, self.process_and_publish)

        self.get_logger().info(f"DepthPublisher initialized")
        self.get_logger().info(f"  Subscribing to: {rgb_topic}")
        self.get_logger().info(f"  Publishing depth to: {depth_output_topic}")
        if self.publish_visualization:
            self.get_logger().info(f"  Publishing visualization to: {depth_viz_topic}")
        self.get_logger().info(f"  Publish rate: {publish_rate} Hz")
        self.get_logger().info(f"  Depth range: {self.depth_min}m - {self.depth_max}m")
        self.get_logger().info(f"  Adaptive range: {self.use_adaptive_range}")

    def rgb_callback(self, msg: Image):
        """
        Callback for RGB image messages.
        Stores the latest image for processing.
        """
        try:
            # Convert ROS Image to OpenCV format
            self.last_rgb_image = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding="bgr8"
            )
            self.last_rgb_header = msg.header
        except Exception as e:
            self.get_logger().error(f"Error converting RGB image: {e}")

    def normalize_depth(self, depth_map: np.ndarray) -> np.ndarray:
        """
        Normalize depth map to metric range [depth_min, depth_max].
        
        DepthAnything outputs inverse depth (closer objects = higher values).
        We need to:
        1. Invert it to get actual depth
        2. Scale to metric range
        
        :param depth_map: Raw depth output from model
        :return: Normalized depth in meters
        """
        # Handle edge cases
        if depth_map is None or depth_map.size == 0:
            return None
            
        # Get min/max values
        if self.use_adaptive_range:
            # Use percentiles to avoid outliers
            vmin = np.percentile(depth_map, 2)
            vmax = np.percentile(depth_map, 98)
        else:
            vmin = depth_map.min()
            vmax = depth_map.max()
        
        # Avoid division by zero
        if vmax - vmin < 1e-6:
            return np.full_like(depth_map, self.depth_min, dtype=np.float32)
        
        # Normalize to [0, 1]
        depth_normalized = (depth_map - vmin) / (vmax - vmin)
        
        # IMPORTANT: DepthAnything outputs inverse depth
        # Invert: closer objects should have smaller depth values
        depth_normalized = 1.0 - depth_normalized
        
        # Scale to metric range [depth_min, depth_max]
        depth_metric = depth_normalized * (self.depth_max - self.depth_min) + self.depth_min
        
        return depth_metric.astype(np.float32)

    def create_depth_visualization(self, depth_map: np.ndarray) -> np.ndarray:
        """
        Create a colored visualization of the depth map.
        
        :param depth_map: Depth map in meters
        :return: BGR image for visualization
        """
        # Clip to expected range
        depth_clipped = np.clip(depth_map, self.depth_min, self.depth_max)
        
        # Normalize to [0, 255]
        depth_norm = ((depth_clipped - self.depth_min) / (self.depth_max - self.depth_min) * 255).astype(np.uint8)
        
        # Apply colormap (closer = red/warm, farther = blue/cool)
        depth_colored = cv2.applyColorMap(depth_norm, cv2.COLORMAP_TURBO)
        
        return depth_colored

    def process_and_publish(self):
        """
        Process the latest RGB image and publish depth map.
        """
        if self.last_rgb_image is None:
            self.get_logger().warn(
                "No RGB image received yet", throttle_duration_sec=5.0
            )
            return

        try:
            # Run depth estimation
            raw_depth = self.depth_model.detect(self.last_rgb_image)

            if raw_depth is None or raw_depth.size == 0:
                self.get_logger().error("Depth detection returned invalid result")
                return

            # Normalize to metric depth
            depth_metric = self.normalize_depth(raw_depth)
            
            if depth_metric is None:
                self.get_logger().error("Depth normalization failed")
                return

            # Create and publish raw depth message (32-bit float for accuracy)
            depth_msg = self.bridge.cv2_to_imgmsg(depth_metric, encoding="32FC1")
            
            # Use the same header as the RGB image for proper synchronization
            if self.last_rgb_header is not None:
                depth_msg.header = self.last_rgb_header

            # Publish raw depth
            self.depth_pub.publish(depth_msg)

            # Publish visualization if enabled
            if self.publish_visualization:
                depth_viz = self.create_depth_visualization(depth_metric)
                viz_msg = self.bridge.cv2_to_imgmsg(depth_viz, encoding="bgr8")
                viz_msg.header = depth_msg.header
                self.depth_viz_pub.publish(viz_msg)

            self.get_logger().debug(
                f"Published depth map - shape: {depth_metric.shape}, "
                f"range: [{np.min(depth_metric):.3f}, {np.max(depth_metric):.3f}] meters"
            )

        except Exception as e:
            self.get_logger().error(f"Error processing depth: {e}")


def main(args=None):
    rclpy.init(args=args)

    try:
        node = DepthPublisher()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error in depth publisher: {e}")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
