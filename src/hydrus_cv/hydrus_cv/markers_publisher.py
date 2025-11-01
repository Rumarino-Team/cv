#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from vision_msgs.msg import BoundingBox3D
from geometry_msgs.msg import Pose
from interfaces.msg import Map


def setColor(ros_color: ColorRGBA, r: int, g: int, b: int, a: int):
    # Convert from 0-255 range to 0.0-1.0 range for ROS compatibility
    ros_color.r = float(r) / 255.0
    ros_color.g = float(g) / 255.0
    ros_color.b = float(b) / 255.0
    ros_color.a = float(a) / 255.0


class DetectionVisualizer(Node):
    def __init__(self):
        super().__init__("detection_viz")

        # Declare parameters
        self.declare_parameter("enabled", True)
        self.declare_parameter("map_topic", "map_objects")
        self.declare_parameter("frame_id", "world")  # Must match ORB-SLAM world frame

        # Get parameters
        self.enabled = self.get_parameter("enabled").get_parameter_value().bool_value
        map_topic = self.get_parameter("map_topic").get_parameter_value().string_value
        self.frame_id = (
            self.get_parameter("frame_id").get_parameter_value().string_value
        )

        # Check if visualization is enabled
        if not self.enabled:
            self.get_logger().info(
                "Marker visualization is DISABLED. Shutting down node."
            )
            # Shutdown the node gracefully
            self.destroy_node()
            rclpy.shutdown()
            return

        # Define colors for different classes
        self.colors = [ColorRGBA() for _ in range(4)]
        setColor(self.colors[0], 255, 0, 0, 100)  # Red
        setColor(self.colors[1], 255, 255, 0, 100)  # Yellow
        setColor(self.colors[2], 255, 0, 255, 100)  # Magenta
        setColor(self.colors[3], 0, 255, 255, 100)  # Cyan

        # Create subscriber and publisher
        self.sub = self.create_subscription(Map, map_topic, self.callback, 10)
        self.pub = self.create_publisher(MarkerArray, "detection_markers", 10)

        self.get_logger().info(f"DetectionVisualizer initialized")
        self.get_logger().info(f"  Subscribing to: {map_topic}")
        self.get_logger().info(f"  Publishing markers with frame_id: {self.frame_id}")

    def callback(self, msg: Map):
        """Process Map message containing multiple MapObjects"""
        arr = MarkerArray()
        
        # Create a marker for each object in the map
        for idx, map_object in enumerate(msg.objects):
            m = Marker()
            m.header.frame_id = self.frame_id
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = "bbox"
            m.id = idx
            m.type = Marker.CUBE
            m.action = Marker.ADD
            
            # Use the actual 3D position from the map object
            m.pose = map_object.bbox.center
            
            # Ensure scale values are not zero (minimum 0.1m for visibility)
            m.scale.x = max(float(map_object.bbox.size.x), 0.1)
            m.scale.y = max(float(map_object.bbox.size.y), 0.1)
            m.scale.z = max(float(map_object.bbox.size.z), 0.1)
            
            # Set color based on object class
            m.color = self.colors[map_object.cls % len(self.colors)]
            
            # Set lifetime to 0 for persistent markers (they stay until updated/deleted)
            m.lifetime.sec = 0
            m.lifetime.nanosec = 0
            
            arr.markers.append(m)
        
        # If there are no objects, publish an empty marker array to clear old markers
        if len(msg.objects) == 0:
            # Add a DELETE_ALL marker to clear previous markers
            m = Marker()
            m.action = Marker.DELETEALL
            arr.markers.append(m)
        
        self.pub.publish(arr)
        
        # Log marker publication for debugging
        if len(msg.objects) > 0:
            # Log first marker position for debugging
            if len(arr.markers) > 0:
                first_marker = arr.markers[0]
                self.get_logger().info(
                    f"📦 Published {len(arr.markers)} markers | "
                    f"First marker pos: ({first_marker.pose.position.x:.2f}, "
                    f"{first_marker.pose.position.y:.2f}, {first_marker.pose.position.z:.2f}) | "
                    f"scale: ({first_marker.scale.x:.2f}, {first_marker.scale.y:.2f}, {first_marker.scale.z:.2f})",
                    throttle_duration_sec=5.0
                )


def main():
    rclpy.init()
    node = DetectionVisualizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
