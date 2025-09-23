import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from pyCV import YoloManager

class ComputerVisionPublisher(Node):
    def __init__(self):
        super().__init__('ComputerVisionPublisher')
        self.publisher_ = self.create_publisher(String, 'detections', 10)

    def timer_callback(self):
        msg = String()
        msg.data = f'Hello ROS2: {self.i}'
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing: "{msg.data}"')
        self.i += 1


def main(args=None):
    rclpy.init(args=args)

    node = MinimalPublisher()
    rclpy.spin(node)  # keep node alive until Ctrl+C

    # Cleanup
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

