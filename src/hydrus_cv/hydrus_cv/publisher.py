import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from interfaces.msg import Map, MapObject
from vision_msgs.msg import BoundingBox3D
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Point, Pose 
from message_filters import Subscriber, ApproximateTimeSynchronizer
from ...pyCV.detection_core import map_objects 
from ...pyCV.custom_types import CameraIntrinsics, MapObject, MapState, Point3D, Rotation3D
import numpy as np
class ComputerVisionPublisher(Node):
    def __init__(self):
        super().__init__('ComputerVisionPublisher')


        self.declare_parameter('rgb_topic', '/webcam_rgb')
        self.declare_parameter("depth_topic", "/webcam_depth")
        self.declare_parameter("camera_info_topic", "/webcam_intrinsics")
        self.declare_parameter("imu_topic", "/imu")
        rgb_topic = self.get_parameter('rgb_topic').get_parameter_value().string_value
        depth_topic = self.get_parameter("depth_topic").get_parameter_value().string_value
        cam_intrinsic_topic = self.get_parameter("camera_info_topic").get_parameter_value().string_value
        imu_topic = self.get_parameter("imu_topic").get_parameter_value().string_value

        self.rgb_sub = Subscriber(self, Image, rgb_topic)
        self.info_sub = Subscriber(self, CameraInfo, cam_intrinsic_topic)
        self.depth_sub = Subscriber(self, Image, depth_topic)
        self.imu_sub = Subscriber(self, Pose, imu_topic)
        self.ts = ApproximateTimeSynchronizer(
            [self.rgb_sub, self.info_sub, self.depth_sub, self.imu_sub],
            queue_size=10,
            slop=0.1
                                                )
        self.ts.registerCallback(self.sync_callback)


        self.publisher = self.create_publisher( MapObject, 'detections', 10)
        self.cv_timer = self.create_timer(0.5, self.timer_callback)


        self.last_rgb : np.ndarray | None = None
        self.last_depth: np.ndarray | None = None
        self.last_cam_intrinsic : CameraIntrinsics | None = None
        self.last_position:  Point3D | None = None
        self.last_rotation: Rotation3D | None = None

        self.map_state = MapState()

    

    def sync_inputs(self,rgb : Image,depth : Image, cam_intrinsic: CameraInfo, imu: Pose):
        self.last_rgb = rgb
        self.last_depth = depth
        self.last_cam_intrinsic = cam_intrinsic # TODO 
        self.last_position = Point3D(imu.position.x, imu.position.y, imu.position.z)
        self.last_rotation = Rotation3D(imu.orientation.x, imu.orientation.y, imu.orientation.z, imu.orientation.w)

    def ros_map_objects(self):
        map_objects(self.map_state,self.class_name, self.box_map, self.yoloManager,self.last_rgb, 
                    self.last_cam_intrinsic, self.last_depth, self.last_position, self.last_rotation)
        
        MapObject()
        map = Map()
        for 
        self.publisher.publish()

def main(args=None):
    rclpy.init(args=args)

    node = ComputerVisionPublisher()
    rclpy.spin(node) 
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

