import rclpy
import cv2
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, Imu
from geometry_msgs.msg import PoseStamped, Quaternion, TransformStamped
from cv_bridge import CvBridge
from tf2_ros import TransformBroadcaster
import numpy as np
import os
import pandas as pd
from scipy.spatial.transform import Rotation


class CameraWrapper(Node):
    def __init__(self):
        super().__init__("CameraWrapperNode")

        # Declare parameters
        self.declare_parameter("source_type", "camera")  # 'camera' or 'video'
        self.declare_parameter("camera_index", 0)  # Camera device index
        self.declare_parameter("video_path", "")  # Path to MP4 file
        self.declare_parameter("imu_data_folder", "")  # Path to folder with IMU CSV files
        self.declare_parameter("enable_imu", True)  # Enable/disable IMU publishing
        self.declare_parameter("loop_video", True)  # Loop video when it ends
        self.declare_parameter("rgb_topic", "/webcam_rgb")
        self.declare_parameter("camera_info_topic", "/webcam_intrinsics")
        self.declare_parameter("pose_topic", "/camera_pose")
        self.declare_parameter("imu_topic", "/imu")
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
        self.imu_data_folder = (
            self.get_parameter("imu_data_folder").get_parameter_value().string_value
        )
        self.enable_imu = (
            self.get_parameter("enable_imu").get_parameter_value().bool_value
        )
        self.loop_video = (
            self.get_parameter("loop_video").get_parameter_value().bool_value
        )
        rgb_topic = self.get_parameter("rgb_topic").get_parameter_value().string_value
        camera_info_topic = (
            self.get_parameter("camera_info_topic").get_parameter_value().string_value
        )
        pose_topic = self.get_parameter("pose_topic").get_parameter_value().string_value
        imu_topic = self.get_parameter("imu_topic").get_parameter_value().string_value
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
        self.pose_publisher = self.create_publisher(PoseStamped, pose_topic, 10)
        self.imu_publisher = self.create_publisher(Imu, imu_topic, 10)

        # Initialize TF broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)

        # Initialize CV Bridge
        self.bridge = CvBridge()

        # Initialize IMU data
        self.imu_data_loaded = False
        self.accel_data = None
        self.gyro_data = None
        self.mag_data = None
        self.video_start_time = None
        self.current_orientation = np.array([1.0, 0.0, 0.0, 0.0])  # quaternion [w, x, y, z]
        
        # Load IMU data if enabled and folder is provided
        if self.enable_imu and self.imu_data_folder and os.path.exists(self.imu_data_folder):
            self._load_imu_data()
        elif self.enable_imu and self.imu_data_folder and not os.path.exists(self.imu_data_folder):
            self.get_logger().warn(f"IMU enabled but folder not found: {self.imu_data_folder}")
        elif not self.enable_imu:
            self.get_logger().info("IMU publishing disabled by configuration")
        
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
                    # Reset orientation when looping
                    self.current_orientation = np.array([1.0, 0.0, 0.0, 0.0])
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
            # Get current timestamp
            current_time = self.get_clock().now()
            
            img_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            img_msg.header.stamp = current_time.to_msg()
            img_msg.header.frame_id = "camera_frame"

            # Publish image
            self.rgb_publisher.publish(img_msg)

            # Create and publish camera info
            camera_info_msg = self._create_camera_info(frame.shape[1], frame.shape[0])
            camera_info_msg.header.stamp = img_msg.header.stamp
            camera_info_msg.header.frame_id = img_msg.header.frame_id
            self.camera_info_publisher.publish(camera_info_msg)
            
            # Publish IMU and Pose data if enabled and available
            if self.enable_imu and self.imu_data_loaded and self.source_type == "video":
                # Calculate timestamp in nanoseconds from video start
                current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                video_fps = self.cap.get(cv2.CAP_PROP_FPS)
                
                if video_fps > 0:
                    # Calculate video timestamp in nanoseconds
                    video_time_ns = self.video_start_time + int((current_frame / video_fps) * 1e9)
                    self._publish_pose_and_imu(video_time_ns, img_msg.header.stamp)

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

    def _load_imu_data(self):
        """Load IMU data from CSV files in the specified folder"""
        try:
            # Find CSV files in the folder
            files = os.listdir(self.imu_data_folder)
            accel_file = None
            gyro_file = None
            mag_file = None
            
            for file in files:
                if file.endswith("accel.csv"):
                    accel_file = os.path.join(self.imu_data_folder, file)
                elif file.endswith("gyro.csv"):
                    gyro_file = os.path.join(self.imu_data_folder, file)
                elif file.endswith("magnetic.csv"):
                    mag_file = os.path.join(self.imu_data_folder, file)
            
            if accel_file and gyro_file:
                # Load accelerometer data
                self.accel_data = pd.read_csv(
                    accel_file, 
                    header=None, 
                    names=['accel_x', 'accel_y', 'accel_z', 'timestamp']
                )
                
                # Load gyroscope data
                self.gyro_data = pd.read_csv(
                    gyro_file,
                    header=None,
                    names=['gyro_x', 'gyro_y', 'gyro_z', 'timestamp']
                )
                
                # Load magnetometer data if available
                if mag_file:
                    self.mag_data = pd.read_csv(
                        mag_file,
                        header=None,
                        names=['mag_x', 'mag_y', 'mag_z', 'timestamp']
                    )
                
                # Get the video start time from the first IMU timestamp
                self.video_start_time = self.accel_data['timestamp'].iloc[0]
                
                self.imu_data_loaded = True
                self.get_logger().info(
                    f"Loaded IMU data: {len(self.accel_data)} accel samples, "
                    f"{len(self.gyro_data)} gyro samples"
                )
                if self.mag_data is not None:
                    self.get_logger().info(f"Loaded {len(self.mag_data)} magnetometer samples")
            else:
                self.get_logger().warn(
                    f"Could not find required IMU files in {self.imu_data_folder}"
                )
        except Exception as e:
            self.get_logger().error(f"Error loading IMU data: {str(e)}")

    def _get_imu_at_timestamp(self, timestamp_ns):
        """Get interpolated IMU data at a specific timestamp"""
        if not self.imu_data_loaded:
            return None, None, None
        
        try:
            # Find closest accelerometer reading
            accel_idx = (self.accel_data['timestamp'] - timestamp_ns).abs().idxmin()
            accel = self.accel_data.iloc[accel_idx]
            
            # Find closest gyroscope reading
            gyro_idx = (self.gyro_data['timestamp'] - timestamp_ns).abs().idxmin()
            gyro = self.gyro_data.iloc[gyro_idx]
            
            # Find closest magnetometer reading if available
            mag = None
            if self.mag_data is not None:
                mag_idx = (self.mag_data['timestamp'] - timestamp_ns).abs().idxmin()
                mag = self.mag_data.iloc[mag_idx]
            
            return accel, gyro, mag
        except Exception as e:
            self.get_logger().error(f"Error getting IMU data: {str(e)}")
            return None, None, None

    def _integrate_orientation(self, gyro, dt):
        """Integrate gyroscope data to update orientation using quaternions"""
        if gyro is None or dt <= 0:
            return
        
        # Get angular velocities (rad/s)
        omega_x = gyro['gyro_x']
        omega_y = gyro['gyro_y']
        omega_z = gyro['gyro_z']
        
        # Create rotation vector
        angle = np.sqrt(omega_x**2 + omega_y**2 + omega_z**2) * dt
        
        if angle < 1e-10:  # Avoid division by zero
            return
        
        # Axis-angle to quaternion
        axis = np.array([omega_x, omega_y, omega_z]) / np.sqrt(omega_x**2 + omega_y**2 + omega_z**2)
        half_angle = angle / 2.0
        sin_half = np.sin(half_angle)
        
        delta_q = np.array([
            np.cos(half_angle),
            axis[0] * sin_half,
            axis[1] * sin_half,
            axis[2] * sin_half
        ])
        
        # Multiply quaternions: q_new = q_current * delta_q
        w1, x1, y1, z1 = self.current_orientation
        w2, x2, y2, z2 = delta_q
        
        self.current_orientation = np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])
        
        # Normalize quaternion
        norm = np.linalg.norm(self.current_orientation)
        if norm > 0:
            self.current_orientation /= norm

    def _publish_pose_and_imu(self, timestamp_ns, frame_timestamp):
        """Publish pose and IMU data for the current frame"""
        if not self.imu_data_loaded:
            return
        
        accel, gyro, mag = self._get_imu_at_timestamp(timestamp_ns)
        
        if accel is None or gyro is None:
            return
        
        # Calculate dt from frame rate
        dt = 1.0 / self.frame_rate
        
        # Update orientation using gyroscope integration
        self._integrate_orientation(gyro, dt)
        
        # Publish IMU message
        imu_msg = Imu()
        imu_msg.header.stamp = frame_timestamp
        imu_msg.header.frame_id = "imu_frame"
        
        # Linear acceleration
        imu_msg.linear_acceleration.x = float(accel['accel_x'])
        imu_msg.linear_acceleration.y = float(accel['accel_y'])
        imu_msg.linear_acceleration.z = float(accel['accel_z'])
        
        # Angular velocity
        imu_msg.angular_velocity.x = float(gyro['gyro_x'])
        imu_msg.angular_velocity.y = float(gyro['gyro_y'])
        imu_msg.angular_velocity.z = float(gyro['gyro_z'])
        
        # Orientation from integration
        imu_msg.orientation.w = float(self.current_orientation[0])
        imu_msg.orientation.x = float(self.current_orientation[1])
        imu_msg.orientation.y = float(self.current_orientation[2])
        imu_msg.orientation.z = float(self.current_orientation[3])
        
        self.imu_publisher.publish(imu_msg)
        
        # Publish Pose message (orientation only, no position from IMU)
        pose_msg = PoseStamped()
        pose_msg.header.stamp = frame_timestamp
        pose_msg.header.frame_id = "camera_frame"
        
        # Position at origin (IMU cannot reliably provide position)
        pose_msg.pose.position.x = 0.0
        pose_msg.pose.position.y = 0.0
        pose_msg.pose.position.z = 0.0
        
        # Orientation from IMU integration
        pose_msg.pose.orientation.w = float(self.current_orientation[0])
        pose_msg.pose.orientation.x = float(self.current_orientation[1])
        pose_msg.pose.orientation.y = float(self.current_orientation[2])
        pose_msg.pose.orientation.z = float(self.current_orientation[3])
        
        self.pose_publisher.publish(pose_msg)
        
        # Publish TF transform
        t = TransformStamped()
        t.header.stamp = frame_timestamp
        t.header.frame_id = "world"  # Parent frame
        t.child_frame_id = "camera_frame"  # Child frame
        
        # Position at origin (IMU cannot provide position)
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.0
        
        # Orientation from IMU
        t.transform.rotation.w = float(self.current_orientation[0])
        t.transform.rotation.x = float(self.current_orientation[1])
        t.transform.rotation.y = float(self.current_orientation[2])
        t.transform.rotation.z = float(self.current_orientation[3])
        
        self.tf_broadcaster.sendTransform(t)
        
        # Log IMU information (every 30 frames to avoid spam)
        current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        if current_frame % 30 == 0:
            self.get_logger().info(
                f"IMU [frame {current_frame}] - "
                f"Accel: ({accel['accel_x']:.3f}, {accel['accel_y']:.3f}, {accel['accel_z']:.3f}) m/s² | "
                f"Gyro: ({gyro['gyro_x']:.3f}, {gyro['gyro_y']:.3f}, {gyro['gyro_z']:.3f}) rad/s | "
                f"Orientation: w={self.current_orientation[0]:.3f}, x={self.current_orientation[1]:.3f}, "
                f"y={self.current_orientation[2]:.3f}, z={self.current_orientation[3]:.3f}"
            )

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
