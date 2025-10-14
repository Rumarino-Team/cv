from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
import os


def generate_launch_description():
    """
    Combined launch file for ORB-SLAM3 and HydrusCV pipeline.
    
    This launch file starts:
    1. USB camera node (usb_cam)
    2. ORB-SLAM3 monocular SLAM node
    3. Depth publisher (DepthAnything V2)
    4. CV publisher (YOLO object detection)
    
    The system provides:
    - Visual SLAM and camera localization via ORB-SLAM3
    - Object detection and mapping via HydrusCV
    - Depth estimation for 3D object positioning
    
    Usage:
        ros2 launch hydrus_cv slam_cv_pipeline.launch.py
        
    Or with custom parameters:
        ros2 launch hydrus_cv slam_cv_pipeline.launch.py \
            camera_device:=/dev/video2 \
            image_width:=1280 \
            image_height:=720 \
            use_orb_viewer:=false
    """
    
    # Get absolute paths
    weights_dir = os.path.join(os.path.expanduser('~'), 'Projects', 'auv', 'weights')
    orb_slam_pkg_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'orb_slam')
    orb_slam3_dir = os.path.join(os.path.expanduser('~'), 'Projects', 'ORB_SLAM3')
    
    # ========== Launch Arguments ==========
    
    # Camera arguments
    camera_device_arg = DeclareLaunchArgument(
        'camera_device',
        default_value='/dev/video0',
        description='USB camera device path'
    )
    
    image_width_arg = DeclareLaunchArgument(
        'image_width',
        default_value='640',
        description='Camera image width'
    )
    
    image_height_arg = DeclareLaunchArgument(
        'image_height',
        default_value='480',
        description='Camera image height'
    )
    
    framerate_arg = DeclareLaunchArgument(
        'framerate',
        default_value='30.0',
        description='Camera framerate'
    )
    
    pixel_format_arg = DeclareLaunchArgument(
        'pixel_format',
        default_value='yuyv',
        description='Camera pixel format (yuyv, mjpeg, etc.)'
    )
    
    # Topic names
    rgb_topic_arg = DeclareLaunchArgument(
        'rgb_topic',
        default_value='/camera1/image_raw',
        description='Topic name for RGB images (shared between ORB-SLAM and CV)'
    )
    
    depth_topic_arg = DeclareLaunchArgument(
        'depth_topic',
        default_value='/camera1/depth',
        description='Topic name for depth images'
    )
    
    camera_info_topic_arg = DeclareLaunchArgument(
        'camera_info_topic',
        default_value='/camera1/camera_info',
        description='Topic name for camera info'
    )
    
    # ORB-SLAM3 arguments
    orb_vocabulary_arg = DeclareLaunchArgument(
        'orb_vocabulary',
        default_value=os.path.join(orb_slam3_dir, 'Vocabulary', 'ORBvoc.txt'),
        description='Path to ORB-SLAM3 vocabulary file'
    )
    
    orb_settings_arg = DeclareLaunchArgument(
        'orb_settings',
        default_value=os.path.join(orb_slam_pkg_dir, 'config', 'webcam.yaml'),
        description='Path to ORB-SLAM3 camera settings file'
    )
    
    use_orb_viewer_arg = DeclareLaunchArgument(
        'use_orb_viewer',
        default_value='true',
        description='Enable ORB-SLAM3 viewer'
    )
    
    # CV Model paths
    depth_model_path_arg = DeclareLaunchArgument(
        'depth_model_path',
        default_value=os.path.join(weights_dir, 'dav2_s.pt'),
        description='Path to DepthAnything model weights'
    )
    
    yolo_model_path_arg = DeclareLaunchArgument(
        'yolo_model_path',
        default_value=os.path.join(weights_dir, 'yolov8.pt'),
        description='Path to YOLO model weights'
    )
    
    # Processing rates
    depth_publish_rate_arg = DeclareLaunchArgument(
        'depth_publish_rate',
        default_value='10.0',
        description='Depth estimation rate in Hz'
    )
    
    # ========== Nodes ==========
    
    # 1. USB Camera Node
    usb_cam_node = Node(
        package='usb_cam',
        executable='usb_cam_node_exe',
        name='usb_cam',
        output='screen',
        parameters=[{
            'video_device': LaunchConfiguration('camera_device'),
            'image_width': LaunchConfiguration('image_width'),
            'image_height': LaunchConfiguration('image_height'),
            'framerate': LaunchConfiguration('framerate'),
            'pixel_format': LaunchConfiguration('pixel_format'),
            'camera_frame_id': 'camera',
            'io_method': 'mmap',
        }],
        remappings=[
            ('/image_raw', LaunchConfiguration('rgb_topic')),
            ('/camera_info', LaunchConfiguration('camera_info_topic')),
        ]
    )
    
    # 2. ORB-SLAM3 Monocular Node
    orb_slam3_node = Node(
        package='orb_slam3_ros2',
        executable='mono',
        name='orb_slam3_mono',
        output='screen',
        parameters=[{
            'vocabulary_path': LaunchConfiguration('orb_vocabulary'),
            'settings_path': LaunchConfiguration('orb_settings'),
            'use_viewer': LaunchConfiguration('use_orb_viewer'),
            'use_depth': False,  # Monocular mode
            'image_topic': LaunchConfiguration('rgb_topic'),
            'pose_topic': 'orb_slam3/camera_pose',
            'path_topic': 'orb_slam3/camera_path',
            'world_frame_id': 'world',
            'camera_frame_id': 'camera',
            'queue_size': 10,
            'publish_tf': True,
        }],
        arguments=[
            LaunchConfiguration('orb_vocabulary'),
            LaunchConfiguration('orb_settings'),
        ],
        remappings=[
            ('/camera/image_raw', LaunchConfiguration('rgb_topic')),
        ]
    )
    
    # 3. Depth Publisher Node (DepthAnything V2)
    depth_publisher_node = Node(
        package='hydrus_cv',
        executable='depth_publisher',
        name='depth_publisher',
        output='screen',
        parameters=[{
            'rgb_topic': LaunchConfiguration('rgb_topic'),
            'depth_output_topic': '/depth_anything/depth',
            'depth_model_path': LaunchConfiguration('depth_model_path'),
            'publish_rate': LaunchConfiguration('depth_publish_rate'),
        }]
    )
    
    # 4. CV Publisher Node (YOLO + Processing)
    cv_publisher_node = Node(
        package='hydrus_cv',
        executable='cv_publisher',
        name='cv_publisher',
        output='screen',
        parameters=[{
            'rgb_topic': LaunchConfiguration('rgb_topic'),
            'depth_topic': '/depth_anything/depth',  # Use depth from depth_publisher
            'camera_info_topic': LaunchConfiguration('camera_info_topic'),
            'imu_topic': 'orb_slam3/camera_pose',  # Use pose from ORB-SLAM3
            'yolo_model_path': LaunchConfiguration('yolo_model_path'),
            'depth_model_path': LaunchConfiguration('depth_model_path'),
        }]
    )
    
    # ========== Launch Description ==========
    
    return LaunchDescription([
        # Launch arguments
        camera_device_arg,
        image_width_arg,
        image_height_arg,
        framerate_arg,
        pixel_format_arg,
        rgb_topic_arg,
        depth_topic_arg,
        camera_info_topic_arg,
        orb_vocabulary_arg,
        orb_settings_arg,
        use_orb_viewer_arg,
        depth_model_path_arg,
        yolo_model_path_arg,
        depth_publish_rate_arg,
        
        # Nodes
        usb_cam_node,
        orb_slam3_node,
        depth_publisher_node,
        cv_publisher_node,
    ])