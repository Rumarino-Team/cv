from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
import os


def generate_launch_description():
    """
    Launch file for the complete computer vision pipeline.
    
    Starts:
    1. USB camera node (usb_cam)
    2. Depth publisher (DepthAnything V2)
    3. CV publisher (YOLO object detection)
    
    Usage:
        ros2 launch hydrus_cv full_pipeline.launch.py
        
    Or with custom parameters:
        ros2 launch hydrus_cv full_pipeline.launch.py \
            camera_device:=/dev/video2 \
            image_width:=1280 \
            image_height:=720 \
            framerate:=30.0
    """
    
    # Get the absolute path to weights directory
    weights_dir = os.path.join(os.path.expanduser('~'), 'Projects', 'auv', 'weights')
    
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
        default_value='/webcam_rgb',
        description='Topic name for RGB images'
    )
    
    depth_topic_arg = DeclareLaunchArgument(
        'depth_topic',
        default_value='/webcam_depth',
        description='Topic name for depth images'
    )
    
    camera_info_topic_arg = DeclareLaunchArgument(
        'camera_info_topic',
        default_value='/webcam_intrinsics',
        description='Topic name for camera info'
    )
    
    # Model paths
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
            # Remap to our standard topic names
        }],
        remappings=[
            ('/image_raw', LaunchConfiguration('rgb_topic')),
            ('/camera_info', LaunchConfiguration('camera_info_topic')),
        ]
    )
    
    # 2. Depth Publisher Node (DepthAnything V2)
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
    
    # 3. CV Publisher Node (YOLO + Processing)
    cv_publisher_node = Node(
        package='hydrus_cv',
        executable='cv_publisher',
        name='cv_publisher',
        output='screen',
        parameters=[{
            'rgb_topic': LaunchConfiguration('rgb_topic'),
            'depth_topic': '/depth_anything/depth',  # Use depth from depth_publisher
            'camera_info_topic': LaunchConfiguration('camera_info_topic'),
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
        depth_model_path_arg,
        yolo_model_path_arg,
        depth_publish_rate_arg,
        
        # Nodes
        usb_cam_node,
        depth_publisher_node,
        cv_publisher_node,
    ])
