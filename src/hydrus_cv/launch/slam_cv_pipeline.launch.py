from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.conditions import IfCondition
from launch_ros.substitutions import FindPackageShare
import os
import yaml


def load_yaml_config(config_path):
    """Load parameters from YAML config file if it exists"""
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return {}


def generate_launch_description():
    """
    Combined launch file for ORB-SLAM3 and HydrusCV pipeline.
    
    This launch file starts:
    1. Camera/Video source node (camera_ros) - supports USB cameras and video files
    2. ORB-SLAM3 monocular SLAM node
    3. Depth publisher (DepthAnything V2)
    4. CV publisher (YOLO object detection)
    5. Marker visualization (RViz markers for detections)
    
    The system provides:
    - Visual SLAM and camera localization via ORB-SLAM3
    - Object detection and mapping via HydrusCV
    - Depth estimation for 3D object positioning
    - RViz visualization of detected objects
    
    Usage:
        # Use default parameters (camera)
        ros2 launch hydrus_cv slam_cv_pipeline.launch.py
        
        # Use USB camera with specific device
        ros2 launch hydrus_cv slam_cv_pipeline.launch.py \
            source_type:=camera \
            camera_device:=0
        
        # Use video file (with looping)
        ros2 launch hydrus_cv slam_cv_pipeline.launch.py \
            source_type:=video \
            video_path:=/path/to/video.mp4
        
        # Use video file (without looping - stops when video ends)
        ros2 launch hydrus_cv slam_cv_pipeline.launch.py \
            source_type:=video \
            video_path:=/path/to/video.mp4 \
            loop_video:=false
        
        # Use config file
        ros2 launch hydrus_cv slam_cv_pipeline.launch.py \
            config_file:=/path/to/slam_cv_pipeline.yaml
        
        # Use config file and override specific parameters
        ros2 launch hydrus_cv slam_cv_pipeline.launch.py \
            config_file:=slam_cv_pipeline.yaml \
            camera_device:=0 \
            image_width:=1280 \
            image_height:=720 \
            use_orb_viewer:=false
    """

    # Get absolute paths
    pkg_share = FindPackageShare("hydrus_cv").find("hydrus_cv")
    default_config_path = os.path.join(pkg_share, "config", "slam_cv_pipeline.yaml")
    weights_dir = os.path.join(os.path.expanduser("~"), "Projects", "auv", "weights")
    orb_slam_pkg_dir = os.path.join(
        os.path.expanduser("~"), "Projects", "auv", "src", "orb_slam"
    )
    orb_slam3_dir = os.path.join(os.path.expanduser("~"), "Projects", "ORB_SLAM3")

    # Load config file if exists
    config_file_arg = DeclareLaunchArgument(
        "config_file",
        default_value="",
        description="Path to YAML config file. If not provided, uses command-line arguments.",
    )

    # Try to load default config
    config = {}
    if os.path.exists(default_config_path):
        config = load_yaml_config(default_config_path)

    # Helper function to get config value with fallback
    def get_config(key_path, default):
        """Get value from nested config dict using dot notation"""
        keys = key_path.split(".")
        value = config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value if value is not None else default

    # ========== Launch Arguments ==========

    # Camera/Video source arguments
    source_type_arg = DeclareLaunchArgument(
        "source_type",
        default_value=str(get_config("camera.source_type", "camera")),
        description='Source type: "camera" or "video"',
    )

    camera_device_arg = DeclareLaunchArgument(
        "camera_device",
        default_value=str(get_config("camera.device", "0")),
        description='Camera device index (0, 1, 2, etc.) used when source_type is "camera"',
    )

    video_path_arg = DeclareLaunchArgument(
        "video_path",
        default_value=str(get_config("camera.video_path", "")),
        description='Path to video file (used when source_type is "video")',
    )

    loop_video_arg = DeclareLaunchArgument(
        "loop_video",
        default_value=str(get_config("camera.loop_video", True)).lower(),
        description="Loop video when it ends (only applies to video files)",
    )

    image_width_arg = DeclareLaunchArgument(
        "image_width",
        default_value=str(get_config("camera.image_width", 640)),
        description="Camera image width",
    )

    image_height_arg = DeclareLaunchArgument(
        "image_height",
        default_value=str(get_config("camera.image_height", 480)),
        description="Camera image height",
    )

    framerate_arg = DeclareLaunchArgument(
        "framerate",
        default_value=str(get_config("camera.framerate", 30.0)),
        description="Camera framerate",
    )

    pixel_format_arg = DeclareLaunchArgument(
        "pixel_format",
        default_value=str(get_config("camera.pixel_format", "yuyv")),
        description="Camera pixel format (yuyv, mjpeg, etc.)",
    )

    # Topic names
    rgb_topic_arg = DeclareLaunchArgument(
        "rgb_topic",
        default_value=str(get_config("topics.rgb_topic", "/camera1/image_raw")),
        description="Topic name for RGB images (shared between ORB-SLAM and CV)",
    )

    depth_topic_arg = DeclareLaunchArgument(
        "depth_topic",
        default_value=str(get_config("topics.depth_topic", "/depth_anything/depth")),
        description="Topic name for depth images consumed by ORB-SLAM and CV pipeline",
    )

    camera_info_topic_arg = DeclareLaunchArgument(
        "camera_info_topic",
        default_value=str(
            get_config("topics.camera_info_topic", "/camera1/camera_info")
        ),
        description="Topic name for camera info",
    )

    # ORB-SLAM3 arguments
    orb_vocabulary_arg = DeclareLaunchArgument(
        "orb_vocabulary",
        default_value=os.path.expanduser(
            str(
                get_config(
                    "orb_slam3.vocabulary_path",
                    os.path.join(orb_slam3_dir, "Vocabulary", "ORBvoc.txt"),
                )
            )
        ),
        description="Path to ORB-SLAM3 vocabulary file",
    )

    orb_settings_arg = DeclareLaunchArgument(
        "orb_settings",
        default_value=os.path.expanduser(
            str(
                get_config(
                    "orb_slam3.settings_path",
                    os.path.join(orb_slam_pkg_dir, "config", "webcam.yaml"),
                )
            )
        ),
        description="Path to ORB-SLAM3 camera settings file",
    )

    use_orb_viewer_arg = DeclareLaunchArgument(
        "use_orb_viewer",
        default_value=str(get_config("orb_slam3.use_viewer", True)).lower(),
        description="Enable ORB-SLAM3 viewer",
    )

    orb_use_depth_arg = DeclareLaunchArgument(
        "orb_use_depth",
        default_value=str(get_config("orb_slam3.use_depth", True)).lower(),
        description="Enable RGB-D mode in ORB-SLAM3",
    )

    orb_pose_topic_arg = DeclareLaunchArgument(
        "orb_pose_topic",
        default_value=str(get_config("topics.orb_pose_topic", "/orb_slam3/camera_pose")),
        description="Pose topic from ORB-SLAM3",
    )

    orb_path_topic_arg = DeclareLaunchArgument(
        "orb_path_topic",
        default_value=str(get_config("topics.orb_path_topic", "orb_slam3/camera_path")),
        description="Topic name for ORB-SLAM3 path output",
    )

    orb_world_frame_arg = DeclareLaunchArgument(
        "orb_world_frame",
        default_value=str(get_config("orb_slam3.world_frame_id", "world")),
        description="TF frame id for ORB-SLAM3 world frame",
    )

    orb_camera_frame_arg = DeclareLaunchArgument(
        "orb_camera_frame",
        default_value=str(get_config("orb_slam3.camera_frame_id", "camera")),
        description="TF frame id for ORB-SLAM3 camera frame",
    )

    orb_queue_size_arg = DeclareLaunchArgument(
        "orb_queue_size",
        default_value=str(get_config("orb_slam3.queue_size", 10)),
        description="Queue size for ORB-SLAM3 subscriptions and publishers",
    )

    orb_publish_tf_arg = DeclareLaunchArgument(
        "orb_publish_tf",
        default_value=str(get_config("orb_slam3.publish_tf", True)).lower(),
        description="Publish TF transforms for ORB-SLAM3 pose",
    )

    # CV Model paths
    depth_model_path_arg = DeclareLaunchArgument(
        "depth_model_path",
        default_value=os.path.expanduser(
            str(
                get_config(
                    "depth_estimation.model_path",
                    os.path.join(weights_dir, "dav2_s.pt"),
                )
            )
        ),
        description="Path to DepthAnything model weights",
    )

    yolo_model_path_arg = DeclareLaunchArgument(
        "yolo_model_path",
        default_value=os.path.expanduser(
            str(
                get_config(
                    "computer_vision.yolo_model_path",
                    os.path.join(weights_dir, "yolov8.pt"),
                )
            )
        ),
        description="Path to YOLO model weights",
    )

    # Processing rates
    depth_publish_rate_arg = DeclareLaunchArgument(
        "depth_publish_rate",
        default_value=str(get_config("depth_estimation.publish_rate", 10.0)),
        description="Depth estimation rate in Hz",
    )

    depth_enabled_arg = DeclareLaunchArgument(
        "depth_enabled",
        default_value=str(get_config("depth_estimation.enabled", True)).lower(),
        description="Enable/disable depth estimation processing",
    )

    # Computer Vision arguments
    publish_annotated_image_arg = DeclareLaunchArgument(
        "publish_annotated_image",
        default_value=str(
            get_config("computer_vision.publish_annotated_image", True)
        ).lower(),
        description="Enable/disable publishing of annotated images with YOLO detections",
    )

    annotated_image_topic_arg = DeclareLaunchArgument(
        "annotated_image_topic",
        default_value=str(
            get_config("computer_vision.annotated_image_topic", "annotated_image")
        ),
        description="Topic for publishing annotated images",
    )

    # Visualization arguments
    visualization_enabled_arg = DeclareLaunchArgument(
        "visualization_enabled",
        default_value=str(get_config("visualization.enabled", True)).lower(),
        description="Enable/disable RViz marker visualization",
    )

    visualization_map_topic_arg = DeclareLaunchArgument(
        "visualization_map_topic",
        default_value=str(get_config("visualization.map_topic", "map_objects")),
        description="Input topic with detection/map data",
    )

    visualization_marker_topic_arg = DeclareLaunchArgument(
        "visualization_marker_topic",
        default_value=str(
            get_config("visualization.marker_topic", "detection_markers")
        ),
        description="Output topic for RViz markers",
    )

    visualization_frame_id_arg = DeclareLaunchArgument(
        "visualization_frame_id",
        default_value=str(get_config("visualization.frame_id", "world")),
        description="TF frame ID for markers (should match ORB-SLAM world_frame_id)",
    )

    # ========== Nodes ==========

    # 1. Camera/Video Source Node (camera_ros)
    camera_ros_node = Node(
        package="hydrus_cv",
        executable="camera_ros",
        name="camera_wrapper_node",
        output="screen",
        parameters=[
            {
                "source_type": LaunchConfiguration("source_type"),
                "camera_index": ParameterValue(
                    LaunchConfiguration("camera_device"), value_type=int
                ),
                "video_path": LaunchConfiguration("video_path"),
                "loop_video": ParameterValue(
                    LaunchConfiguration("loop_video"), value_type=bool
                ),
                "rgb_topic": LaunchConfiguration("rgb_topic"),
                "camera_info_topic": LaunchConfiguration("camera_info_topic"),
                "frame_rate": ParameterValue(
                    LaunchConfiguration("framerate"), value_type=float
                ),
                "width": ParameterValue(
                    LaunchConfiguration("image_width"), value_type=int
                ),
                "height": ParameterValue(
                    LaunchConfiguration("image_height"), value_type=int
                ),
            }
        ],
    )

    # 2. ORB-SLAM3 Monocular Node
    orb_slam3_node = Node(
        package="orb_slam3_ros2",
        executable="mono",
        name="orb_slam3_mono",
        output="screen",
        parameters=[
            {
                "vocabulary_path": LaunchConfiguration("orb_vocabulary"),
                "settings_path": LaunchConfiguration("orb_settings"),
                "use_viewer": ParameterValue(
                    LaunchConfiguration("use_orb_viewer"), value_type=bool
                ),
                "use_depth": ParameterValue(
                    LaunchConfiguration("orb_use_depth"), value_type=bool
                ),
                "image_topic": LaunchConfiguration("rgb_topic"),
                "depth_topic": LaunchConfiguration("depth_topic"),
                "pose_topic": LaunchConfiguration("orb_pose_topic"),
                "path_topic": LaunchConfiguration("orb_path_topic"),
                "world_frame_id": LaunchConfiguration("orb_world_frame"),
                "camera_frame_id": LaunchConfiguration("orb_camera_frame"),
                "queue_size": ParameterValue(
                    LaunchConfiguration("orb_queue_size"), value_type=int
                ),
                "publish_tf": ParameterValue(
                    LaunchConfiguration("orb_publish_tf"), value_type=bool
                ),
            }
        ],
        remappings=[
            ("/camera/image_raw", LaunchConfiguration("rgb_topic")),
            ("/camera/depth/image_raw", LaunchConfiguration("depth_topic")),
        ],
    )

    # 3. Depth Publisher Node (DepthAnything V2) - Only run if depth is enabled in ORB-SLAM
    depth_publisher_node = Node(
        package="hydrus_cv",
        executable="depth_publisher",
        name="depth_publisher",
        output="screen",
        condition=IfCondition(LaunchConfiguration("depth_enabled")),
        parameters=[
            {
                "enabled": ParameterValue(
                    LaunchConfiguration("depth_enabled"), value_type=bool
                ),
                "rgb_topic": LaunchConfiguration("rgb_topic"),
                "depth_output_topic": LaunchConfiguration("depth_topic"),
                "depth_model_path": LaunchConfiguration("depth_model_path"),
                "publish_rate": ParameterValue(
                    LaunchConfiguration("depth_publish_rate"), value_type=float
                ),
            }
        ],
    )

    # 4. YOLO Publisher Node (Object Detection)
    yolo_publisher_node = Node(
        package="hydrus_cv",
        executable="yolo_publisher",
        name="yolo_publisher",
        output="screen",
        parameters=[
            {
                "rgb_topic": LaunchConfiguration("rgb_topic"),
                "yolo_model_path": LaunchConfiguration("yolo_model_path"),
                "detections_topic": "yolo_detections",
                "annotated_image_topic": LaunchConfiguration("annotated_image_topic"),
                "confidence_threshold": 0.25,
                "publish_annotated": LaunchConfiguration("publish_annotated_image"),
            }
        ],
    )

    # 5. CV Publisher Node (3D Mapping + Tracking)
    cv_publisher_node = Node(
        package="hydrus_cv",
        executable="cv_publisher",
        name="cv_publisher",
        output="screen",
        parameters=[
            {
                "rgb_topic": LaunchConfiguration("rgb_topic"),
                "depth_topic": LaunchConfiguration("depth_topic"),
                "camera_info_topic": LaunchConfiguration("camera_info_topic"),
                "imu_topic": LaunchConfiguration("orb_pose_topic"),
                "yolo_detections_topic": "yolo_detections",
            }
        ],
    )

    # 5. Marker Visualization Node (RViz markers)
    markers_publisher_node = Node(
        package="hydrus_cv",
        executable="map_visualizer",
        name="detection_viz",
        output="screen",
        parameters=[
            {
                "enabled": ParameterValue(
                    LaunchConfiguration("visualization_enabled"), value_type=bool
                ),
                "map_topic": LaunchConfiguration("visualization_map_topic"),
                "frame_id": LaunchConfiguration("visualization_frame_id"),
            }
        ],
        remappings=[
            ("detection_markers", LaunchConfiguration("visualization_marker_topic")),
        ],
    )

    return LaunchDescription(
        [
            # Launch arguments
            config_file_arg,
            source_type_arg,
            camera_device_arg,
            video_path_arg,
            loop_video_arg,
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
            orb_use_depth_arg,
            orb_pose_topic_arg,
            orb_path_topic_arg,
            orb_world_frame_arg,
            orb_camera_frame_arg,
            orb_queue_size_arg,
            orb_publish_tf_arg,
            depth_model_path_arg,
            yolo_model_path_arg,
            depth_publish_rate_arg,
            depth_enabled_arg,
            publish_annotated_image_arg,
            annotated_image_topic_arg,
            visualization_enabled_arg,
            visualization_map_topic_arg,
            visualization_marker_topic_arg,
            visualization_frame_id_arg,
            camera_ros_node,
            depth_publisher_node,
            # 2. Delay ORB-SLAM3 to allow depth publisher to initialize
            TimerAction(
                period=float(get_config("orb_slam3.startup_delay", 5.0)),
                actions=[orb_slam3_node],
            ),
            # 3. Start YOLO publisher after SLAM is ready
            TimerAction(
                period=float(get_config("yolo.startup_delay", 6.0)),
                actions=[yolo_publisher_node],
            ),
            # 4. Start CV publisher after YOLO and ORB-SLAM are ready
            TimerAction(
                period=float(get_config("computer_vision.startup_delay", 7.0)),
                actions=[cv_publisher_node],
            ),
            # 5. Start marker visualization after CV publisher
            TimerAction(
                period=float(get_config("visualization.startup_delay", 7.5)),
                actions=[markers_publisher_node],
            ),
        ]
    )
