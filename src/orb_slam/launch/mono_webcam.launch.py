from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
import os

def generate_launch_description():
    # Get the package directory
    pkg_dir = os.path.dirname(os.path.dirname(__file__))
    orb_slam3_dir = os.path.join(pkg_dir, '..', '..')
    
    # Declare launch arguments
    vocabulary_arg = DeclareLaunchArgument(
        'vocabulary',
        default_value=os.path.join(orb_slam3_dir, 'Vocabulary', 'ORBvoc.txt'),
        description='Path to ORB vocabulary file'
    )
    
    settings_arg = DeclareLaunchArgument(
        'settings',
        default_value=os.path.join(pkg_dir, 'config', 'webcam.yaml'),
        description='Path to ORB-SLAM3 camera settings file'
    )
    
    params_arg = DeclareLaunchArgument(
        'params',
        default_value=os.path.join(pkg_dir, 'config', 'params.yaml'),
        description='Path to ROS2 parameters file'
    )
    

    # ORB-SLAM3 Monocular node
    orb_slam3_node = Node(
        package='orb_slam3_ros2',
        executable='mono',
        name='orb_slam3_mono',
        parameters=[LaunchConfiguration('params')],
        arguments=[
            LaunchConfiguration('vocabulary'),
            LaunchConfiguration('settings'),
        ],
        output='screen'
    )

    return LaunchDescription([
        vocabulary_arg,
        settings_arg,
        params_arg,
        orb_slam3_node
    ])
