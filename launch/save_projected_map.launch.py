from launch import LaunchDescription
from launch_ros.actions import Node
import os

def generate_launch_description():
    package_name = 'fast_lio'
    executable_path = os.path.join(
        '/workspace/src/onix/install', package_name, 'lib', package_name, 'save_projected_map_node.py'
    )

    return LaunchDescription([
        Node(
            package=package_name,
            executable=executable_path,
            name='save_projected_map_node',
            output='screen',
            parameters=[
                {"save_directory": "/workspace/saved_data"},
                {"save_interval_sec": 30.0}
            ]
        )
    ])
