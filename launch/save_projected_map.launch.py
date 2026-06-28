from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument, ExecuteProcess

import os

def generate_launch_description():
    save_dir = LaunchConfiguration('save_dir')
    script_path = LaunchConfiguration('script_path')

    return LaunchDescription([
        DeclareLaunchArgument('save_dir', default_value='/workspace/saved_data'),
        DeclareLaunchArgument('script_path', default_value='/media/data/cheekybot_ws/src/FAST_LIO/save_projected_map_node.py'),

        ExecuteProcess(
            cmd=['python3', script_path],
            name='save_projected_map_node',
            output='screen',
            additional_env={'SAVE_DIR': save_dir}
        ),
    ])
