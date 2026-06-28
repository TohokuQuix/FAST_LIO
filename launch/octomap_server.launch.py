from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='octomap_server',
            executable='octomap_server_node',
            name='octomap_server',
            output='screen',
            parameters=[
                {
                    'frame_id': 'initial_base_link', #'map',
                    'resolution': 0.1,  # adjust to your desired voxel size
                    'sensor_model/max_range': 5.0,
                    'sensor_model/hit': 0.7,
                    'sensor_model/miss': 0.4,
                    'sensor_model/min': 0.12,
                    'occupancy_min_z': 0.1,   # <<-- start slice from 5 cm
                    'occupancy_max_z': 1.0,   # <<-- end slice at 1 meter
                }
            ],
            remappings=[
                ('cloud_in', '/Laser_map'),
                ('tf', '/tf'),
                ('tf_static', '/tf_static'),
                ('odom', '/Odometry')
            ]
        )
    ])
