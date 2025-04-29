#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import MarkerArray
import numpy as np
import pandas as pd
from pyntcloud import PyntCloud
import os
import time
from datetime import datetime
import warnings

class MapSaver(Node):
    def __init__(self):
        super().__init__('save_projected_map_node')

        self.save_2d_map = False   # <== Turn ON or OFF saving 2D map
        self.save_3d_map = True    # <== Turn ON or OFF saving 3D map

        if self.save_2d_map:
            self.projected_map_sub = self.create_subscription(
                OccupancyGrid,
                '/projected_map',
                self.projected_map_callback,
                10)

        if self.save_3d_map:
            self.occupied_cells_sub = self.create_subscription(
                MarkerArray,
                '/occupied_cells_vis_array',
                self.occupied_cells_callback,
                10)

        self.projected_map = None
        self.occupied_cells = None

        self.save_dir = '/workspace/saved_data'
        os.makedirs(self.save_dir, exist_ok=True)

        self.timer = self.create_timer(30.0, self.save_maps)

    def projected_map_callback(self, msg):
        self.projected_map = msg

    def occupied_cells_callback(self, msg):
        self.occupied_cells = msg

    def save_maps(self):
        now_str = datetime.now().strftime("%Y%m%d_%H%M%S")

        warnings.filterwarnings("ignore", category=FutureWarning)

        if self.save_2d_map and self.projected_map:
            file_path = os.path.join(self.save_dir, f'map_{now_str}.pgm')
            self.save_pgm(file_path, self.projected_map)
            os.chmod(file_path, 0o777)
            
            yaml_path = os.path.join(self.save_dir, f'map_{now_str}.yaml')
            self.save_yaml(yaml_path, file_path, self.projected_map)
            os.chmod(yaml_path, 0o777)
            
            self.get_logger().info(f"Saved 2D map to {file_path} and {yaml_path}")

        if self.save_3d_map and self.occupied_cells:
            ply_path = os.path.join(self.save_dir, f'occupied_cells_{now_str}.ply')
            self.save_occupied_cells_ply(ply_path, self.occupied_cells)
            os.chmod(ply_path, 0o777)
            self.get_logger().info(f"Saved 3D occupied cells to {ply_path}")

    def save_pgm(self, filename, map_msg):
        width = map_msg.info.width
        height = map_msg.info.height
        data = np.array(map_msg.data, dtype=np.int8).reshape((height, width))
        img = np.full((height, width), 205, dtype=np.uint8)
        img[data == 0] = 254
        img[data == 100] = 0

        with open(filename, 'wb') as f:
            f.write(b'P5\n')
            f.write(f"{width} {height}\n255\n".encode())
            f.write(img.tobytes())

    def save_yaml(self, yaml_path, pgm_filename, map_msg):
        yaml_content = f"""image: {os.path.basename(pgm_filename)}
resolution: {map_msg.info.resolution}
origin: [{map_msg.info.origin.position.x}, {map_msg.info.origin.position.y}, {map_msg.info.origin.position.z}]
negate: 0
occupied_thresh: 0.65
free_thresh: 0.196
"""
        with open(yaml_path, 'w') as f:
            f.write(yaml_content)


    def save_occupied_cells_ply(self, filename, marker_array_msg):
        points = []

        for marker in marker_array_msg.markers:
            for p in marker.points:
                points.append([p.x, p.y, p.z])

        if len(points) == 0:
            self.get_logger().warn("No points in occupied_cells!")
            return

        df = pd.DataFrame(points, columns=["x", "y", "z"])
        cloud = PyntCloud(df)
        cloud.to_file(filename, as_text=True)

def main(args=None):
    rclpy.init(args=args)
    node = MapSaver()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
