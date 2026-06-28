#!/usr/bin/env python3

import os
from datetime import datetime
import warnings

import numpy as np
import pandas as pd
import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from visualization_msgs.msg import MarkerArray
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from pyntcloud import PyntCloud


class MapSaver(Node):
    def __init__(self):
        super().__init__('save_projected_map_node')

        # ------------------------------------------------------------
        # Parameters
        # ------------------------------------------------------------
        self.declare_parameter('save_2d_map', False)
        self.declare_parameter('save_3d_map', True)

        # 3D map source mode:
        #   pointcloud2     : subscribe PointCloud2 directly, default /Laser_map
        #   occupied_cells  : keep original behavior, subscribe MarkerArray
        self.declare_parameter('map_3d_source', 'pointcloud2')
        self.declare_parameter('pointcloud_topic', '/Laser_map')
        self.declare_parameter('occupied_cells_topic', '/occupied_cells_vis_array')
        self.declare_parameter('projected_map_topic', '/projected_map')

        self.declare_parameter('save_dir', os.environ.get('SAVE_DIR', '/workspace/saved_data'))
        self.declare_parameter('save_period_sec', 30.0)
        self.declare_parameter('file_prefix', 'RoboCup2026_Quix_Mapping_00')

        # If true, NaN/Inf points are removed when saving PointCloud2.
        self.declare_parameter('skip_invalid_points', True)

        # Global component filter for removing clearly separated outlier clusters.
        # This is safer than a local density filter. It first groups points into
        # connected components on a coarse voxel grid, then removes only tiny
        # components that are separated from the main point cloud.
        self.declare_parameter('enable_component_outlier_filter', True)
        self.declare_parameter('component_voxel_size_m', 0.50)
        self.declare_parameter('component_min_points', 10)
        self.declare_parameter('component_keep_ratio_to_largest', 0.02)
        self.declare_parameter('component_filter_max_removed_ratio', 0.25)
        self.declare_parameter('component_filter_apply_to_occupied_cells', False)

        # Optional local isolated-point filter. It is off by default because the
        # component filter above is usually enough for flying points.
        self.declare_parameter('enable_isolated_point_filter', False)
        self.declare_parameter('isolated_neighbor_radius_m', 0.30)
        self.declare_parameter('isolated_min_neighbors', 1)
        self.declare_parameter('isolated_filter_max_removed_ratio', 0.10)
        self.declare_parameter('isolated_filter_apply_to_occupied_cells', False)

        self.save_2d_map = self.get_parameter('save_2d_map').value
        self.save_3d_map = self.get_parameter('save_3d_map').value
        self.map_3d_source = self.get_parameter('map_3d_source').value
        self.pointcloud_topic = self.get_parameter('pointcloud_topic').value
        self.occupied_cells_topic = self.get_parameter('occupied_cells_topic').value
        self.projected_map_topic = self.get_parameter('projected_map_topic').value
        self.save_dir = self.get_parameter('save_dir').value
        self.save_period_sec = float(self.get_parameter('save_period_sec').value)
        self.file_prefix = self.get_parameter('file_prefix').value
        self.skip_invalid_points = self.get_parameter('skip_invalid_points').value
        self.enable_component_outlier_filter = self.get_parameter('enable_component_outlier_filter').value
        self.component_voxel_size_m = float(self.get_parameter('component_voxel_size_m').value)
        self.component_min_points = int(self.get_parameter('component_min_points').value)
        self.component_keep_ratio_to_largest = float(
            self.get_parameter('component_keep_ratio_to_largest').value)
        self.component_filter_max_removed_ratio = float(
            self.get_parameter('component_filter_max_removed_ratio').value)
        self.component_filter_apply_to_occupied_cells = self.get_parameter(
            'component_filter_apply_to_occupied_cells').value
        self.enable_isolated_point_filter = self.get_parameter('enable_isolated_point_filter').value
        self.isolated_neighbor_radius_m = float(self.get_parameter('isolated_neighbor_radius_m').value)
        self.isolated_min_neighbors = int(self.get_parameter('isolated_min_neighbors').value)
        self.isolated_filter_max_removed_ratio = float(
            self.get_parameter('isolated_filter_max_removed_ratio').value)
        self.isolated_filter_apply_to_occupied_cells = self.get_parameter(
            'isolated_filter_apply_to_occupied_cells').value

        if self.component_voxel_size_m <= 0.0:
            raise ValueError(
                "Parameter 'component_voxel_size_m' must be positive, "
                f"but got: {self.component_voxel_size_m}")
        if self.component_min_points < 1:
            raise ValueError(
                "Parameter 'component_min_points' must be >= 1, "
                f"but got: {self.component_min_points}")
        if not (0.0 <= self.component_keep_ratio_to_largest <= 1.0):
            raise ValueError(
                "Parameter 'component_keep_ratio_to_largest' must be in [0, 1], "
                f"but got: {self.component_keep_ratio_to_largest}")
        if not (0.0 < self.component_filter_max_removed_ratio <= 1.0):
            raise ValueError(
                "Parameter 'component_filter_max_removed_ratio' must be in (0, 1], "
                f"but got: {self.component_filter_max_removed_ratio}")

        if self.isolated_neighbor_radius_m <= 0.0:
            raise ValueError(
                "Parameter 'isolated_neighbor_radius_m' must be positive, "
                f"but got: {self.isolated_neighbor_radius_m}")
        if self.isolated_min_neighbors < 1:
            raise ValueError(
                "Parameter 'isolated_min_neighbors' must be >= 1, "
                f"but got: {self.isolated_min_neighbors}")
        if not (0.0 < self.isolated_filter_max_removed_ratio <= 1.0):
            raise ValueError(
                "Parameter 'isolated_filter_max_removed_ratio' must be in (0, 1], "
                f"but got: {self.isolated_filter_max_removed_ratio}")

        if self.map_3d_source not in ('pointcloud2', 'occupied_cells'):
            raise ValueError(
                "Parameter 'map_3d_source' must be 'pointcloud2' or 'occupied_cells', "
                f"but got: {self.map_3d_source}"
            )

        os.makedirs(self.save_dir, exist_ok=True)

        self.projected_map = None
        self.occupied_cells = None
        self.pointcloud = None

        # ------------------------------------------------------------
        # Subscribers
        # ------------------------------------------------------------
        if self.save_2d_map:
            self.projected_map_sub = self.create_subscription(
                OccupancyGrid,
                self.projected_map_topic,
                self.projected_map_callback,
                10)
            self.get_logger().info(
                f"2D map saving enabled. Subscribing: {self.projected_map_topic}")

        if self.save_3d_map:
            if self.map_3d_source == 'pointcloud2':
                self.pointcloud_sub = self.create_subscription(
                    PointCloud2,
                    self.pointcloud_topic,
                    self.pointcloud_callback,
                    10)
                self.get_logger().info(
                    f"3D map source: PointCloud2. Subscribing: {self.pointcloud_topic}")
            else:
                self.occupied_cells_sub = self.create_subscription(
                    MarkerArray,
                    self.occupied_cells_topic,
                    self.occupied_cells_callback,
                    10)
                self.get_logger().info(
                    f"3D map source: occupied cells MarkerArray. Subscribing: {self.occupied_cells_topic}")

        self.timer = self.create_timer(self.save_period_sec, self.save_maps)
        self.get_logger().info(
            f"Saving to: {self.save_dir}, period: {self.save_period_sec:.1f} sec")
        self.get_logger().info(
            "Component outlier filter: "
            f"enabled={self.enable_component_outlier_filter}, "
            f"voxel_size={self.component_voxel_size_m:.3f} m, "
            f"min_points={self.component_min_points}, "
            f"keep_ratio_to_largest={self.component_keep_ratio_to_largest:.3f}, "
            f"max_removed_ratio={self.component_filter_max_removed_ratio:.2f}, "
            f"apply_to_occupied_cells={self.component_filter_apply_to_occupied_cells}")
        self.get_logger().info(
            "Local isolated-point filter: "
            f"enabled={self.enable_isolated_point_filter}, "
            f"neighbor_radius={self.isolated_neighbor_radius_m:.3f} m, "
            f"min_neighbors={self.isolated_min_neighbors}, "
            f"max_removed_ratio={self.isolated_filter_max_removed_ratio:.2f}, "
            f"apply_to_occupied_cells={self.isolated_filter_apply_to_occupied_cells}")

    def projected_map_callback(self, msg):
        self.projected_map = msg

    def occupied_cells_callback(self, msg):
        self.occupied_cells = msg

    def pointcloud_callback(self, msg):
        self.pointcloud = msg

    def save_maps(self):
        now_str = datetime.now().strftime("%H-%M-%S")
        warnings.filterwarnings("ignore", category=FutureWarning)

        if self.save_2d_map:
            if self.projected_map is None:
                self.get_logger().warn("2D map saving is enabled, but no OccupancyGrid has been received yet.")
            else:
                file_path = os.path.join(self.save_dir, f'map_{now_str}.pgm')
                self.save_pgm(file_path, self.projected_map)
                os.chmod(file_path, 0o777)

                yaml_path = os.path.join(self.save_dir, f'map_{now_str}.yaml')
                self.save_yaml(yaml_path, file_path, self.projected_map)
                os.chmod(yaml_path, 0o777)

                self.get_logger().info(f"Saved 2D map to {file_path} and {yaml_path}")

        if self.save_3d_map:
            ply_path = os.path.join(self.save_dir, f'{self.file_prefix}_{now_str}.ply')

            if self.map_3d_source == 'pointcloud2':
                if self.pointcloud is None:
                    self.get_logger().warn(
                        f"3D PointCloud2 saving is enabled, but no message has been received from {self.pointcloud_topic} yet.")
                else:
                    self.save_pointcloud2_ply(ply_path, self.pointcloud)
                    self.get_logger().info(f"Saved PointCloud2 to {ply_path}")
            else:
                if self.occupied_cells is None:
                    self.get_logger().warn(
                        f"3D occupied-cells saving is enabled, but no message has been received from {self.occupied_cells_topic} yet.")
                else:
                    self.save_occupied_cells_ply(ply_path, self.occupied_cells)
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

        self.sync_file_and_directory(filename)

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

        self.sync_file_and_directory(yaml_path)

    def save_pointcloud2_ply(self, filename, pointcloud_msg):
        field_names = [field.name for field in pointcloud_msg.fields]

        required_fields = {'x', 'y', 'z'}
        missing = required_fields - set(field_names)
        if missing:
            self.get_logger().error(
                f"PointCloud2 is missing required fields: {sorted(missing)}")
            return

        points = []
        use_rgb = 'rgb' in field_names or 'rgba' in field_names
        color_field = 'rgb' if 'rgb' in field_names else 'rgba' if 'rgba' in field_names else None

        read_fields = ['x', 'y', 'z']
        if color_field is not None:
            read_fields.append(color_field)

        for p in point_cloud2.read_points(
                pointcloud_msg,
                field_names=read_fields,
                skip_nans=self.skip_invalid_points):
            x = float(p[0])
            y = float(p[1])
            z = float(p[2])

            if self.skip_invalid_points and not np.isfinite([x, y, z]).all():
                continue

            if use_rgb:
                r, g, b = self.unpack_rgb(p[3])
                points.append([x, y, z, r, g, b])
            else:
                points.append([x, y, z])

        if len(points) == 0:
            self.get_logger().warn("No valid points in PointCloud2!")
            return

        if use_rgb:
            df = pd.DataFrame(points, columns=['x', 'y', 'z', 'red', 'green', 'blue'])
            df[['red', 'green', 'blue']] = df[['red', 'green', 'blue']].astype(np.uint8)
        else:
            df = pd.DataFrame(points, columns=['x', 'y', 'z'])

        df = self.apply_isolated_point_filter_if_enabled(df, source_name='PointCloud2')
        if df.empty:
            self.get_logger().warn(
                "All PointCloud2 points were removed by the isolated-point filter; PLY was not saved.")
            return

        self.write_ply_atomically(filename, df)

    def save_occupied_cells_ply(self, filename, marker_array_msg):
        points = []

        for marker in marker_array_msg.markers:
            for p in marker.points:
                points.append([p.x, p.y, p.z])

        if len(points) == 0:
            self.get_logger().warn("No points in occupied_cells!")
            return

        df = pd.DataFrame(points, columns=['x', 'y', 'z'])

        if self.component_filter_apply_to_occupied_cells or self.isolated_filter_apply_to_occupied_cells:
            df = self.apply_isolated_point_filter_if_enabled(df, source_name='occupied_cells')
            if df.empty:
                self.get_logger().warn(
                    "All occupied-cell points were removed by the isolated-point filter; PLY was not saved.")
                return

        self.write_ply_atomically(filename, df)


    def apply_isolated_point_filter_if_enabled(self, dataframe, source_name):
        # Keep the original function name for compatibility, but now the default
        # filter is global component outlier removal. The local isolated filter is
        # optional and runs after the component filter only when explicitly enabled.
        df = dataframe

        if self.enable_component_outlier_filter:
            before = len(df)
            filtered = self.filter_small_separated_components_by_voxel(
                df,
                voxel_size_m=self.component_voxel_size_m,
                min_points=self.component_min_points,
                keep_ratio_to_largest=self.component_keep_ratio_to_largest)
            after = len(filtered)
            removed = before - after
            removed_ratio = removed / max(before, 1)

            if removed_ratio > self.component_filter_max_removed_ratio:
                self.get_logger().warn(
                    f"Component outlier filter for {source_name} would remove "
                    f"{removed}/{before} points ({100.0 * removed_ratio:.1f}%), "
                    f"which is larger than component_filter_max_removed_ratio="
                    f"{self.component_filter_max_removed_ratio:.2f}. "
                    "Skipping component filter and keeping the current cloud.")
            else:
                self.get_logger().info(
                    f"Component outlier filter for {source_name}: kept {after}/{before} points "
                    f"and removed {removed} points in small separated components "
                    f"({100.0 * removed_ratio:.1f}%).")
                df = filtered

        if not self.enable_isolated_point_filter:
            return df

        before = len(df)
        if before == 0:
            return df

        filtered = self.filter_isolated_points_by_radius(
            df,
            radius_m=self.isolated_neighbor_radius_m,
            min_neighbors=self.isolated_min_neighbors)

        after = len(filtered)
        removed = before - after
        removed_ratio = removed / max(before, 1)

        if removed_ratio > self.isolated_filter_max_removed_ratio:
            self.get_logger().warn(
                f"Local isolated-point filter for {source_name} would remove "
                f"{removed}/{before} points ({100.0 * removed_ratio:.1f}%), "
                f"which is larger than isolated_filter_max_removed_ratio="
                f"{self.isolated_filter_max_removed_ratio:.2f}. "
                "Skipping local isolated filter and keeping the current cloud.")
            return df

        self.get_logger().info(
            f"Local isolated-point filter for {source_name}: kept {after}/{before} points "
            f"and removed {removed} isolated points ({100.0 * removed_ratio:.1f}%).")
        return filtered


    @staticmethod
    def filter_small_separated_components_by_voxel(
            dataframe,
            voxel_size_m,
            min_points,
            keep_ratio_to_largest):
        """Remove small point clusters that are globally separated from the main cloud.

        Algorithm:
          1. Convert points to coarse voxels.
          2. Build connected components between adjacent occupied voxels.
          3. Count points in each component.
          4. Keep large components; remove tiny separated components.

        This is different from a local density filter. A sparse but connected
        wall or object can remain, while single flying points far from the main
        cloud are removed.
        """
        xyz = dataframe[['x', 'y', 'z']].to_numpy(dtype=np.float64, copy=False)

        finite_mask = np.isfinite(xyz).all(axis=1)
        if not finite_mask.all():
            dataframe = dataframe.loc[finite_mask].reset_index(drop=True)
            xyz = dataframe[['x', 'y', 'z']].to_numpy(dtype=np.float64, copy=False)

        n = len(dataframe)
        if n == 0:
            return dataframe
        if n <= min_points:
            return dataframe

        voxels = np.floor(xyz / voxel_size_m).astype(np.int64)

        voxel_to_point_count = {}
        voxel_to_point_indices = {}
        for i, v in enumerate(voxels):
            key = (int(v[0]), int(v[1]), int(v[2]))
            voxel_to_point_count[key] = voxel_to_point_count.get(key, 0) + 1
            voxel_to_point_indices.setdefault(key, []).append(i)

        voxel_keys = list(voxel_to_point_count.keys())
        voxel_set = set(voxel_keys)
        visited = set()
        keep_point_mask = np.zeros(n, dtype=bool)

        offsets = [
            (dx, dy, dz)
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for dz in (-1, 0, 1)
            if not (dx == 0 and dy == 0 and dz == 0)
        ]

        components = []
        for start in voxel_keys:
            if start in visited:
                continue

            stack = [start]
            visited.add(start)
            component_voxels = []
            point_count = 0

            while stack:
                key = stack.pop()
                component_voxels.append(key)
                point_count += voxel_to_point_count[key]

                x, y, z = key
                for dx, dy, dz in offsets:
                    neighbor = (x + dx, y + dy, z + dz)
                    if neighbor in voxel_set and neighbor not in visited:
                        visited.add(neighbor)
                        stack.append(neighbor)

            components.append((point_count, component_voxels))

        if not components:
            return dataframe

        largest_points = max(point_count for point_count, _ in components)
        dynamic_min_points = int(np.ceil(largest_points * keep_ratio_to_largest))
        keep_threshold = max(int(min_points), dynamic_min_points)

        # Always keep the largest component. This prevents accidental deletion if
        # the entire map is still small at the start of mapping.
        largest_index = max(range(len(components)), key=lambda idx: components[idx][0])

        for comp_index, (point_count, component_voxels) in enumerate(components):
            if comp_index == largest_index or point_count >= keep_threshold:
                for key in component_voxels:
                    keep_point_mask[voxel_to_point_indices[key]] = True

        return dataframe.loc[keep_point_mask].reset_index(drop=True)

    @staticmethod
    def filter_isolated_points_by_radius(dataframe, radius_m, min_neighbors):
        """Remove only points that have too few nearby neighbors.

        This is intentionally weaker than a density filter.

        Rule:
          keep point p if there are at least min_neighbors OTHER points within
          radius_m [m] from p.

        With the default min_neighbors=1, only a point that has no neighbor near
        it is removed. Dense/normal map regions are preserved.
        """
        xyz = dataframe[['x', 'y', 'z']].to_numpy(dtype=np.float64, copy=False)

        finite_mask = np.isfinite(xyz).all(axis=1)
        if not finite_mask.all():
            dataframe = dataframe.loc[finite_mask].reset_index(drop=True)
            xyz = dataframe[['x', 'y', 'z']].to_numpy(dtype=np.float64, copy=False)

        n = len(dataframe)
        if n == 0:
            return dataframe
        if n <= min_neighbors:
            return dataframe

        # Voxel hash for fast radius search without requiring scipy/open3d.
        # Cell size equals radius. Candidate neighbors are in surrounding 27 cells.
        voxel = np.floor(xyz / radius_m).astype(np.int64)
        cell_to_indices = {}
        for i, (vx, vy, vz) in enumerate(voxel):
            key = (int(vx), int(vy), int(vz))
            cell_to_indices.setdefault(key, []).append(i)

        offsets = [
            (dx, dy, dz)
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for dz in (-1, 0, 1)
        ]

        radius_sq = radius_m * radius_m
        keep_mask = np.zeros(n, dtype=bool)

        for i, (vx, vy, vz) in enumerate(voxel):
            p = xyz[i]
            neighbor_count = 0
            bx = int(vx)
            by = int(vy)
            bz = int(vz)

            for dx, dy, dz in offsets:
                candidate_indices = cell_to_indices.get((bx + dx, by + dy, bz + dz), [])
                for j in candidate_indices:
                    if j == i:
                        continue
                    diff = xyz[j] - p
                    if float(np.dot(diff, diff)) <= radius_sq:
                        neighbor_count += 1
                        if neighbor_count >= min_neighbors:
                            keep_mask[i] = True
                            break
                if keep_mask[i]:
                    break

        return dataframe.loc[keep_mask].reset_index(drop=True)

    @staticmethod
    def unpack_rgb(rgb_value):
        # PointCloud2 RGB is often stored as a packed float32, but sometimes as int.
        # This handles both representations.
        if isinstance(rgb_value, float) or isinstance(rgb_value, np.floating):
            rgb_uint32 = np.asarray([rgb_value], dtype=np.float32).view(np.uint32)[0]
        else:
            rgb_uint32 = np.uint32(rgb_value)

        r = np.uint8((rgb_uint32 >> 16) & 0xFF)
        g = np.uint8((rgb_uint32 >> 8) & 0xFF)
        b = np.uint8(rgb_uint32 & 0xFF)
        return r, g, b

    def write_ply_atomically(self, filename, dataframe):
        cloud = PyntCloud(dataframe)
        tmp_filename = filename + '.tmp.ply'

        cloud.to_file(tmp_filename, as_text=True)

        self.sync_file_and_directory(tmp_filename)
        os.replace(tmp_filename, filename)
        os.chmod(filename, 0o777)
        self.sync_directory(os.path.dirname(filename))

    @staticmethod
    def sync_file_and_directory(filename):
        with open(filename, 'rb') as f:
            os.fsync(f.fileno())
        MapSaver.sync_directory(os.path.dirname(filename))

    @staticmethod
    def sync_directory(directory):
        if not directory:
            directory = '.'
        dir_fd = os.open(directory, os.O_DIRECTORY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)


def main(args=None):
    rclpy.init(args=args)
    node = MapSaver()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
