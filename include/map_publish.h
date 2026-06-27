#pragma once

#include <memory>

#include <pcl/filters/voxel_grid.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "ikd-Tree/ikd_Tree.h"

template <typename PointT>
std::shared_ptr<pcl::PointCloud<PointT>> build_map_cloud_from_ikdtree(
    KD_TREE<PointT> &ikdtree) {
  auto map_cloud = std::make_shared<pcl::PointCloud<PointT>>();
  ikdtree.PCL_Storage.clear();
  ikdtree.flatten(ikdtree.Root_Node, ikdtree.PCL_Storage, NOT_RECORD);
  map_cloud->points = ikdtree.PCL_Storage;
  map_cloud->width = map_cloud->points.size();
  map_cloud->height = 1;
  map_cloud->is_dense = false;
  return map_cloud;
}

template <typename PointT>
std::shared_ptr<pcl::PointCloud<PointT>> downsample_map_cloud_for_publish(
    const pcl::PointCloud<PointT> &input_cloud, float leaf_size) {
  auto filtered_cloud = std::make_shared<pcl::PointCloud<PointT>>();
  pcl::VoxelGrid<PointT> voxel_grid;
  voxel_grid.setLeafSize(leaf_size, leaf_size, leaf_size);
  voxel_grid.setInputCloud(input_cloud.makeShared());
  voxel_grid.filter(*filtered_cloud);
  filtered_cloud->width = filtered_cloud->points.size();
  filtered_cloud->height = 1;
  filtered_cloud->is_dense = false;
  return filtered_cloud;
}
