#include <gtest/gtest.h>

#include "map_publish.h"

using TestPointType = pcl::PointXYZINormal;

TEST(MapPublish, FlattensToOnePointPerLeaf)
{
  pcl::PointCloud<TestPointType> input_cloud;
  input_cloud.height = 1;
  input_cloud.is_dense = false;

  TestPointType p1;
  p1.x = 0.10f;
  p1.y = 0.10f;
  p1.z = 0.10f;
  input_cloud.points.push_back(p1);

  TestPointType p2;
  p2.x = 0.20f;
  p2.y = 0.20f;
  p2.z = 0.20f;
  input_cloud.points.push_back(p2);
  input_cloud.width = input_cloud.points.size();

  auto map_cloud = downsample_map_cloud_for_publish(input_cloud, 0.5f);

  ASSERT_NE(map_cloud, nullptr);
  EXPECT_EQ(map_cloud->points.size(), 1u);
}
