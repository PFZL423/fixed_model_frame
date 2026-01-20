#!/bin/bash

# 启动supervoxel节点的脚本
# 这个脚本会自动设置必要的坐标系变换

echo "Setting up coordinate frames..."

# 发布静态坐标系变换
rosrun tf2_ros static_transform_publisher 0 0 0 0 0 0 map base_link &
rosrun tf2_ros static_transform_publisher 0 0 0 0 0 0 base_link openni_rgb_optical_frame &

sleep 2

echo "Starting supervoxel node..."
rosrun super_voxel supervoxel_node
