#!/bin/bash

echo "=== ROS Supervoxel Debugging Script ==="

# 检查是否有roscore运行
if ! pgrep -x "rosmaster" > /dev/null; then
    echo "Starting roscore..."
    roscore &
    sleep 3
fi

echo "1. Setting up coordinate transforms..."

# 发布静态坐标系变换 - 注意没有前导斜杠
rosrun tf2_ros static_transform_publisher 0 0 0 0 0 0 map base_link &
TF1_PID=$!

rosrun tf2_ros static_transform_publisher 0 0 0 0 0 0 base_link openni_rgb_optical_frame &
TF2_PID=$!

sleep 2

echo "2. Checking available topics..."
rostopic list | grep -E "(camera|points|cloud)"

echo ""
echo "3. Starting supervoxel node..."
echo "   Input topic: /camera/rgb/points"  
echo "   Output topic: /supervoxel_cloud"
echo "   Marker topic: /supervoxel_hulls"
echo ""

# 启动supervoxel节点
rosrun super_voxel supervoxel_node

# 清理函数
cleanup() {
    echo "Cleaning up..."
    kill $TF1_PID $TF2_PID 2>/dev/null
    killall rosmaster 2>/dev/null
}

# 设置信号处理
trap cleanup EXIT INT TERM
