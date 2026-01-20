#!/bin/bash

echo "=== SupervoxelProcessor Library Test Script ==="

# 检查是否有roscore运行
if ! pgrep -x "rosmaster" > /dev/null; then
    echo "Starting roscore..."
    roscore &
    sleep 3
fi

echo "1. Setting up coordinate transforms..."

# 发布静态坐标系变换 - 注意没有前导斜杠（复用原有方案）
rosrun tf2_ros static_transform_publisher 0 0 0 0 0 0 map base_link &
TF1_PID=$!

rosrun tf2_ros static_transform_publisher 0 0 0 0 0 0 base_link openni_rgb_optical_frame &
TF2_PID=$!

sleep 2

echo "2. Checking available topics..."
rostopic list | grep -E "(camera|points|cloud)"

echo ""
echo "3. Starting test library node..."
echo "   Input topic: /camera/rgb/points"  
echo "   Supervoxel cloud: /test_supervoxel_cloud"
echo "   Convex hulls: /test_convex_hulls"
echo ""
echo "   Library features:"
echo "   - SupervoxelProcessor class"
echo "   - No voxel downsampling"
echo "   - 2D convex hull computation"
echo "   - Performance statistics"
echo ""

# 启动测试节点
rosrun super_voxel test_library

# 清理函数
cleanup() {
    echo "Cleaning up..."
    kill $TF1_PID $TF2_PID 2>/dev/null
    killall rosmaster 2>/dev/null
}

# 设置信号处理
trap cleanup EXIT INT TERM
