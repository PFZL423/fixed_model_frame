cmake_minimum_required(VERSION 3.0.2)
project(gpu_demo LANGUAGES CXX CUDA)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  roscpp
  std_msgs
  pcl_ros
  sensor_msgs
  rosbag
  rospy
  cv_bridge
  tf2_msgs
)

## System dependencies
find_package(PCL 1.8 REQUIRED COMPONENTS
  common
  io
  filters
  sample_consensus
)
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)

# 查找CUDA工具包路径
find_path(CUDA_TOOLKIT_ROOT_DIR
    NAMES bin/nvcc
    PATHS /usr/local/cuda /usr/local/cuda-12.8
    NO_DEFAULT_PATH
)

# 添加CUDA库路径
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/targets/x86_64-linux/lib)

# 设置CUDA编译选项
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -diag-suppress 20012")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")

# 🆕 添加简化错误信息的编译选项（只使用广泛支持的选项）
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ftemplate-backtrace-limit=5")
# CUDA nvcc doesn't support backtrace limit, so only add to CXX flags

###################################
## catkin specific configuration ##
###################################
catkin_package(
 INCLUDE_DIRS include
 LIBRARIES quadric_detect_gpu_lib gpu_preprocessor_lib
 CATKIN_DEPENDS roscpp std_msgs pcl_ros sensor_msgs rosbag rospy cv_bridge tf2_msgs
 DEPENDS PCL OpenCV
)

###########
## Build ##
###########

## Specify additional locations of header files
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  ${CUDA_TOOLKIT_ROOT_DIR}/include  # 🆕 确保包含CUDA工具包头文件（包含CUB）
)

# 添加PCL编译定义
add_definitions(${PCL_DEFINITIONS})

## 声明Batch-RANSAC GPU检测库
add_library(quadric_detect_gpu_lib
  src/QuadricDetect.cpp
  src/QuadricDetect.cu
)

## 🆕 新增：GPU预处理库
add_library(gpu_preprocessor_lib
  src/GPUPreprocessor.cpp
  src/GPUPreprocessor_kernels.cu
)

# 🆕 为 gpu_preprocessor_lib 设置分离编译
set_property(TARGET gpu_preprocessor_lib 
             PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# 设置CUDA编译选项
target_compile_options(quadric_detect_gpu_lib PRIVATE 
  $<$<COMPILE_LANGUAGE:CUDA>:-diag-suppress 20012>
  $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
  $<$<COMPILE_LANGUAGE:CXX>:-ftemplate-backtrace-limit=5>
)

target_compile_options(gpu_preprocessor_lib PRIVATE 
  $<$<COMPILE_LANGUAGE:CUDA>:-diag-suppress 20012>
  $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
  $<$<COMPILE_LANGUAGE:CXX>:-ftemplate-backtrace-limit=5>
)

# 链接依赖库
target_link_libraries(quadric_detect_gpu_lib
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${CUDA_LIBRARIES}
  cusolver  # cuSolver for batch SVD
  curand    # cuRAND for random number generation
)

# 添加 OpenMP 支持
if(OpenMP_CXX_FOUND)
    target_link_libraries(gpu_preprocessor_lib
      ${catkin_LIBRARIES}
      ${PCL_LIBRARIES}
      ${CUDA_LIBRARIES}
      curand    # cuRAND for random number generation
      OpenMP::OpenMP_CXX
    )
    # 设置 OpenMP 编译标志
    target_compile_options(gpu_preprocessor_lib PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${OpenMP_CXX_FLAGS}>)
else()
    target_link_libraries(gpu_preprocessor_lib
      ${catkin_LIBRARIES}
      ${PCL_LIBRARIES}
      ${CUDA_LIBRARIES}
      curand    # cuRAND for random number generation
    )
endif()

# GPU检查节点 (🆕 更新为完整的GPU流水线测试)
add_executable(gpu_check_node src/gpu_check.cpp)
target_link_libraries(gpu_check_node
  quadric_detect_gpu_lib
  gpu_preprocessor_lib
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${CUDA_LIBRARIES}
  cusolver
  curand
)
add_dependencies(gpu_check_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})


add_executable(SuperVoxel src/SuperVoxel.cpp)
target_link_libraries(SuperVoxel
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
)

# 测试demo_test ROS节点 - 代码保留在demo_test.cpp.disabled
# add_executable(demo_test src/demo_test.cpp)
# target_link_libraries(demo_test
#   quadric_detect_gpu_lib
#   gpu_preprocessor_lib
#   ${catkin_LIBRARIES}
#   ${PCL_LIBRARIES}
#   ${CUDA_LIBRARIES}
#   cusolver
#   curand
# )
# add_dependencies(demo_test ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

# 🆕 新增：GPU预处理测试节点 - 测试GPU预处理功能
add_executable(test_preprocess_node src/test_preprocess.cpp)
target_link_libraries(test_preprocess_node
  gpu_preprocessor_lib
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${CUDA_LIBRARIES}
)
add_dependencies(test_preprocess_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

# 🆕 新增：法线计算测试节点 - 测试空间哈希法线计算
add_executable(normal_test src/normal_test.cpp)
target_link_libraries(normal_test
  gpu_preprocessor_lib
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${CUDA_LIBRARIES}
)
add_dependencies(normal_test ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

#############
## Install ##
#############

# 安装库文件
install(TARGETS
  quadric_detect_gpu_lib
  gpu_preprocessor_lib
  gpu_check_node
  # demo_test  # 已禁用
  test_preprocess_node
  normal_test
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# 安装头文件
install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
  FILES_MATCHING PATTERN "*.h" PATTERN "*.cuh"
  PATTERN ".svn" EXCLUDE
)

# 安装Python脚本
install(PROGRAMS
  script/generate_pointcloud.py
  script/transform.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# 安装launch和config文件
install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
  FILES_MATCHING PATTERN "*.launch"
)

install(DIRECTORY config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
  FILES_MATCHING PATTERN "*.yaml"
)

#############
## Testing ##
#############

# 添加基于gtest的测试（可选）
# catkin_add_gtest(${PROJECT_NAME}-test test/test_gpu_demo.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test quadric_detect_gpu_lib)
# endif()
