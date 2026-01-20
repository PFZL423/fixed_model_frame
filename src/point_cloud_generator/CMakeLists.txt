cmake_minimum_required(VERSION 3.0.2)
project(point_cloud_generator LANGUAGES CXX CUDA)

## Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  pcl_ros
  roscpp
  sensor_msgs
)

## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)
find_package(PCL 1.8 REQUIRED COMPONENTS
  common
  io
  filters
  sample_consensus
)
find_package(CUDA REQUIRED)
# 1. 查找头文件所在的目录
find_path(CERES_INCLUDE_DIR ceres/ceres.h)
find_path(GLOG_INCLUDE_DIR glog/logging.h)
find_path(GFLAGS_INCLUDE_DIR gflags/gflags.h)
find_path(CUDA_TOOLKIT_ROOT_DIR
    NAMES bin/nvcc
    PATHS /usr/local/cuda /usr/local/cuda-12.8
    NO_DEFAULT_PATH
)

link_directories(${CUDA_TOOLKIT_ROOT_DIR}/targets/x86_64-linux/lib)


# 2. 查找库文件 (.so)
find_library(CERES_LIBRARY NAMES ceres)
find_library(GLOG_LIBRARY NAMES glog)
find_library(GFLAGS_LIBRARY NAMES gflags)

## Uncomment this if the package has a setup.py. This macro ensures
## modules and global scripts declared therein get installed
## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
# catkin_python_setup()

################################################
## Declare ROS messages, services and actions ##
################################################

## To declare and build messages, services or actions from within this
## package, follow these steps:
## * Let MSG_DEP_SET be the set of packages whose message types you use in
##   your messages/services/actions (e.g. std_msgs, actionlib_msgs, ...).
## * In the file package.xml:
##   * add a build_depend tag for "message_generation"
##   * add a build_depend and a exec_depend tag for each package in MSG_DEP_SET
##   * If MSG_DEP_SET isn't empty the following dependency has been pulled in
##     but can be declared for certainty nonetheless:
##     * add a exec_depend tag for "message_runtime"
## * In this file (CMakeLists.txt):
##   * add "message_generation" and every package in MSG_DEP_SET to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * add "message_runtime" and every package in MSG_DEP_SET to
##     catkin_package(CATKIN_DEPENDS ...)
##   * uncomment the add_*_files sections below as needed
##     and list every .msg/.srv/.action file to be processed
##   * uncomment the generate_messages entry below
##   * add every package in MSG_DEP_SET to generate_messages(DEPENDENCIES ...)

## Generate messages in the 'msg' folder
# add_message_files(
#   FILES
#   Message1.msg
#   Message2.msg
# )

## Generate services in the 'srv' folder
# add_service_files(
#   FILES
#   Service1.srv
#   Service2.srv
# )

## Generate actions in the 'action' folder
# add_action_files(
#   FILES
#   Action1.action
#   Action2.action
# )

## Generate added messages and services with any dependencies listed here
# generate_messages(
#   DEPENDENCIES
#   sensor_msgs
# )

################################################
## Declare ROS dynamic reconfigure parameters ##
################################################

## To declare and build dynamic reconfigure parameters within this
## package, follow these steps:
## * In the file package.xml:
##   * add a build_depend and a exec_depend tag for "dynamic_reconfigure"
## * In this file (CMakeLists.txt):
##   * add "dynamic_reconfigure" to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * uncomment the "generate_dynamic_reconfigure_options" section below
##     and list every .cfg file to be processed

## Generate dynamic reconfigure parameters in the 'cfg' folder
# generate_dynamic_reconfigure_options(
#   cfg/DynReconf1.cfg
#   cfg/DynReconf2.cfg
# )

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if your package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
 INCLUDE_DIRS include
 LIBRARIES 
  Point_cloud_preprocessor_lib
  shape_detector_lib
  shape_detector_gpu_lib
 CATKIN_DEPENDS pcl_ros roscpp sensor_msgs
#  DEPENDS system_lib
)
# 找到 CUDA_NVCC_FLAGS 变量或者自己设置
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -diag-suppress 20012")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")


# 禁用 Eigen 的内存对齐要求，但不关闭 SSE/AVX
# add_definitions(-DEIGEN_DONT_ALIGN_STATICALLY)
# add_definitions(-DEIGEN_DONT_ALIGN)
# add_definitions(-DEIGEN_MAX_ALIGN_BYTES=0)
# add_definitions(-DEIGEN_MAX_STATIC_ALIGN_BYTES=0)



###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIR}
  ${GLOG_INCLUDE_DIR}
  ${GFLAGS_INCLUDE_DIR}
  ${CUDA_INCLUDE_DIRS} # <-- NEW: 添加CUDA头文件路径
)
# 4. 检查是否所有东西都找到了，如果没找到就报错
if(NOT CERES_LIBRARY OR NOT GLOG_LIBRARY OR NOT GFLAGS_LIBRARY)
    message(FATAL_ERROR "Could not find Ceres, Glog, or GFlags libraries manually. "
                        "Please ensure libceres-dev, libgoogle-glog-dev, and libgflags-dev are installed correctly.")
endif()






# 声明库
add_library(Point_cloud_preprocessor_lib
  src/Point_cloud_preprocessor.cpp
  
)




# --- 第二个库: 形状检测模块 (新增) ---
# 名称: shape_detector_lib 
# 源文件: src/MinimalSampleQuadric.cpp
add_library(shape_detector_lib
  src/MinimalSampleQuadric.cpp
)
# 为形状检测库链接它所需要的库。
# 关键：它依赖预处理库，所以必须链接 Point_cloud_preprocessor_lib
target_link_libraries(shape_detector_lib
  Point_cloud_preprocessor_lib # <--- 链接到我们包内的另一个库
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
 # 使用我们手动找到的库
  ${CERES_LIBRARY}
  ${GLOG_LIBRARY}
  ${GFLAGS_LIBRARY}
)

# --- 第三个库: GPU形状检测模块 (原有实现) ---
add_library(shape_detector_gpu_lib
  src/MinimalSampleQuadric_GPU.cu
)
target_compile_options(shape_detector_gpu_lib PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-diag-suppress 20012>)

target_link_libraries(shape_detector_gpu_lib
  Point_cloud_preprocessor_lib
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
  ${CUDA_LIBRARIES}
  cusolver
)


# 创建最终的ROS节点可执行文件
add_executable(final_test_node
  src/final_test.cpp
)
target_link_libraries(final_test_node
  shape_detector_lib # <--- 链接到主库
  ${catkin_LIBRARIES}
  ${Ceres_LIBRARIES} 
  ${GLOG_LIBRARIES}     # <-- 确认是大写

)


# GPU 测试节点 (原有)
add_executable(gpu_test_node src/gpu_test_node.cpp)
target_link_libraries(gpu_test_node
    ${catkin_LIBRARIES}
    shape_detector_gpu_lib
    cusolver
)
add_dependencies(gpu_test_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Declare a C++ library
# add_library(${PROJECT_NAME}
#   src/${PROJECT_NAME}/point_cloud_generator.cpp
# )

## Add cmake target dependencies of the library
## as an example, code may need to be generated before libraries
## either from message generation or dynamic reconfigure
# add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Declare a C++ executable
## With catkin_make all packages are built within a single CMake context
## The recommended prefix ensures that target names across packages don't collide
# add_executable(${PROJECT_NAME}_node src/point_cloud_generator_node.cpp)
# add_executable(pc_generator src/pc_generator.cpp)
# add_executable(pc_generator_advanced src/pc_generator_advanced.cpp)
add_executable(pc_generator_finished src/pc_generator_finished.cpp)






## Rename C++ executable without prefix
## The above recommended prefix causes long target names, the following renames the
## target back to the shorter version for ease of user use
## e.g. "rosrun someones_pkg node" instead of "rosrun someones_pkg someones_pkg_node"
# set_target_properties(${PROJECT_NAME}_node PROPERTIES OUTPUT_NAME node PREFIX "")

## Add cmake target dependencies of the executable
## same as for the library above
# add_dependencies(${PROJECT_NAME}_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Specify libraries to link a library or executable target against
# target_link_libraries(${PROJECT_NAME}_node
#   ${catkin_LIBRARIES}
# )

target_link_libraries(pc_generator_finished
  ${catkin_LIBRARIES}
  ${PCL_LIBRARIES}
)
# add_executable(simple_fitter_node src/simple_fitter_node.cpp)
# target_link_libraries(simple_fitter_node
#   ${catkin_LIBRARIES}
#   ${PCL_LIBRARIES}
#   ${EIGEN3_LIBRARIES}
#   Point_cloud_preprocessor_lib
# )
#############
## Install ##
#############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
# catkin_install_python(PROGRAMS
#   scripts/my_python_script
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark executables for installation
## See http://docs.ros.org/melodic/api/catkin/html/howto/format1/building_executables.html
# install(TARGETS ${PROJECT_NAME}_node
#   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark libraries for installation
## See http://docs.ros.org/melodic/api/catkin/html/howto/format1/building_libraries.html
# install(TARGETS ${PROJECT_NAME}
#   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
# )


install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)


# 安装库文件 (.so 或 .a)
install(TARGETS
  Point_cloud_preprocessor_lib # 安装预处理库
  shape_detector_lib           # 安装检测器库
  final_test_node              # 安装可执行节点
  ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# 安装头文件 (.h)
install(DIRECTORY include/${PROJECT_NAME}/
  DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)

## Mark cpp header files for installation
# install(DIRECTORY include/${PROJECT_NAME}/
#   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
#   FILES_MATCHING PATTERN "*.h"
#   PATTERN ".svn" EXCLUDE
# )

## Mark other files for installation (e.g. launch and bag files, etc.)
# install(FILES
#   # myfile1
#   # myfile2
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
# )

#############
## Testing ##
#############

## Add gtest based cpp test target and link libraries
# catkin_add_gtest(${PROJECT_NAME}-test test/test_point_cloud_generator.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)
install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)