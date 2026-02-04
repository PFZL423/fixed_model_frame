#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <geometry_msgs/Point.h>
#include <std_msgs/ColorRGBA.h>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <Eigen/Dense>
#include <omp.h>
#include <limits>
#include <iostream>
#include <iomanip>

#include "PlaneDetect/PlaneDetect.h"
#include "super_voxel/supervoxel.h"
#include "gpu_demo/GPUPreprocessor.h"
#include "gpu_demo/QuadricDetect.h"
#include <cuda_runtime.h>
#include <memory>

class PlaneSupervoxelNode
{
public:
    PlaneSupervoxelNode(ros::NodeHandle &nh, ros::NodeHandle &pnh)
        : nh_(nh), pnh_(pnh), plane_detector_(nullptr), unified_stream_(nullptr)
    {
        // 读取参数
        loadParameters();

        // 初始化平面检测器
        initializePlaneDetector();

        // 初始化 GPU 预处理器和统一 CUDA 流
        initializeGPUPreprocessor();

        // 初始化二次曲面检测器
        initializeQuadricDetector();

        // 初始化超体素处理器
        initializeSupervoxelProcessor();

        // 设置订阅和发布
        setupPubSub();

        ROS_INFO("Unified Detection Node initialized successfully");
    }

    ~PlaneSupervoxelNode()
    {
        // 销毁 CUDA 流
        if (unified_stream_ != nullptr)
        {
            cudaStreamDestroy(unified_stream_);
            unified_stream_ = nullptr;
        }
    }

private:
    ros::NodeHandle nh_, pnh_;
    
    // 订阅者
    ros::Subscriber camera_sub_;           // 订阅原始点云 /camera/rgb/points
    
    // 发布者
    ros::Publisher plane_marker_pub_;      // 平面可视化
    ros::Publisher convex_hull_marker_pub_;  // 凸包可视化
    ros::Publisher result_cloud_pub_;      // 最终结果点云

    std::unique_ptr<PlaneDetect<pcl::PointXYZI>> plane_detector_;
    DetectorParams detector_params_;

    // GPU 预处理器和流
    std::shared_ptr<GPUPreprocessor> gpu_preprocessor_;
    cudaStream_t unified_stream_;

    // 二次曲面检测器
    std::shared_ptr<QuadricDetect> quadric_detector_;
    quadric::DetectorParams quadric_params_;

    // 超体素处理器
    std::unique_ptr<super_voxel::SupervoxelProcessor> sv_processor_;
    super_voxel::SupervoxelParams sv_params_;


    // 参数
    bool enable_voxel_filter_;
    double voxel_leaf_size_;
    std::string input_topic_;
    std::string output_frame_;

    // 可视化参数（平面美化）
    std::string plane_color_scheme_ = "pastel"; // pastel | hsv | tab10
    bool plane_checkerboard_ = true;             // 是否使用棋盘纹理（交替色）
    double plane_alpha_ = 0.6;                   // 平面透明度 [0,1]
    int plane_grid_size_ = 20;                   // 网格密度（越大越细）

    // 平面裁剪到内点凸包参数
    bool plane_clip_to_hull_ = true;             // 是否裁剪到内点凸包
    double plane_hull_padding_ = 0.02;           // 凸包外扩（米）
    double plane_hull_smooth_factor_ = 0.15;     // 轻度平滑 [0,1]

    // 离群点移除参数
    bool enable_outlier_removal_;
    int outlier_k_neighbors_;
    double outlier_std_dev_thresh_;

    // 超体素功能开关
    bool enable_supervoxel_;
    int min_remaining_points_for_supervoxel_;

    // 全局/分项可视化开关
    bool enable_visualization_ = true;              // 全局开关
    bool enable_plane_visualization_ = true;        // 平面网格/法线
    bool enable_convex_hull_visualization_ = true;  // 超体素凸包

    void loadParameters()
    {
        // 体素降采样参数
        pnh_.param("enable_voxel_filter", enable_voxel_filter_, true);
        pnh_.param("voxel_leaf_size", voxel_leaf_size_, 0.02);

        // 离群点移除参数
        pnh_.param("enable_outlier_removal", enable_outlier_removal_, true);
        pnh_.param("outlier_k_neighbors", outlier_k_neighbors_, 50);
        pnh_.param("outlier_std_dev_thresh", outlier_std_dev_thresh_, 1.0);

        // 话题和坐标系参数
    pnh_.param<std::string>("input_topic", input_topic_, "/camera/depth_registered/points");
    pnh_.param<std::string>("output_frame", output_frame_, "camera_rgb_optical_frame");

    // 平面可视化美化参数
    pnh_.param<std::string>("plane_color_scheme", plane_color_scheme_, plane_color_scheme_);
    pnh_.param("plane_checkerboard", plane_checkerboard_, plane_checkerboard_);
    pnh_.param("plane_alpha", plane_alpha_, plane_alpha_);
    pnh_.param("plane_grid_size", plane_grid_size_, plane_grid_size_);
    pnh_.param("plane_clip_to_hull", plane_clip_to_hull_, plane_clip_to_hull_);
    pnh_.param("plane_hull_padding", plane_hull_padding_, plane_hull_padding_);
    pnh_.param("plane_hull_smooth_factor", plane_hull_smooth_factor_, plane_hull_smooth_factor_);

        // PlaneDetect算法参数
        pnh_.param("min_remaining_points_percentage", detector_params_.min_remaining_points_percentage, 0.03);
        pnh_.param("plane_distance_threshold", detector_params_.plane_distance_threshold, 0.02);
        pnh_.param("min_plane_inlier_count_absolute", detector_params_.min_plane_inlier_count_absolute, 500);
        pnh_.param("plane_max_iterations", detector_params_.plane_max_iterations, 2000);
        pnh_.param("min_plane_inlier_percentage", detector_params_.min_plane_inlier_percentage, 0.05);
        pnh_.param("batch_size", detector_params_.batch_size, 2048);
        pnh_.param("verbosity", detector_params_.verbosity, 1);
        
        // 两阶段RANSAC竞速参数
        pnh_.param("ransac_coarse_ratio", detector_params_.ransac_coarse_ratio, 0.02);
        pnh_.param("ransac_fine_k", detector_params_.ransac_fine_k, 20);

    // 超体素功能开关
        pnh_.param("enable_supervoxel", enable_supervoxel_, false);
        pnh_.param("min_remaining_points_for_supervoxel", min_remaining_points_for_supervoxel_, 500);

    // 全局/分项可视化开关
    pnh_.param("enable_visualization", enable_visualization_, enable_visualization_);
    pnh_.param("enable_plane_visualization", enable_plane_visualization_, enable_plane_visualization_);
    pnh_.param("enable_convex_hull_visualization", enable_convex_hull_visualization_, enable_convex_hull_visualization_);

        // 二次曲面检测参数
        pnh_.param("quadric_min_remaining_points_percentage", quadric_params_.min_remaining_points_percentage, 0.03);
        pnh_.param("quadric_distance_threshold", quadric_params_.quadric_distance_threshold, 0.02);
        pnh_.param("min_quadric_inlier_count_absolute", quadric_params_.min_quadric_inlier_count_absolute, 500);
        pnh_.param("quadric_max_iterations", quadric_params_.quadric_max_iterations, 5000);
        pnh_.param("min_quadric_inlier_percentage", quadric_params_.min_quadric_inlier_percentage, 0.05);
        pnh_.param("quadric_verbosity", quadric_params_.verbosity, 1);

        // 超体素算法参数
        pnh_.param("sv_voxel_resolution", sv_params_.voxel_resolution, 0.05);
        pnh_.param("sv_seed_resolution", sv_params_.seed_resolution, 0.2);
        pnh_.param("sv_color_importance", sv_params_.color_importance, 0.2);
        pnh_.param("sv_spatial_importance", sv_params_.spatial_importance, 0.4);
        pnh_.param("sv_normal_importance", sv_params_.normal_importance, 1.0);
        pnh_.param("sv_enable_voxel_downsample", sv_params_.enable_voxel_downsample, false);
        pnh_.param("sv_downsample_leaf_size", sv_params_.downsample_leaf_size, 0.02);
        pnh_.param("sv_use_2d_convex_hull", sv_params_.use_2d_convex_hull, true);
        int min_points_tmp = 3;
        pnh_.param("sv_min_points_for_hull", min_points_tmp, 3);
        sv_params_.min_points_for_hull = static_cast<size_t>(min_points_tmp);

        ROS_INFO("Parameters loaded:");
        ROS_INFO("  Voxel filter: %s (leaf_size=%.4f)", enable_voxel_filter_ ? "enabled" : "disabled", voxel_leaf_size_);
        ROS_INFO("  Outlier removal: %s (k=%d, std_dev=%.2f)", enable_outlier_removal_ ? "enabled" : "disabled", outlier_k_neighbors_, outlier_std_dev_thresh_);
        ROS_INFO("  Input topic: %s", input_topic_.c_str());
        ROS_INFO("  Distance threshold: %.4f", detector_params_.plane_distance_threshold);
        ROS_INFO("  Min inliers: %d", detector_params_.min_plane_inlier_count_absolute);
        ROS_INFO("  Batch size: %d", detector_params_.batch_size);
        ROS_INFO("  RANSAC coarse ratio: %.4f (stride=%d)", 
                 detector_params_.ransac_coarse_ratio,
                 (detector_params_.ransac_coarse_ratio >= 1.0) ? 1 : 
                 static_cast<int>(1.0 / detector_params_.ransac_coarse_ratio));
        ROS_INFO("  RANSAC fine k: %d", detector_params_.ransac_fine_k);
        ROS_INFO("  Supervoxel: %s", enable_supervoxel_ ? "enabled" : "disabled");
        if (enable_supervoxel_)
        {
            ROS_INFO("    Voxel resolution: %.3f", sv_params_.voxel_resolution);
            ROS_INFO("    Seed resolution: %.3f", sv_params_.seed_resolution);
            ROS_INFO("    Min points threshold: %d", min_remaining_points_for_supervoxel_);
        }
    ROS_INFO("  Plane viz: scheme=%s, checkerboard=%s, alpha=%.2f, grid=%d, clip_to_hull=%s, padding=%.3f, smooth=%.2f",
         plane_color_scheme_.c_str(), plane_checkerboard_ ? "true" : "false", plane_alpha_, plane_grid_size_,
         plane_clip_to_hull_ ? "true" : "false", plane_hull_padding_, plane_hull_smooth_factor_);
    ROS_INFO("  Visualization toggles: global=%s, planes=%s, convex_hulls=%s",
         enable_visualization_ ? "on" : "off",
         enable_plane_visualization_ ? "on" : "off",
         enable_convex_hull_visualization_ ? "on" : "off");
    }

    void initializePlaneDetector()
    {
        plane_detector_ = std::make_unique<PlaneDetect<pcl::PointXYZI>>(detector_params_);
        ROS_INFO("PlaneDetector initialized with batch_size=%d", detector_params_.batch_size);
    }

    void initializeGPUPreprocessor()
    {
        // 初始化 GPU 预处理器
        gpu_preprocessor_ = std::make_shared<GPUPreprocessor>();

        // 创建统一 CUDA 流
        cudaError_t err = cudaStreamCreate(&unified_stream_);
        if (err != cudaSuccess)
        {
            ROS_ERROR("Failed to create CUDA stream: %s", cudaGetErrorString(err));
            throw std::runtime_error("CUDA stream creation failed");
        }

        // 配置统一流（关键：确保串行无锁流水线）
        gpu_preprocessor_->setStream(unified_stream_);
        plane_detector_->setStream(unified_stream_);

        ROS_INFO("GPUPreprocessor initialized with unified CUDA stream");
    }

    void initializeQuadricDetector()
    {
        // 初始化二次曲面检测器
        quadric_detector_ = std::make_shared<QuadricDetect>(quadric_params_);
        
        // 绑定统一流
        quadric_detector_->setStream(unified_stream_);

        ROS_INFO("QuadricDetector initialized with threshold=%.3f", 
                 quadric_params_.quadric_distance_threshold);
    }

    void initializeSupervoxelProcessor()
    {
        if (enable_supervoxel_)
        {
            sv_processor_ = std::make_unique<super_voxel::SupervoxelProcessor>(sv_params_);
            ROS_INFO("SupervoxelProcessor initialized");
        }
    }

    void setupPubSub()
    {
        //  订阅原始点云（队列=1，最小延迟）
        camera_sub_ = nh_.subscribe(input_topic_, 1, &PlaneSupervoxelNode::cameraCallback, this);
        
        // 发布平面可视化
        plane_marker_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("plane_markers", 1, true);
        
        // 发布凸包可视化
        convex_hull_marker_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("convex_hull_markers", 1, true);
        
        // 发布最终结果点云
        result_cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("remaining_cloud", 1, true);

        ROS_INFO("=== Unified Detection Node: Plane + Quadric ===");
        ROS_INFO("Subscribed to camera: %s", input_topic_.c_str());
        ROS_INFO("Publishing plane markers to: plane_markers");
        ROS_INFO("Publishing final result to: remaining_cloud");
    }

    // 回调1：处理相机原始点云，进行平面检测
    void cameraCallback(const sensor_msgs::PointCloud2::ConstPtr &msg)
    {
        // 全链路总计时开始
        auto total_start = std::chrono::high_resolution_clock::now();
        
        ROS_INFO("Received point cloud with %d points", msg->width * msg->height);

        // ========== 计时：ROS消息转换 ==========
        auto ros_convert_start = std::chrono::high_resolution_clock::now();
        pcl::PointCloud<pcl::PointXYZI>::Ptr input_cloud(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::fromROSMsg(*msg, *input_cloud);
        auto ros_convert_end = std::chrono::high_resolution_clock::now();
        float ros_convert_time = std::chrono::duration<float, std::milli>(ros_convert_end - ros_convert_start).count();

        if (input_cloud->empty())
        {
            ROS_WARN("Received empty point cloud");
            return;
        }

        // ========== 计时：点云格式转换 ==========
        auto format_convert_start = std::chrono::high_resolution_clock::now();
        // ========== Step 1: GPU 预处理（体素下采样 + 离群点移除）==========
        // 转换为 PointXYZ（GPUPreprocessor 当前只支持 PointXYZ）
        // 使用 OpenMP 并行化点云转换，提升性能
        pcl::PointCloud<pcl::PointXYZ>::Ptr input_xyz(new pcl::PointCloud<pcl::PointXYZ>);
        input_xyz->points.resize(input_cloud->size());
        input_xyz->width = input_cloud->size();
        input_xyz->height = 1;
        input_xyz->is_dense = false;  // 可能包含无效点，稍后过滤
        
        #pragma omp parallel for
        for (size_t i = 0; i < input_cloud->size(); ++i)
        {
            const auto &pt = input_cloud->points[i];
            if (std::isfinite(pt.x) && std::isfinite(pt.y) && std::isfinite(pt.z))
            {
                input_xyz->points[i] = pcl::PointXYZ(pt.x, pt.y, pt.z);
            }
            else
            {
                // 标记为无效点（使用 NaN）
                input_xyz->points[i].x = std::numeric_limits<float>::quiet_NaN();
                input_xyz->points[i].y = std::numeric_limits<float>::quiet_NaN();
                input_xyz->points[i].z = std::numeric_limits<float>::quiet_NaN();
            }
        }
        
        // 过滤无效点
        input_xyz->points.erase(
            std::remove_if(input_xyz->points.begin(), input_xyz->points.end(),
                           [](const pcl::PointXYZ& pt) {
                               return !std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z);
                           }),
            input_xyz->points.end());
        input_xyz->width = input_xyz->size();
        input_xyz->height = 1;
        input_xyz->is_dense = true;
        auto format_convert_end = std::chrono::high_resolution_clock::now();
        float format_convert_time = std::chrono::duration<float, std::milli>(format_convert_end - format_convert_start).count();

        // 配置 GPU 预处理参数
        PreprocessConfig gpu_config;
        gpu_config.enable_voxel_filter = enable_voxel_filter_;
        gpu_config.voxel_size = static_cast<float>(voxel_leaf_size_);
        gpu_config.enable_outlier_removal = enable_outlier_removal_;
        gpu_config.outlier_method = PreprocessConfig::STATISTICAL;
        gpu_config.statistical_k = outlier_k_neighbors_;
        gpu_config.statistical_stddev = static_cast<float>(outlier_std_dev_thresh_);
        gpu_config.compute_normals = false;  // 平面检测不需要法线

        // 执行 GPU 预处理（数据已在 GPU 上）
        auto gpu_preprocess_start = std::chrono::high_resolution_clock::now();
        ProcessingResult gpu_result = gpu_preprocessor_->process(input_xyz, gpu_config);
        auto gpu_preprocess_end = std::chrono::high_resolution_clock::now();
        float gpu_preprocess_time = std::chrono::duration<float, std::milli>(
            gpu_preprocess_end - gpu_preprocess_start).count();

        // 输出 GPU 预处理性能统计
        const auto &gpu_stats = gpu_preprocessor_->getLastStats();
        ROS_INFO("GPU preprocessing: %.2f ms (upload=%.2fms, voxel=%.2fms, outlier=%.2fms), output: %zu points", 
                 gpu_preprocess_time, gpu_stats.upload_time_ms, gpu_stats.voxel_filter_time_ms,
                 gpu_stats.outlier_removal_time_ms, gpu_result.getPointCount());

        // ========== Step 2: 零拷贝指针传递 ==========
        GPUPoint3f* d_points = gpu_preprocessor_->getOutputBuffer();
        size_t point_count = gpu_preprocessor_->getOutputCount();

        if (d_points == nullptr || point_count == 0)
        {
            ROS_WARN("GPU preprocessing produced no valid points");
            return;
        }

        ROS_INFO("Zero-copy: borrowing GPU buffer (%zu points)", point_count);

        // ========== Step 3: 平面检测（零拷贝）==========
        auto plane_detect_start = std::chrono::high_resolution_clock::now();
        plane_detector_->findPlanesFromGPU(d_points, point_count);
        auto plane_detect_end = std::chrono::high_resolution_clock::now();
        float plane_detect_time = std::chrono::duration<float, std::milli>(
            plane_detect_end - plane_detect_start).count();

        ROS_INFO("Plane detection (zero-copy): %.2f ms", plane_detect_time);

        // ========== Step 3: GPU 压实接力 ==========
        size_t rem_count = 0;
        GPUPoint3f* d_rem_ptr = plane_detector_->getRemainingPointsGPU(rem_count);
        
        if (d_rem_ptr == nullptr || rem_count == 0)
        {
            ROS_WARN("No remaining points after plane detection");
            // 安全同步与释放
            cudaStreamSynchronize(unified_stream_);
            plane_detector_->releaseExternalBuffer();
            return;
        }

        ROS_INFO("GPU compaction: %zu remaining points ready for quadric detection", rem_count);

        // ========== Step 4: 二次曲面检测（零拷贝）==========
        auto quadric_detect_start = std::chrono::high_resolution_clock::now();
        bool quadric_success = quadric_detector_->processCloudDirect(d_rem_ptr, rem_count);
        auto quadric_detect_end = std::chrono::high_resolution_clock::now();
        float quadric_detect_time = std::chrono::duration<float, std::milli>(
            quadric_detect_end - quadric_detect_start).count();

        ROS_INFO("Quadric detection (zero-copy): %.2f ms", quadric_detect_time);

        // ========== Step 5: 安全同步与释放 ==========
        cudaError_t err = cudaStreamSynchronize(unified_stream_);
        if (err != cudaSuccess)
        {
            ROS_ERROR("CUDA stream synchronization failed: %s", cudaGetErrorString(err));
            plane_detector_->releaseExternalBuffer();
            return;
        }

        // 全链路总计时结束（在同步之后，确保所有 GPU 操作完成）
        auto total_end = std::chrono::high_resolution_clock::now();
        float total_ms = std::chrono::duration<float, std::milli>(total_end - total_start).count();

        // 释放外部显存缓冲区
        plane_detector_->releaseExternalBuffer();

        // ========== Step 6: 结果输出（详细计时）==========
        auto result_start = std::chrono::high_resolution_clock::now();
        
        // 获取平面检测结果
        auto get_planes_start = std::chrono::high_resolution_clock::now();
        const auto &detected_planes = plane_detector_->getDetectedPrimitives();
        auto get_planes_end = std::chrono::high_resolution_clock::now();
        float get_planes_time = std::chrono::duration<float, std::milli>(get_planes_end - get_planes_start).count();
        
        ROS_INFO("=== PLANE DETECTION RESULTS ===");
        ROS_INFO("Number of planes detected: %zu", detected_planes.size());

        // 输出每个平面的参数（访问 inliers->size() 可能触发数据传输）
        auto plane_log_start = std::chrono::high_resolution_clock::now();
        float total_plane_inlier_access_time = 0.0f;
        for (size_t i = 0; i < detected_planes.size(); ++i)
        {
            const auto &plane = detected_planes[i];
            ROS_INFO("Plane %zu:", i + 1);
            ROS_INFO("  Equation: %.4fx + %.4fy + %.4fz + %.4f = 0",
                     plane.model_coefficients[0], plane.model_coefficients[1],
                     plane.model_coefficients[2], plane.model_coefficients[3]);
            auto inlier_access_start = std::chrono::high_resolution_clock::now();
            size_t inlier_count = plane.inliers->size();
            auto inlier_access_end = std::chrono::high_resolution_clock::now();
            float inlier_access_time = std::chrono::duration<float, std::milli>(inlier_access_end - inlier_access_start).count();
            total_plane_inlier_access_time += inlier_access_time;
            ROS_INFO("  Inliers: %zu points", inlier_count);
            if (inlier_access_time > 1.0f) {
                ROS_INFO("    [WARNING] 访问内点数据耗时: %.2f ms (可能触发GPU->CPU传输)", inlier_access_time);
            }
        }
        auto plane_log_end = std::chrono::high_resolution_clock::now();
        float plane_log_time = std::chrono::duration<float, std::milli>(plane_log_end - plane_log_start).count();

        // 获取二次曲面检测结果
        float get_quadrics_time = 0.0f;
        float quadric_log_time = 0.0f;
        float total_quadric_inlier_access_time = 0.0f;
        if (quadric_success)
        {
            auto get_quadrics_start = std::chrono::high_resolution_clock::now();
            const auto &detected_quadrics = quadric_detector_->getDetectedPrimitives();
            auto get_quadrics_end = std::chrono::high_resolution_clock::now();
            get_quadrics_time = std::chrono::duration<float, std::milli>(get_quadrics_end - get_quadrics_start).count();
            
            ROS_INFO("=== QUADRIC DETECTION RESULTS ===");
            ROS_INFO("Number of quadrics detected: %zu", detected_quadrics.size());

            auto quadric_log_start = std::chrono::high_resolution_clock::now();
            for (size_t i = 0; i < detected_quadrics.size(); ++i)
            {
                const auto &quadric = detected_quadrics[i];
                auto quadric_inlier_start = std::chrono::high_resolution_clock::now();
                size_t inlier_count = quadric.inliers->size();
                auto quadric_inlier_end = std::chrono::high_resolution_clock::now();
                float quadric_inlier_time = std::chrono::duration<float, std::milli>(quadric_inlier_end - quadric_inlier_start).count();
                total_quadric_inlier_access_time += quadric_inlier_time;
                ROS_INFO("Quadric %zu: %zu inliers", i + 1, inlier_count);
                if (quadric_inlier_time > 1.0f) {
                    ROS_INFO("    [WARNING] 访问内点数据耗时: %.2f ms (可能触发GPU->CPU传输)", quadric_inlier_time);
                }
            }
            auto quadric_log_end = std::chrono::high_resolution_clock::now();
            quadric_log_time = std::chrono::duration<float, std::milli>(quadric_log_end - quadric_log_start).count();
        }
        
        auto result_end = std::chrono::high_resolution_clock::now();
        float result_time = std::chrono::duration<float, std::milli>(result_end - result_start).count();

        // ========== 详细时间分解 ==========
        std::cout << "----------------------------------------" << std::endl;
        std::cout << "  [详细时间分解]" << std::endl;
        std::cout << "  - ROS消息转换: " << std::fixed << std::setprecision(2) << ros_convert_time << " ms" << std::endl;
        std::cout << "  - 点云格式转换: " << std::fixed << std::setprecision(2) << format_convert_time << " ms" << std::endl;
        std::cout << "  - GPU预处理: " << std::fixed << std::setprecision(2) << gpu_preprocess_time << " ms" << std::endl;
        std::cout << "  - 平面检测: " << std::fixed << std::setprecision(2) << plane_detect_time << " ms" << std::endl;
        std::cout << "  - 二次曲面检测: " << std::fixed << std::setprecision(2) << quadric_detect_time << " ms" << std::endl;
        std::cout << "  - 获取平面结果: " << std::fixed << std::setprecision(2) << get_planes_time << " ms" << std::endl;
        std::cout << "  - 平面内点访问总时间: " << std::fixed << std::setprecision(2) << total_plane_inlier_access_time << " ms" << std::endl;
        std::cout << "  - 平面日志输出: " << std::fixed << std::setprecision(2) << plane_log_time << " ms" << std::endl;
        std::cout << "  - 获取二次曲面结果: " << std::fixed << std::setprecision(2) << get_quadrics_time << " ms" << std::endl;
        std::cout << "  - 二次曲面内点访问总时间: " << std::fixed << std::setprecision(2) << total_quadric_inlier_access_time << " ms" << std::endl;
        std::cout << "  - 二次曲面日志输出: " << std::fixed << std::setprecision(2) << quadric_log_time << " ms" << std::endl;
        std::cout << "  - 结果输出总时间: " << std::fixed << std::setprecision(2) << result_time << " ms" << std::endl;
        float accounted_time = ros_convert_time + format_convert_time + gpu_preprocess_time + 
                               plane_detect_time + quadric_detect_time + result_time;
        float unaccounted_time = total_ms - accounted_time;
        std::cout << "  - 已统计时间: " << std::fixed << std::setprecision(2) << accounted_time << " ms" << std::endl;
        std::cout << "  - 未统计时间: " << std::fixed << std::setprecision(2) << unaccounted_time << " ms" << std::endl;
        std::cout << "  [全链路总耗时] TOTAL LATENCY: " << std::fixed << std::setprecision(2) << total_ms << " ms" << std::endl;
        std::cout << "----------------------------------------" << std::endl;
    }

    void visualizePlanes(const std::vector<DetectedPrimitive<pcl::PointXYZI>> &planes,
                         const std_msgs::Header &header)
    {
        if (!(enable_visualization_ && enable_plane_visualization_)) return;
        visualization_msgs::MarkerArray marker_array;

        // 清除之前的标记
        visualization_msgs::Marker clear_marker;
        clear_marker.header = header;
        clear_marker.header.frame_id = output_frame_;
        clear_marker.ns = "planes";
        clear_marker.action = visualization_msgs::Marker::DELETEALL;
        marker_array.markers.push_back(clear_marker);

        // 调色板（tab10）备用
        auto tab10 = [&](size_t idx){
            static const float C[10][3] = {
                {0.1216f, 0.4667f, 0.7059f}, // 蓝
                {1.0000f, 0.4980f, 0.0549f}, // 橙
                {0.1725f, 0.6275f, 0.1725f}, // 绿
                {0.8392f, 0.1529f, 0.1569f}, // 红
                {0.5804f, 0.4039f, 0.7412f}, // 紫
                {0.5490f, 0.3373f, 0.2941f}, // 棕
                {0.8902f, 0.4667f, 0.7608f}, // 粉
                {0.4980f, 0.4980f, 0.4980f}, // 灰
                {0.7373f, 0.7412f, 0.1333f}, // 橄榄
                {0.0902f, 0.7451f, 0.8118f}  // 青
            };
            const float *c = C[idx % 10];
            return std::array<float,3>{c[0], c[1], c[2]};
        };

        auto chooseColor = [&](size_t i){
            if (plane_color_scheme_ == "tab10") return tab10(i);
            // hsv/pastel: 使用现有 hsvToRgb，并调饱和度
            float hue = (float)i / std::max<size_t>(1, planes.size()) * 360.0f;
            if (plane_color_scheme_ == "hsv") return hsvToRgb(hue, 0.9f, 0.95f);
            // pastel 默认：低饱和高明度
            return hsvToRgb(hue, 0.35f, 0.95f);
        };

        // 为每个平面创建可视化标记
        for (size_t i = 0; i < planes.size(); ++i)
        {
            const auto &plane = planes[i];

            // 创建平面标记（使用三角网格）
            visualization_msgs::Marker plane_marker;
            plane_marker.header = header;
            plane_marker.header.frame_id = output_frame_;
            plane_marker.ns = "planes";
            plane_marker.id = i;
            plane_marker.type = visualization_msgs::Marker::TRIANGLE_LIST;
            plane_marker.action = visualization_msgs::Marker::ADD;

            // 计算平面的可视化网格（优先使用内点凸包裁剪）
            bool hull_done = false;
            if (plane_clip_to_hull_)
            {
                hull_done = generatePlaneVisualizationHull(plane, plane_marker);
            }
            if (!hull_done)
            {
                // 回退到矩形网格
                generatePlaneVisualization(plane, plane_marker);
            }

            // 设置颜色（更美观的配色与透明度）
            auto base = chooseColor(i);
            plane_marker.color.r = base[0];
            plane_marker.color.g = base[1];
            plane_marker.color.b = base[2];
            plane_marker.color.a = static_cast<float>(std::max(0.0, std::min(1.0, plane_alpha_)));

            plane_marker.scale.x = 1.0;
            plane_marker.scale.y = 1.0;
            plane_marker.scale.z = 1.0;

            // 可选：棋盘纹理（基于平面局部UV网格的方格着色，避免“辐射状”）
            if (plane_checkerboard_ && !plane_marker.points.empty())
            {
                // 计算局部平面坐标系与参考中心
                const float A = plane.model_coefficients[0];
                const float B = plane.model_coefficients[1];
                const float C = plane.model_coefficients[2];
                const float nlen = std::sqrt(A*A + B*B + C*C);
                Eigen::Vector3f n(0,0,1);
                if (nlen > 1e-6f) n = Eigen::Vector3f(A/nlen, B/nlen, C/nlen);
                // 用内点质心作为原点
                Eigen::Vector3f p0(0,0,0);
                for (const auto &pt : plane.inliers->points) p0 += Eigen::Vector3f(pt.x, pt.y, pt.z);
                if (!plane.inliers->empty()) p0 /= static_cast<float>(plane.inliers->size());
                // 构建 (u,v)
                Eigen::Vector3f ref = (std::fabs(n.z()) < 0.9f) ? Eigen::Vector3f(0,0,1) : Eigen::Vector3f(1,0,0);
                Eigen::Vector3f u = n.cross(ref); float ul=u.norm(); if (ul>1e-6f) u/=ul; else u=Eigen::Vector3f(1,0,0);
                Eigen::Vector3f v = n.cross(u); v.normalize();

                // 扫描当前网格三角形顶点的UV范围
                float minU=std::numeric_limits<float>::max(), minV=std::numeric_limits<float>::max();
                float maxU=-std::numeric_limits<float>::max(), maxV=-std::numeric_limits<float>::max();
                for (const auto &gp : plane_marker.points)
                {
                    Eigen::Vector3f P(gp.x, gp.y, gp.z); Eigen::Vector3f d = P - p0;
                    float uu = u.dot(d), vv = v.dot(d);
                    minU = std::min(minU, uu); maxU = std::max(maxU, uu);
                    minV = std::min(minV, vv); maxV = std::max(maxV, vv);
                }
                float rangeU = std::max(1e-6f, maxU - minU);
                float rangeV = std::max(1e-6f, maxV - minV);
                int cells = std::max(4, plane_grid_size_); // 与网格密度一致
                float cell = std::max(rangeU, rangeV) / cells; // 方格边长
                if (cell < 1e-6f) cell = 1.0f; // 兜底

                // 着色：以三角形质心所在UV格子的奇偶决定浅/深色
                std::array<float,3> shadeA = base;
                std::array<float,3> shadeB = {base[0]*0.85f, base[1]*0.85f, base[2]*0.85f};
                plane_marker.colors.clear();
                plane_marker.colors.reserve(plane_marker.points.size());
                for (size_t p = 0; p + 2 < plane_marker.points.size(); p += 3)
                {
                    // 质心
                    Eigen::Vector3f P1(plane_marker.points[p].x, plane_marker.points[p].y, plane_marker.points[p].z);
                    Eigen::Vector3f P2(plane_marker.points[p+1].x, plane_marker.points[p+1].y, plane_marker.points[p+1].z);
                    Eigen::Vector3f P3(plane_marker.points[p+2].x, plane_marker.points[p+2].y, plane_marker.points[p+2].z);
                    Eigen::Vector3f Pc = (P1 + P2 + P3) / 3.0f; Eigen::Vector3f d = Pc - p0;
                    float uu = u.dot(d), vv = v.dot(d);
                    int iu = static_cast<int>(std::floor((uu - minU) / cell));
                    int iv = static_cast<int>(std::floor((vv - minV) / cell));
                    bool alt = ((iu + iv) & 1) != 0;
                    const auto &c = alt ? shadeB : shadeA;
                    std_msgs::ColorRGBA col; col.r=c[0]; col.g=c[1]; col.b=c[2]; col.a=plane_marker.color.a;
                    plane_marker.colors.push_back(col);
                    plane_marker.colors.push_back(col);
                    plane_marker.colors.push_back(col);
                }
            }

            marker_array.markers.push_back(plane_marker);

            // 创建法向量箭头
            visualization_msgs::Marker normal_marker;
            normal_marker.header = header;
            normal_marker.header.frame_id = output_frame_;
            normal_marker.ns = "normals";
            normal_marker.id = i;
            normal_marker.type = visualization_msgs::Marker::ARROW;
            normal_marker.action = visualization_msgs::Marker::ADD;

            // 计算平面中心和法向量
            generateNormalVisualization(plane, normal_marker);

            // 法向量颜色（更亮）
            normal_marker.color.r = base[0];
            normal_marker.color.g = base[1];
            normal_marker.color.b = base[2];
            normal_marker.color.a = 1.0f;

            normal_marker.scale.x = 0.02; // 箭头轴直径
            normal_marker.scale.y = 0.04; // 箭头头部直径
            normal_marker.scale.z = 0.06; // 箭头头部长度

            marker_array.markers.push_back(normal_marker);
        }

        plane_marker_pub_.publish(marker_array);
        ROS_INFO("Published %zu plane markers", planes.size());
    }

    void generatePlaneVisualization(const DetectedPrimitive<pcl::PointXYZI> &plane,
                                    visualization_msgs::Marker &marker)
    {
        if (plane.inliers->empty())
            return;

        // 计算内点的包围盒
        float min_x = std::numeric_limits<float>::max();
        float max_x = std::numeric_limits<float>::lowest();
        float min_y = std::numeric_limits<float>::max();
        float max_y = std::numeric_limits<float>::lowest();
        float min_z = std::numeric_limits<float>::max();
        float max_z = std::numeric_limits<float>::lowest();

        for (const auto &point : plane.inliers->points)
        {
            min_x = std::min(min_x, point.x);
            max_x = std::max(max_x, point.x);
            min_y = std::min(min_y, point.y);
            max_y = std::max(max_y, point.y);
            min_z = std::min(min_z, point.z);
            max_z = std::max(max_z, point.z);
        }

        // 扩展包围盒
        float padding = 0.1f;
        min_x -= padding;
        max_x += padding;
        min_y -= padding;
        max_y += padding;
        min_z -= padding;
        max_z += padding;

        // 平面参数
        float A = plane.model_coefficients[0];
        float B = plane.model_coefficients[1];
        float C = plane.model_coefficients[2];
        float D = plane.model_coefficients[3];

        // 改进的网格生成：使用更高分辨率和规整的网格
    int grid_size = std::max(4, plane_grid_size_); // 可配置网格分辨率

        // 创建规整的网格点矩阵
        std::vector<std::vector<geometry_msgs::Point>> grid_matrix(grid_size + 1,
                                                                   std::vector<geometry_msgs::Point>(grid_size + 1));
        std::vector<std::vector<bool>> valid_points(grid_size + 1,
                                                    std::vector<bool>(grid_size + 1, false));

        // 生成规整网格
        for (int i = 0; i <= grid_size; ++i)
        {
            for (int j = 0; j <= grid_size; ++j)
            {
                float u = (float)i / grid_size;
                float v = (float)j / grid_size;

                // 在包围盒中插值
                float x = min_x + u * (max_x - min_x);
                float y = min_y + v * (max_y - min_y);

                // 根据平面方程计算z坐标
                if (std::abs(C) > 1e-6)
                {
                    float z = -(A * x + B * y + D) / C;

                    // 检查z是否在合理范围内
                    if (z >= min_z && z <= max_z)
                    {
                        geometry_msgs::Point p;
                        p.x = x;
                        p.y = y;
                        p.z = z;
                        grid_matrix[i][j] = p;
                        valid_points[i][j] = true;
                    }
                }
            }
        }

        // 改进的三角形生成：只连接有效的相邻点
        // 计算平面法向量用于统一方向
        geometry_msgs::Vector3 normal;
        float norm = std::sqrt(A * A + B * B + C * C);
        if (norm > 1e-6)
        {
            normal.x = A / norm;
            normal.y = B / norm;
            normal.z = C / norm;
        }
        else
        {
            normal.x = 0;
            normal.y = 0;
            normal.z = 1;
        }

        for (int i = 0; i < grid_size; ++i)
        {
            for (int j = 0; j < grid_size; ++j)
            {
                // 检查四个角点是否都有效
                if (valid_points[i][j] && valid_points[i][j + 1] &&
                    valid_points[i + 1][j] && valid_points[i + 1][j + 1])
                {
                    // 第一个三角形 (确保逆时针方向)
                    marker.points.push_back(grid_matrix[i][j]);
                    marker.points.push_back(grid_matrix[i][j + 1]);
                    marker.points.push_back(grid_matrix[i + 1][j]);

                    // 第二个三角形 (确保逆时针方向)
                    marker.points.push_back(grid_matrix[i][j + 1]);
                    marker.points.push_back(grid_matrix[i + 1][j + 1]);
                    marker.points.push_back(grid_matrix[i + 1][j]);
                }
            }
        }
    }

    // 使用内点的2D凸包生成裁剪后的平面三角形网格。成功返回true。
    bool generatePlaneVisualizationHull(const DetectedPrimitive<pcl::PointXYZI> &plane,
                                        visualization_msgs::Marker &marker)
    {
        if (!plane.inliers || plane.inliers->size() < 3) return false;

        // 平面法向量
        const float A = plane.model_coefficients[0];
        const float B = plane.model_coefficients[1];
        const float C = plane.model_coefficients[2];
        const float D = plane.model_coefficients[3];
        (void)D; // 未使用但保留
        const float nlen = std::sqrt(A*A + B*B + C*C);
        if (nlen < 1e-6f) return false;
        Eigen::Vector3f n(A/nlen, B/nlen, C/nlen);

        // 质心
        Eigen::Vector3f p0(0,0,0);
        for (const auto &pt : plane.inliers->points) p0 += Eigen::Vector3f(pt.x, pt.y, pt.z);
        p0 /= static_cast<float>(plane.inliers->size());

        // 平面内正交基 (u, v)
        Eigen::Vector3f ref = (std::fabs(n.z()) < 0.9f) ? Eigen::Vector3f(0,0,1) : Eigen::Vector3f(1,0,0);
        Eigen::Vector3f u = n.cross(ref);
        float ul = u.norm(); if (ul < 1e-6f) return false; u /= ul;
        Eigen::Vector3f v = n.cross(u); v.normalize();

        // 投影到2D
        struct P2{ float x,y; };
        std::vector<P2> pts; pts.reserve(plane.inliers->size());
        for (const auto &pt : plane.inliers->points)
        {
            Eigen::Vector3f d(pt.x, pt.y, pt.z); d -= p0;
            pts.push_back(P2{u.dot(d), v.dot(d)});
        }
        if (pts.size() < 3) return false;

        // 单调链凸包
        auto cross = [](const P2 &O, const P2 &A, const P2 &B){
            return (A.x - O.x)*(B.y - O.y) - (A.y - O.y)*(B.x - O.x);
        };
        std::sort(pts.begin(), pts.end(), [](const P2&a, const P2&b){ if (a.x==b.x) return a.y<b.y; return a.x<b.x; });
        pts.erase(std::unique(pts.begin(), pts.end(), [](const P2&a, const P2&b){ return a.x==b.x && a.y==b.y; }), pts.end());
        if (pts.size() < 3) return false;
        std::vector<P2> H; H.reserve(pts.size()*2);
        for (const auto &p : pts){ while (H.size()>=2 && cross(H[H.size()-2], H[H.size()-1], p) <= 0) H.pop_back(); H.push_back(p); }
        size_t t = H.size()+1;
        for (int i = (int)pts.size()-2; i>=0; --i){ const auto &p = pts[i]; while (H.size()>=t && cross(H[H.size()-2], H[H.size()-1], p) <= 0) H.pop_back(); H.push_back(p); }
        if (!H.empty()) H.pop_back();
        if (H.size() < 3) return false;

        // 轻度平滑（邻点均值）与外扩 padding
        double smooth = std::max(0.0, std::min(1.0, plane_hull_smooth_factor_));
        double pad = std::max(0.0, plane_hull_padding_);
        // 质心（2D）
        P2 c{0.f,0.f}; for (const auto &q : H){ c.x += q.x; c.y += q.y; } c.x/=H.size(); c.y/=H.size();
        if (smooth > 0.0){
            std::vector<P2> S = H; S.reserve(H.size());
            for (size_t i=0;i<H.size();++i){ const auto &pr = H[(i+H.size()-1)%H.size()]; const auto &nx = H[(i+1)%H.size()];
                P2 avg{ (pr.x+nx.x)*0.5f, (pr.y+nx.y)*0.5f };
                S[i].x = (float)((1.0 - smooth)*H[i].x + smooth*avg.x);
                S[i].y = (float)((1.0 - smooth)*H[i].y + smooth*avg.y);
            }
            H.swap(S);
        }
        if (pad > 1e-6){
            for (auto &q : H){ float dx=q.x-c.x, dy=q.y-c.y; float L=std::sqrt(dx*dx+dy*dy); if (L>1e-6f){ q.x += (float)(pad*dx/L); q.y += (float)(pad*dy/L);} }
        }

        // 网格裁剪到凸包：在 (u,v) 上生成规整网格，仅保留多边形内部的网格三角
        auto to3D = [&](const P2 &q){ Eigen::Vector3f p = p0 + u*q.x + v*q.y; geometry_msgs::Point g; g.x=p.x(); g.y=p.y(); g.z=p.z(); return g; };

        // 计算凸包2D包围盒
        float minU = H[0].x, maxU = H[0].x, minV = H[0].y, maxV = H[0].y;
        for (const auto &q : H){
            minU = std::min(minU, q.x); maxU = std::max(maxU, q.x);
            minV = std::min(minV, q.y); maxV = std::max(maxV, q.y);
        }
        float rangeU = maxU - minU, rangeV = maxV - minV;
        if (rangeU < 1e-6f || rangeV < 1e-6f) return false;

        // 点在多边形内测试（射线法）
        auto pointInPoly = [&](const std::vector<P2> &poly, const P2 &p){
            bool inside = false; size_t n = poly.size();
            for (size_t i=0, j=n-1; i<n; j=i++){
                const P2 &pi = poly[i], &pj = poly[j];
                bool inter = ((pi.y>p.y) != (pj.y>p.y)) &&
                             (p.x < (pj.x - pi.x) * (p.y - pi.y) / ((pj.y - pi.y) + 1e-12f) + pi.x);
                if (inter) inside = !inside;
            }
            return inside;
        };

        int grid = std::max(4, plane_grid_size_);
        float du = rangeU / grid;
        float dv = rangeV / grid;
        if (du < 1e-8f || dv < 1e-8f) return false;

        // 规则网格
        std::vector<std::vector<P2>> G(grid+1, std::vector<P2>(grid+1));
        for (int i=0; i<=grid; ++i){
            for (int j=0; j<=grid; ++j){
                G[i][j].x = minU + du * i;
                G[i][j].y = minV + dv * j;
            }
        }

        // 对每个网格单元的两三角，若质心在多边形内则保留
        for (int i=0; i<grid; ++i){
            for (int j=0; j<grid; ++j){
                P2 t1a = G[i][j], t1b = G[i+1][j], t1c = G[i][j+1];
                P2 c1{ (t1a.x+t1b.x+t1c.x)/3.0f, (t1a.y+t1b.y+t1c.y)/3.0f };
                if (pointInPoly(H, c1)){
                    marker.points.push_back(to3D(t1a));
                    marker.points.push_back(to3D(t1b));
                    marker.points.push_back(to3D(t1c));
                }
                P2 t2a = G[i+1][j], t2b = G[i+1][j+1], t2c = G[i][j+1];
                P2 c2{ (t2a.x+t2b.x+t2c.x)/3.0f, (t2a.y+t2b.y+t2c.y)/3.0f };
                if (pointInPoly(H, c2)){
                    marker.points.push_back(to3D(t2a));
                    marker.points.push_back(to3D(t2b));
                    marker.points.push_back(to3D(t2c));
                }
            }
        }

        return !marker.points.empty();
    }

    void generateNormalVisualization(const DetectedPrimitive<pcl::PointXYZI> &plane,
                                     visualization_msgs::Marker &marker)
    {
        if (plane.inliers->empty())
            return;

        // 计算平面中心
        float cx = 0, cy = 0, cz = 0;
        for (const auto &point : plane.inliers->points)
        {
            cx += point.x;
            cy += point.y;
            cz += point.z;
        }
        cx /= plane.inliers->size();
        cy /= plane.inliers->size();
        cz /= plane.inliers->size();

        // 法向量
        float nx = plane.model_coefficients[0];
        float ny = plane.model_coefficients[1];
        float nz = plane.model_coefficients[2];

        // 箭头起点和终点
        geometry_msgs::Point start, end;
        start.x = cx;
        start.y = cy;
        start.z = cz;
        end.x = cx + nx * 0.3; // 法向量长度0.3m
        end.y = cy + ny * 0.3;
        end.z = cz + nz * 0.3;

        marker.points.push_back(start);
        marker.points.push_back(end);
    }

    std::array<float, 3> hsvToRgb(float h, float s, float v)
    {
        float c = v * s;
        float x = c * (1 - std::abs(fmod(h / 60.0, 2) - 1));
        float m = v - c;

        float r, g, b;
        if (h >= 0 && h < 60)
        {
            r = c;
            g = x;
            b = 0;
        }
        else if (h >= 60 && h < 120)
        {
            r = x;
            g = c;
            b = 0;
        }
        else if (h >= 120 && h < 180)
        {
            r = 0;
            g = c;
            b = x;
        }
        else if (h >= 180 && h < 240)
        {
            r = 0;
            g = x;
            b = c;
        }
        else if (h >= 240 && h < 300)
        {
            r = x;
            g = 0;
            b = c;
        }
        else
        {
            r = c;
            g = 0;
            b = x;
        }

        return {r + m, g + m, b + m};
    }

    void publishRemainingCloud(const std_msgs::Header &header,
                               const pcl::PointCloud<pcl::PointXYZI>::Ptr &remaining_cloud)
    {
        if (remaining_cloud && !remaining_cloud->empty())
        {
            sensor_msgs::PointCloud2 msg;
            pcl::toROSMsg(*remaining_cloud, msg);
            msg.header = header;
            msg.header.frame_id = output_frame_;
            result_cloud_pub_.publish(msg);

            ROS_INFO("Published remaining cloud with %zu points", remaining_cloud->size());
        }
    }

    void processSupervoxels(const std_msgs::Header &header,
                            const pcl::PointCloud<pcl::PointXYZI>::Ptr &remaining_cloud)
    {
        // 使用传入的剩余点云进行超体素处理
        if (!remaining_cloud || remaining_cloud->empty())
        {
            ROS_WARN("No remaining cloud for supervoxel processing");
            return;
        }

        if (static_cast<int>(remaining_cloud->size()) < min_remaining_points_for_supervoxel_)
        {
            ROS_INFO("Remaining cloud too small (%zu points < %d threshold), skipping supervoxel",
                     remaining_cloud->size(), min_remaining_points_for_supervoxel_);
            return;
        }

        ROS_INFO("=== SUPERVOXEL PROCESSING ===");
        ROS_INFO("Input remaining cloud: %zu points", remaining_cloud->size());

        // 2. 执行超体素分割
        // auto sv_start = std::chrono::high_resolution_clock::now();
        bool success = sv_processor_->processPointCloud(remaining_cloud);
        // auto sv_end = std::chrono::high_resolution_clock::now();
        // auto sv_duration = std::chrono::duration_cast<std::chrono::milliseconds>(sv_end - sv_start);

        if (!success)
        {
            ROS_ERROR("Supervoxel processing failed");
            return;
        }

        // 3. 获取结果
        const auto &convex_hulls = sv_processor_->getConvexHulls();
        const auto &stats = sv_processor_->getProcessingStats();

        // ROS_INFO("Supervoxel processing time: %ld ms", sv_duration.count());
        ROS_INFO("Total supervoxels: %zu", stats.total_supervoxels);
        ROS_INFO("Valid convex hulls: %zu", stats.valid_convex_hulls);
        if (stats.valid_convex_hulls > 0)
        {
            ROS_INFO("Avg points per hull: %.1f", stats.getAvgPointsPerHull());
            // ROS_INFO("Avg time per hull: %.2f ms", stats.getAvgTimePerHull());
        }

        // 4. 可视化凸包（受开关控制）
        if (enable_visualization_ && enable_convex_hull_visualization_)
        {
            visualizeConvexHulls(convex_hulls, header);
        }
    }

    void visualizeConvexHulls(const std::vector<super_voxel::ConvexHullData> &hulls,
                              const std_msgs::Header &header)
    {
        visualization_msgs::MarkerArray marker_array;

        // 清除旧的凸包标记
        visualization_msgs::Marker clear_marker;
        clear_marker.header = header;
        clear_marker.header.frame_id = output_frame_;
        clear_marker.ns = "convex_hulls";
        clear_marker.action = visualization_msgs::Marker::DELETEALL;
        marker_array.markers.push_back(clear_marker);

        // 为每个凸包创建轮廓线
        for (size_t i = 0; i < hulls.size(); ++i)
        {
            const auto &hull = hulls[i];

            if (!hull.hull_points || hull.hull_points->empty())
            {
                continue;
            }

            visualization_msgs::Marker hull_marker;
            hull_marker.header = header;
            hull_marker.header.frame_id = output_frame_;
            hull_marker.ns = "convex_hulls";
            hull_marker.id = i;
            hull_marker.type = visualization_msgs::Marker::LINE_STRIP;
            hull_marker.action = visualization_msgs::Marker::ADD;

            // 添加凸包顶点（闭合轮廓）
            for (const auto &pt : hull.hull_points->points)
            {
                geometry_msgs::Point p;
                p.x = pt.x;
                p.y = pt.y;
                p.z = pt.z;
                hull_marker.points.push_back(p);
            }

            // 闭合轮廓（首尾相连）
            if (!hull.hull_points->empty())
            {
                geometry_msgs::Point first;
                first.x = hull.hull_points->points[0].x;
                first.y = hull.hull_points->points[0].y;
                first.z = hull.hull_points->points[0].z;
                hull_marker.points.push_back(first);
            }

            // 设置颜色（按 supervoxel_id 散列）
            float hue = fmod(hull.supervoxel_id * 137.5f, 360.0f); // 黄金角散列
            auto rgb = hsvToRgb(hue, 0.9f, 1.0f);

            hull_marker.color.r = rgb[0];
            hull_marker.color.g = rgb[1];
            hull_marker.color.b = rgb[2];
            hull_marker.color.a = 1.0f;

            hull_marker.scale.x = 0.01; // 线宽

            marker_array.markers.push_back(hull_marker);
        }

        convex_hull_marker_pub_.publish(marker_array); // 使用独立话题
        ROS_INFO("Published %zu convex hull markers", hulls.size());
    }
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "plane_supervoxel_node");
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");

    try
    {
        PlaneSupervoxelNode node(nh, pnh);
        ROS_INFO("Unified Detection Node started, waiting for point clouds...");
        ros::spin();
    }
    catch (const std::exception &e)
    {
        ROS_FATAL("Unified Detection Node failed: %s", e.what());
        return -1;
    }

    return 0;
}