#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/common/transforms.h>
#include <geometry_msgs/Point.h>
#include <std_msgs/ColorRGBA.h>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <Eigen/Dense>

#include "PlaneDetect/PlaneDetect.h"
#include "super_voxel/supervoxel.h"
#include <map>  // 用于帧同步

class PlaneSupervoxelNode
{
public:
    PlaneSupervoxelNode(ros::NodeHandle &nh, ros::NodeHandle &pnh)
        : nh_(nh), pnh_(pnh), plane_detector_(nullptr), current_frame_seq_(0)
    {
        // 读取参数
        loadParameters();

        // 初始化平面检测器
        initializePlaneDetector();

        // 初始化超体素处理器
        initializeSupervoxelProcessor();

        // 设置订阅和发布
        setupPubSub();

        ROS_INFO("PlaneSupervoxelNode (Node 1) initialized successfully");
    }

private:
    ros::NodeHandle nh_, pnh_;
    
    // 订阅者
    ros::Subscriber camera_sub_;           // 订阅原始点云 /camera/rgb/points
    ros::Subscriber quadric_result_sub_;   // 订阅二次曲面处理后的点云 /node2_output
    
    // 发布者
    ros::Publisher plane_remaining_pub_;   // 发布平面检测后的剩余点云给节点2
    ros::Publisher plane_marker_pub_;      // 平面可视化
    ros::Publisher convex_hull_marker_pub_;  // 凸包可视化
    ros::Publisher result_cloud_pub_;      // 最终结果点云

    std::unique_ptr<PlaneDetect<pcl::PointXYZRGB>> plane_detector_;
    DetectorParams detector_params_;

    // 超体素处理器
    std::unique_ptr<super_voxel::SupervoxelProcessor> sv_processor_;
    super_voxel::SupervoxelParams sv_params_;

    // 帧同步：存储待处理的点云（等待二次曲面节点处理完成）
    std::map<uint32_t, std_msgs::Header> pending_frames_;

    // 整体计时：存储每帧的开始时间
    std::map<uint32_t, std::chrono::high_resolution_clock::time_point> frame_start_times_;
    
    uint32_t current_frame_seq_;

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

    // 超体素功能开关
        pnh_.param("enable_supervoxel", enable_supervoxel_, false);
        pnh_.param("min_remaining_points_for_supervoxel", min_remaining_points_for_supervoxel_, 500);

    // 全局/分项可视化开关
    pnh_.param("enable_visualization", enable_visualization_, enable_visualization_);
    pnh_.param("enable_plane_visualization", enable_plane_visualization_, enable_plane_visualization_);
    pnh_.param("enable_convex_hull_visualization", enable_convex_hull_visualization_, enable_convex_hull_visualization_);

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
        plane_detector_ = std::make_unique<PlaneDetect<pcl::PointXYZRGB>>(detector_params_);
        ROS_INFO("PlaneDetector initialized with batch_size=%d", detector_params_.batch_size);
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
        
        //  订阅二次曲面节点的输出（队列=1，最小延迟）
        quadric_result_sub_ = nh_.subscribe("/node2_output", 1, &PlaneSupervoxelNode::quadricResultCallback, this);
        
        //  发布平面检测后的剩余点云给节点2（不使用 latched，实时发布）
        plane_remaining_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/plane_remaining", 1, false);
        
        // 发布平面可视化
        plane_marker_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("plane_markers", 1, true);
        
        // 发布凸包可视化
        convex_hull_marker_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("convex_hull_markers", 1, true);
        
        // 发布最终结果点云
        result_cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("remaining_cloud", 1, true);

        ROS_INFO("=== Node 1: PlaneSupervoxelNode ===");
        ROS_INFO("Subscribed to camera: %s", input_topic_.c_str());
        ROS_INFO("Subscribed to quadric result: /node2_output");
        ROS_INFO("Publishing plane remaining to: /plane_remaining");
        ROS_INFO("Publishing plane markers to: plane_markers");
        ROS_INFO("Publishing final result to: remaining_cloud");
    }

    // 回调1：处理相机原始点云，进行平面检测
    void cameraCallback(const sensor_msgs::PointCloud2::ConstPtr &msg)
    {
        //  开始整体计时
        auto frame_start_time = std::chrono::high_resolution_clock::now();
        
        ROS_INFO("Received point cloud with %d points", msg->width * msg->height);

        // 转换为PCL格式
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr input_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::fromROSMsg(*msg, *input_cloud);

        if (input_cloud->empty())
        {
            ROS_WARN("Received empty point cloud");
            return;
        }

        // 体素降采样（可选）
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr processed_cloud = input_cloud;
        if (enable_voxel_filter_)
        {
            processed_cloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
            pcl::VoxelGrid<pcl::PointXYZRGB> voxel_filter;
            voxel_filter.setInputCloud(input_cloud);
            voxel_filter.setLeafSize(voxel_leaf_size_, voxel_leaf_size_, voxel_leaf_size_);
            voxel_filter.filter(*processed_cloud);

            ROS_INFO("Voxel filtering: %zu -> %zu points (减少了%.1f%%)",
                     input_cloud->size(), processed_cloud->size(),
                     100.0 * (1.0 - (double)processed_cloud->size() / input_cloud->size()));
        }
        else
        {
            ROS_INFO("Voxel filtering: disabled, keeping %zu points", input_cloud->size());
        }

        // 离群点移除（可选）
        if (enable_outlier_removal_)
        {
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
            pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> outlier_filter;
            outlier_filter.setInputCloud(processed_cloud);
            outlier_filter.setMeanK(outlier_k_neighbors_);
            outlier_filter.setStddevMulThresh(outlier_std_dev_thresh_);
            outlier_filter.filter(*filtered_cloud);

            ROS_INFO("Outlier removal: %zu -> %zu points (移除了%zu个离群点)",
                     processed_cloud->size(), filtered_cloud->size(),
                     processed_cloud->size() - filtered_cloud->size());

            processed_cloud = filtered_cloud;
        }
        else
        {
            ROS_INFO("🧹 Outlier removal: disabled");
        }

    ROS_INFO("Final preprocessed cloud: %zu points ready for plane detection", processed_cloud->size());

        // 平面检测
        // auto start_time = std::chrono::high_resolution_clock::now();

        bool success = plane_detector_->processCloud(processed_cloud);

        // auto end_time = std::chrono::high_resolution_clock::now();
        // auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

        if (!success)
        {
            ROS_ERROR("Plane detection failed");
            return;
        }

        // 获取检测结果
        const auto &detected_planes = plane_detector_->getDetectedPrimitives();

        ROS_INFO("=== PLANE DETECTION RESULTS ===");
        // ROS_INFO("Total processing time: %ld ms", duration.count());
        ROS_INFO("Number of planes detected: %zu", detected_planes.size());

        // 输出每个平面的参数
        for (size_t i = 0; i < detected_planes.size(); ++i)
        {
            const auto &plane = detected_planes[i];
            ROS_INFO("Plane %zu:", i + 1);
            ROS_INFO("  Equation: %.4fx + %.4fy + %.4fz + %.4f = 0",
                     plane.model_coefficients[0], plane.model_coefficients[1],
                     plane.model_coefficients[2], plane.model_coefficients[3]);
            ROS_INFO("  Inliers: %zu points", plane.inliers->size());

            // 计算法向量模长验证
            float nx = plane.model_coefficients[0];
            float ny = plane.model_coefficients[1];
            float nz = plane.model_coefficients[2];
            float norm = sqrt(nx * nx + ny * ny + nz * nz);
            ROS_INFO("  Normal vector norm: %.6f (should be ~1.0)", norm);
        }

        // 可视化平面（受开关控制）
        if (enable_visualization_ && enable_plane_visualization_)
        {
            visualizePlanes(detected_planes, msg->header);
        }

        // 获取平面检测后的剩余点云
        auto remaining_cloud = plane_detector_->getFinalCloud();
        ROS_INFO("Points remaining after plane detection: %zu", remaining_cloud->size());

        // 发布剩余点云给节点2（二次曲面检测节点）
        sensor_msgs::PointCloud2 plane_remaining_msg;
        pcl::toROSMsg(*remaining_cloud, plane_remaining_msg);
        
        //  关键修复：完整保留原始 header（包括 stamp 时间戳）
        plane_remaining_msg.header.stamp = msg->header.stamp;      // 保留原始时间戳
        plane_remaining_msg.header.frame_id = output_frame_;       // 使用输出坐标系
        plane_remaining_msg.header.seq = current_frame_seq_++;     // 使用自增序列号用于帧同步
        
        // 保存帧信息，等待节点2返回
        pending_frames_[plane_remaining_msg.header.seq] = msg->header;
        // 记录该帧的起始时间用于整体计时（Plane -> Quadric(Node2) -> Supervoxel）
        frame_start_times_[plane_remaining_msg.header.seq] = frame_start_time;
        
        plane_remaining_pub_.publish(plane_remaining_msg);
        ROS_INFO("📤 [Node1] Published plane remaining cloud (%zu points) to /plane_remaining (frame %u)", 
                 remaining_cloud->size(), plane_remaining_msg.header.seq);
    }

    // 回调2：接收节点2处理后的点云，进行超体素处理
    void quadricResultCallback(const sensor_msgs::PointCloud2::ConstPtr &msg)
    {
        ROS_INFO("📥 [Node1] Received quadric result: %d points (frame %u)", 
                 msg->width * msg->height, msg->header.seq);
        
        // 检查帧是否匹配
        auto it = pending_frames_.find(msg->header.seq);
        if (it == pending_frames_.end())
        {
            ROS_WARN(" [Node1] Received quadric result for unknown frame %u, skipping", msg->header.seq);
            return;
        }
        
        // 转换点云
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr quadric_remaining(new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::fromROSMsg(*msg, *quadric_remaining);
        
        ROS_INFO("Points remaining after quadric detection: %zu", quadric_remaining->size());

        // 发布最终剩余点云
        publishRemainingCloud(msg->header, quadric_remaining);

        // 超体素处理
        if (enable_supervoxel_ && quadric_remaining->size() >= min_remaining_points_for_supervoxel_)
        {
            processSupervoxels(msg->header, quadric_remaining);
        }
        else if (enable_supervoxel_)
        {
            ROS_INFO("Skipping supervoxel processing: insufficient points (%zu < %d)",
                     quadric_remaining->size(), min_remaining_points_for_supervoxel_);
        }

        // ========== 结束整体计时 ==========
        auto frame_end_time = std::chrono::high_resolution_clock::now();
        auto it_time = frame_start_times_.find(msg->header.seq);
        if (it_time != frame_start_times_.end())
        {
            auto frame_duration = std::chrono::duration_cast<std::chrono::milliseconds>(frame_end_time - it_time->second);
            ROS_INFO("========================================");
            ROS_INFO("TOTAL FRAME PROCESSING TIME: %ld ms", frame_duration.count());
            ROS_INFO("========================================");
            // 移除时间记录
            frame_start_times_.erase(it_time);
        }
        else
        {
            ROS_WARN("[Node1] Missing start time for frame %u when computing total time", msg->header.seq);
        }

        // 清理已处理的帧
        pending_frames_.erase(it);
        ROS_INFO(" [Node1] Frame %u processing complete", msg->header.seq);
    }

    void visualizePlanes(const std::vector<DetectedPrimitive<pcl::PointXYZRGB>> &planes,
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

    void generatePlaneVisualization(const DetectedPrimitive<pcl::PointXYZRGB> &plane,
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
    bool generatePlaneVisualizationHull(const DetectedPrimitive<pcl::PointXYZRGB> &plane,
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

    void generateNormalVisualization(const DetectedPrimitive<pcl::PointXYZRGB> &plane,
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
                               const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &remaining_cloud)
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
                            const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &remaining_cloud)
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
        ROS_INFO(" [Node1] PlaneSupervoxelNode started, waiting for point clouds...");
        ros::spin();
    }
    catch (const std::exception &e)
    {
        ROS_FATAL("[Node1] PlaneSupervoxelNode failed: %s", e.what());
        return -1;
    }

    return 0;
}