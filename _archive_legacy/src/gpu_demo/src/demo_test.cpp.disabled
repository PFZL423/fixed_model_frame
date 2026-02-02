/**
 * @file demo_test.cpp
 * @brief GPU二次曲面检测器ROS测试节点
 *
 * 功能：
 * 1. 订阅/generated_cloud话题的点云数据
 * 2. 使用QuadricDetect类进行二次曲面检测
 * 3. 输出检测到的二次曲面参数和统计信息
 * 4. 发布结果点云用于可视化
 * 5. 🆕 只处理第一帧点云，后续忽略
 *
 * @author PFZL-423
 * @date 2025-08-23
 */

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <std_msgs/Header.h>

#include "gpu_demo/QuadricDetect.h"
#include <chrono>
#include <iomanip>
#include <sstream>

/**
 * @class QuadricDetectorNode
 * @brief GPU二次曲面检测器的ROS包装节点
 */
class QuadricDetectorNode
{
private:
    ros::NodeHandle nh_;         ///< ROS节点句柄
    ros::NodeHandle private_nh_; ///< 私有节点句柄

    // ROS通信
    ros::Subscriber cloud_sub_;    ///< 点云订阅器
    ros::Publisher inliers_pub_;   ///< 内点点云发布器
    ros::Publisher remaining_pub_; ///< 剩余点云发布器

    // 核心检测器
    std::unique_ptr<QuadricDetect> detector_; ///< GPU二次曲面检测器
    DetectorParams params_;                   ///< 检测参数

    // 统计信息
    int processed_clouds_;         ///< 已处理点云数量
    double total_processing_time_; ///< 总处理时间

    // 配置参数
    std::string input_topic_;  ///< 输入话题名
    std::string frame_id_;     ///< 坐标系ID
    bool publish_inliers_;     ///< 是否发布内点
    bool publish_remaining_;   ///< 是否发布剩余点云
    bool print_model_details_; ///< 是否打印模型详情
    int verbosity_;            ///< 输出详细级别

    // 🆕 一次性处理控制
    bool has_processed_; ///< 是否已经处理过点云
    
    // 🆕 添加计时器
    GPUTimer total_timer_; ///< 总体处理时间计时器

public:
    QuadricDetectorNode() : private_nh_("~"), processed_clouds_(0), total_processing_time_(0.0),
                            has_processed_(false) // 🔧 初始化为false
    {
        loadParameters();
        initializeDetector();
        setupROS();

        ROS_INFO("[QuadricDetectorNode] Initialization completed, waiting for point cloud data...");
        ROS_INFO("[QuadricDetectorNode] 🎯 Will process ONLY the first point cloud, then ignore subsequent ones");
        ROS_INFO("[QuadricDetectorNode] Subscribing to topic: %s", input_topic_.c_str());
    }

private:
    /**
     * @brief 从ROS参数服务器加载参数
     */
    void loadParameters()
    {
        // ROS话题配置
        private_nh_.param<std::string>("quadric_detector/input_topic", input_topic_, "/generated_cloud");
        private_nh_.param<std::string>("quadric_detector/output/frame_id", frame_id_, "base_link");
        private_nh_.param<bool>("quadric_detector/publish_inliers", publish_inliers_, true);
        private_nh_.param<bool>("quadric_detector/publish_remaining_cloud", publish_remaining_, true);

        // 核心RANSAC参数
        private_nh_.param<double>("quadric_detector/ransac/min_remaining_points_percentage",
                                  params_.min_remaining_points_percentage, 0.03);
        private_nh_.param<double>("quadric_detector/ransac/quadric_distance_threshold",
                                  params_.quadric_distance_threshold, 0.02);
        private_nh_.param<int>("quadric_detector/ransac/min_quadric_inlier_count_absolute",
                               params_.min_quadric_inlier_count_absolute, 500);
        private_nh_.param<int>("quadric_detector/ransac/quadric_max_iterations",
                               params_.quadric_max_iterations, 5000);
        private_nh_.param<double>("quadric_detector/ransac/min_quadric_inlier_percentage",
                                  params_.min_quadric_inlier_percentage, 0.05);

        // LO-RANSAC参数
        private_nh_.param<bool>("quadric_detector/lo_ransac/enable_local_optimization",
                                params_.enable_local_optimization, false);
        private_nh_.param<double>("quadric_detector/lo_ransac/lo_min_inlier_ratio",
                                  params_.lo_min_inlier_ratio, 0.6);
        private_nh_.param<double>("quadric_detector/lo_ransac/desired_prob",
                                  params_.desired_prob, 0.99);
        private_nh_.param<int>("quadric_detector/lo_ransac/lo_sample_size",
                               params_.lo_sample_size, 15);

        // 调试参数
        private_nh_.param<int>("quadric_detector/debug/verbosity", params_.verbosity, 1);
        private_nh_.param<bool>("quadric_detector/debug/print_model_details", print_model_details_, true);

        verbosity_ = params_.verbosity;

        ROS_INFO("[QuadricDetectorNode] Parameters loaded successfully:");
        ROS_INFO("  - Distance threshold: %.4f", params_.quadric_distance_threshold);
        ROS_INFO("  - Min inlier count: %d", params_.min_quadric_inlier_count_absolute);
        ROS_INFO("  - Min remaining points ratio: %.3f", params_.min_remaining_points_percentage);
        ROS_INFO("  - Verbosity level: %d", params_.verbosity);
    }

    /**
     * @brief 初始化GPU检测器
     */
    void initializeDetector()
    {
        try
        {
            detector_ = std::make_unique<QuadricDetect>(params_);
            ROS_INFO("[QuadricDetectorNode] GPU detector initialized successfully");
        }
        catch (const std::exception &e)
        {
            ROS_ERROR("[QuadricDetectorNode] GPU检测器初始化失败: %s", e.what());
            ros::shutdown();
        }
    }

    /**
     * @brief 设置ROS通信
     */
    void setupROS()
    {
        // 订阅点云
        cloud_sub_ = nh_.subscribe(input_topic_, 10, &QuadricDetectorNode::cloudCallback, this);

        // 发布结果 (可选)
        if (publish_inliers_)
        {
            inliers_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/quadric_detector/inliers", 5);
        }
        if (publish_remaining_)
        {
            remaining_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/quadric_detector/remaining", 5);
        }
    }

    /**
     * @brief 点云回调函数 - 🔧 修改为只处理一次
     */
    void cloudCallback(const sensor_msgs::PointCloud2::ConstPtr &msg)
    {
        total_timer_.start();
        // 🎯 关键修复：如果已经处理过，直接返回
        if (has_processed_)
        {
            if (verbosity_ > 1)
            {
                static int skip_count = 0;
                skip_count++;
                if (skip_count % 50 == 1)
                { // 每50次打印一次，避免刷屏
                    ROS_INFO("[QuadricDetectorNode] 🔄 Skipping point cloud #%d (already processed one)", skip_count);
                }
            }
            return;
        }

        // 🎯 标记为已处理（在实际处理之前，避免并发问题）
        has_processed_ = true;

        auto start_time = std::chrono::high_resolution_clock::now();

        ROS_INFO("========================================");
        ROS_INFO("[QuadricDetectorNode] 🎯 Processing THE ONLY point cloud (one-time mode)");
        ROS_INFO("  - Point count: %d", msg->width * msg->height);
        ROS_INFO("  - Frame ID: %s", msg->header.frame_id.c_str());
        ROS_INFO("  - Will ignore all subsequent point clouds");

        // 转换ROS消息为PCL点云
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::fromROSMsg(*msg, *cloud);

        if (cloud->empty())
        {
            ROS_WARN("[QuadricDetectorNode] Received empty point cloud, but still marking as processed");
            return;
        }

        // 🎯 核心：使用GPU检测器处理点云（只这一次）
        bool success = detector_->processCloud(cloud);

        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        double processing_time = duration.count() / 1000.0;
        double processing_time_a = total_timer_.stop();
        if (success)
        {
            // 获取检测结果
            const auto &detected_primitives = detector_->getDetectedPrimitives();

            // 输出检测结果
            outputDetectionResults(detected_primitives, processing_time);

            // 发布结果点云 (可选)
            publishResultClouds(msg->header, detected_primitives);

            ROS_INFO("✅ [QuadricDetectorNode] ONE-TIME processing completed successfully!");
        }
        else
        {
            ROS_WARN("[QuadricDetectorNode] ❌ ONE-TIME processing failed");
        }

        // 更新统计信息
        processed_clouds_ = 1; // 永远只处理1个
        total_processing_time_ = processing_time;
        ROS_INFO("  - Processing time: %.2f ms", processing_time_a);
        ROS_INFO("[QuadricDetectorNode] 🔒 Processing locked. Node will ignore all future point clouds.");
        ROS_INFO("[QuadricDetectorNode] 💡 To process another cloud, restart this node.");
        ROS_INFO("========================================");
    }

    /**
     * @brief 输出二次曲面检测结果
     */
    void outputDetectionResults(const std::vector<DetectedPrimitive, Eigen::aligned_allocator<DetectedPrimitive>> &primitives,
                                double processing_time)
    {
        ROS_INFO("🎯 Detection Results Summary:");
        ROS_INFO("  - Detected %lu quadric surfaces", primitives.size());
        ROS_INFO("  - Processing time: %.3f seconds", processing_time);

        if (primitives.empty())
        {
            ROS_WARN("  ❌ No valid quadric surfaces detected");
            return;
        }

        // Detailed output for each quadric surface
        for (size_t i = 0; i < primitives.size(); ++i)
        {
            const auto &primitive = primitives[i];

            ROS_INFO("📊 Quadric Surface #%lu:", i + 1);
            ROS_INFO("  - Type: %s", primitive.type.c_str());
            ROS_INFO("  - Inlier count: %lu", primitive.inliers->size());

            if (print_model_details_)
            {
                outputQuadricMatrix(primitive.model_coefficients, i + 1);
            }
        }

        // Statistics
        ROS_INFO("📈 Final Statistics:");
        ROS_INFO("  - Processing time: %.3f seconds", processing_time);
        ROS_INFO("  - This was the ONLY point cloud processed");
    }

    /**
     * @brief 输出二次曲面矩阵参数
     */
    void outputQuadricMatrix(const Eigen::Matrix4f &Q, int index)
    {
        ROS_INFO("  📋 Quadric Surface #%d Matrix Parameters (4×4):", index);

        std::stringstream ss;
        ss << std::fixed << std::setprecision(6);

        for (int i = 0; i < 4; ++i)
        {
            ss.str("");
            ss << "    [";
            for (int j = 0; j < 4; ++j)
            {
                ss << std::setw(10) << Q(i, j);
                if (j < 3)
                    ss << ", ";
            }
            ss << "]";
            ROS_INFO("%s", ss.str().c_str());
        }

        // Analyze quadric surface type (simple classification)
        analyzeQuadricType(Q, index);
    }

    /**
     * @brief Simple quadric surface type analysis
     */
    void analyzeQuadricType(const Eigen::Matrix4f &Q, int index)
    {
        // Extract quadratic term coefficients
        float a = Q(0, 0), b = Q(1, 1), c = Q(2, 2);
        float d = Q(0, 1), e = Q(0, 2), f = Q(1, 2);

        std::string type = "Unknown Type";

        // Simplified quadric surface classification
        if (std::abs(d) < 1e-6 && std::abs(e) < 1e-6 && std::abs(f) < 1e-6)
        {
            // Axis-aligned quadric surfaces
            if (a > 0 && b > 0 && c > 0)
            {
                type = "Ellipsoid";
            }
            else if (a > 0 && b > 0 && c < 0)
            {
                type = "Hyperboloid";
            }
            else if (a > 0 && b > 0 && std::abs(c) < 1e-6)
            {
                type = "Elliptic Cylinder";
            }
        }
        else
        {
            type = "Rotated Quadric";
        }

        ROS_INFO("  🔍 Type Analysis: %s", type.c_str());
        ROS_INFO("  📊 Main Coefficients: a=%.4f, b=%.4f, c=%.4f", a, b, c);
    }

    /**
     * @brief Publish result point clouds for visualization
     */
    void publishResultClouds(const std_msgs::Header &header,
                             const std::vector<DetectedPrimitive, Eigen::aligned_allocator<DetectedPrimitive>> &primitives)
    {
        if (!publish_inliers_ && !publish_remaining_)
        {
            return;
        }

        // Publish inlier point clouds (merge all quadric surface inliers)
        if (publish_inliers_ && !primitives.empty())
        {
            pcl::PointCloud<pcl::PointXYZ>::Ptr all_inliers(new pcl::PointCloud<pcl::PointXYZ>);

            for (const auto &primitive : primitives)
            {
                *all_inliers += *(primitive.inliers);
            }

            sensor_msgs::PointCloud2 inliers_msg;
            pcl::toROSMsg(*all_inliers, inliers_msg);
            inliers_msg.header = header;
            inliers_msg.header.frame_id = frame_id_;

            inliers_pub_.publish(inliers_msg);

            if (verbosity_ > 1)
            {
                ROS_INFO("📤 Published inlier point cloud: %lu points", all_inliers->size());
            }
        }

        // Publish remaining point cloud
        if (publish_remaining_)
        {
            auto remaining_cloud = detector_->getFinalCloud();

            sensor_msgs::PointCloud2 remaining_msg;
            pcl::toROSMsg(*remaining_cloud, remaining_msg);
            remaining_msg.header = header;
            remaining_msg.header.frame_id = frame_id_;

            remaining_pub_.publish(remaining_msg);

            if (verbosity_ > 1)
            {
                ROS_INFO("📤 Published remaining point cloud: %lu points", remaining_cloud->size());
            }
        }
    }
};

/**
 * @brief 主函数
 */
int main(int argc, char **argv)
{
    ros::init(argc, argv, "quadric_detector_test");

    try
    {
        QuadricDetectorNode node;

        ROS_INFO("🚀 GPU Quadric Detector Test Node Started Successfully!");
        ROS_INFO("💡 Usage:");
        ROS_INFO("   roslaunch gpu_demo demo_test.launch");
        ROS_INFO("   or publish point cloud to topic: /generated_cloud");
        ROS_INFO("🎯 Note: Will process ONLY the first point cloud received");

        ros::spin();
    }
    catch (const std::exception &e)
    {
        ROS_FATAL("Node startup failed: %s", e.what());
        return -1;
    }

    ROS_INFO("Node exited normally");
    return 0;
}
