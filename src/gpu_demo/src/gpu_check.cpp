/**
 * @file gpu_check.cpp
 * @brief 二次曲面检测调试测试节点
 * 
 * 功能：
 * 1. 订阅 /camera/depth/color/points 点云话题（与plane_test相同）
 * 2. 只调用QuadricDetect进行检测
 * 3. 输出详细调试信息，不发布结果
 * 
 * @author PFZL-423
 * @date 2025-10-14
 */

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <chrono>
#include <memory>

#include "gpu_demo/QuadricDetect.h"

/**
 * @class QuadricTestNode
 * @brief 二次曲面检测调试节点
 */
class QuadricTestNode
{
private:
    ros::NodeHandle nh_;
    ros::NodeHandle private_nh_;
    
    // ROS通信
    ros::Subscriber cloud_sub_;
    
    // 二次曲面检测器
    std::unique_ptr<QuadricDetect> quadric_detector_;
    
    // 配置参数
    std::string input_topic_;
    bool has_processed_;
    
    // 预处理参数（简单的PCL预处理）
    double voxel_leaf_size_;
    int sor_mean_k_;
    double sor_stddev_mul_thresh_;
    
    // 检测参数
    quadric::DetectorParams detector_params_;
    
public:
    QuadricTestNode() : private_nh_("~"), has_processed_(false)
    {
        // Step 1: 加载参数
        loadParameters();
        
        // Step 2: 初始化检测器
        initializeDetector();
        
        // Step 3: 设置ROS订阅
        setupROS();
        
        ROS_INFO("========================================");
        ROS_INFO("🚀 Quadric Detection Test Node Ready!");
        ROS_INFO("   Input topic: %s", input_topic_.c_str());
        ROS_INFO("   Distance threshold: %.4f", detector_params_.quadric_distance_threshold);
        ROS_INFO("   Min inlier count: %d", detector_params_.min_quadric_inlier_count_absolute);
        ROS_INFO("   Max iterations: %d", detector_params_.quadric_max_iterations);
        ROS_INFO("   Verbosity: %d", detector_params_.verbosity);
        ROS_INFO("========================================");
    }
    
private:
    /**
     * @brief 加载ROS参数
     */
    void loadParameters()
    {
        // ROS话题配置 - 订阅 test_points 来接收 plane_test 发布的剩余点云
        private_nh_.param<std::string>("input_topic", input_topic_, "/test_points");

        // 简单预处理参数
        private_nh_.param("voxel_leaf_size", voxel_leaf_size_, 0.01);
        private_nh_.param("sor_mean_k", sor_mean_k_, 50);
        private_nh_.param("sor_stddev_mul_thresh", sor_stddev_mul_thresh_, 1.0);
        
        // 二次曲面检测参数
        private_nh_.param("min_remaining_points_percentage", 
                         detector_params_.min_remaining_points_percentage, 0.03);
        private_nh_.param("quadric_distance_threshold", 
                         detector_params_.quadric_distance_threshold, 0.02);
        private_nh_.param("min_quadric_inlier_count_absolute", 
                         detector_params_.min_quadric_inlier_count_absolute, 200);
        private_nh_.param("quadric_max_iterations", 
                         detector_params_.quadric_max_iterations, 5000);
        private_nh_.param("quadric_verbosity", 
                         detector_params_.verbosity, 2);  // 默认详细输出
    }
    
    /**
     * @brief 初始化检测器
     */
    void initializeDetector()
    {
        try {
            quadric_detector_ = std::make_unique<QuadricDetect>(detector_params_);
            ROS_INFO("[Init] QuadricDetect initialized successfully");
        }
        catch (const std::exception& e) {
            ROS_ERROR("[Init] Failed to initialize detector: %s", e.what());
            exit(EXIT_FAILURE);
        }
    }
    
    /**
     * @brief 设置ROS通信
     */
    void setupROS()
    {
        cloud_sub_ = nh_.subscribe(input_topic_, 1, &QuadricTestNode::cloudCallback, this);
    }
    
    /**
     * @brief 点云回调函数
     */
    void cloudCallback(const sensor_msgs::PointCloud2::ConstPtr& msg)
    {
        // 🔧 修改：不再只处理第一帧，而是持续处理，但限制频率
        static int frame_count = 0;
        frame_count++;
        
        // 每3帧处理一次，避免过于频繁
        if (frame_count % 3 != 0) {
            return;
        }
        
        ROS_INFO("\n========================================");
        ROS_INFO("📥 Received Point Cloud #%d", frame_count / 3);
        ROS_INFO("   Points: %d", msg->width * msg->height);
        ROS_INFO("   Frame: %s", msg->header.frame_id.c_str());
        ROS_INFO("========================================\n");
        
        auto total_start = std::chrono::high_resolution_clock::now();
        
        try {
            // Step 1: 转换为PCL格式
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr input_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
            pcl::fromROSMsg(*msg, *input_cloud);
            
            ROS_INFO("[Step 1] PCL Conversion: %zu points", input_cloud->size());
            
            // Step 2: 体素下采样
            auto voxel_start = std::chrono::high_resolution_clock::now();
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr voxel_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
            pcl::VoxelGrid<pcl::PointXYZRGB> voxel_filter;
            voxel_filter.setInputCloud(input_cloud);
            voxel_filter.setLeafSize(voxel_leaf_size_, voxel_leaf_size_, voxel_leaf_size_);
            voxel_filter.filter(*voxel_cloud);
            auto voxel_end = std::chrono::high_resolution_clock::now();
            float voxel_time = std::chrono::duration<float, std::milli>(voxel_end - voxel_start).count();
            
            ROS_INFO("[Step 2] Voxel Filter: %zu -> %zu points (%.2f ms)", 
                     input_cloud->size(), voxel_cloud->size(), voxel_time);
            
            // Step 3: 统计滤波去除离群点
            // auto sor_start = std::chrono::high_resolution_clock::now();
            // pcl::PointCloud<pcl::PointXYZRGB>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
            // pcl::StatisticalOutlierRemoval<pcl::PointXYZRGB> sor;
            // sor.setInputCloud(voxel_cloud);
            // sor.setMeanK(sor_mean_k_);
            // sor.setStddevMulThresh(sor_stddev_mul_thresh_);
            // sor.filter(*filtered_cloud);
            // auto sor_end = std::chrono::high_resolution_clock::now();
            // float sor_time = std::chrono::duration<float, std::milli>(sor_end - sor_start).count();
            
            // ROS_INFO("[Step 3] Outlier Removal: %zu -> %zu points (%.2f ms)", 
            //          voxel_cloud->size(), filtered_cloud->size(), sor_time);
            
            ROS_INFO("\n=== QUADRIC DETECTION START ===");
            
            // Step 4: 二次曲面检测
            auto detect_start = std::chrono::high_resolution_clock::now();
            bool success = quadric_detector_->processCloud(voxel_cloud);
            auto detect_end = std::chrono::high_resolution_clock::now();
            float detect_time = std::chrono::duration<float, std::milli>(detect_end - detect_start).count();
            
            ROS_INFO("\n=== QUADRIC DETECTION END ===");
            
            auto total_end = std::chrono::high_resolution_clock::now();
            float total_time = std::chrono::duration<float, std::milli>(total_end - total_start).count();
            
            // 输出结果
            ROS_INFO("\n========================================");
            ROS_INFO("📊 DETECTION RESULTS");
            ROS_INFO("========================================");
            
            if (success) {
                const auto& detected_primitives = quadric_detector_->getDetectedPrimitives();
                
                ROS_INFO("✅ Detection Success!");
                ROS_INFO("   Found: %zu quadric surfaces", detected_primitives.size());
                
                // 输出每个检测到的二次曲面信息
                for (size_t i = 0; i < detected_primitives.size(); ++i) {
                    const auto& primitive = detected_primitives[i];
                    ROS_INFO("\n🔸 Quadric Surface #%zu:", i + 1);
                    ROS_INFO("   Type: %s", primitive.type.c_str());
                    ROS_INFO("   Inliers: %zu points", primitive.inliers->size());
                    
                    // 输出4x4二次曲面矩阵
                    const auto& Q = primitive.model_coefficients;
                    ROS_INFO("   Matrix (4×4):");
                    ROS_INFO("   [%8.4f %8.4f %8.4f %8.4f]", Q(0,0), Q(0,1), Q(0,2), Q(0,3));
                    ROS_INFO("   [%8.4f %8.4f %8.4f %8.4f]", Q(1,0), Q(1,1), Q(1,2), Q(1,3));
                    ROS_INFO("   [%8.4f %8.4f %8.4f %8.4f]", Q(2,0), Q(2,1), Q(2,2), Q(2,3));
                    ROS_INFO("   [%8.4f %8.4f %8.4f %8.4f]", Q(3,0), Q(3,1), Q(3,2), Q(3,3));
                }
                
                // 获取剩余点云
                pcl::PointCloud<pcl::PointXYZRGB>::Ptr remaining_cloud = quadric_detector_->getFinalCloud();
                ROS_INFO("\n📍 Remaining Points: %zu", remaining_cloud->size());
                
            } else {
                ROS_WARN("⚠️  Detection Failed (No surfaces found)");
            }
            
            ROS_INFO("\n⏱️  TIMING BREAKDOWN");
            ROS_INFO("========================================");
            ROS_INFO("   Voxel Filter:       %6.2f ms", voxel_time);
            // ROS_INFO("   Outlier Removal:    %6.2f ms", sor_time);
            ROS_INFO("   Quadric Detection:  %6.2f ms", detect_time);
            ROS_INFO("   ─────────────────────────────");
            ROS_INFO("   Total:              %6.2f ms", total_time);
            ROS_INFO("========================================\n");
            
            // 🔧 修改：不再自动退出，继续处理下一帧
            ROS_INFO("⏳ Waiting for next point cloud (processing every 3rd frame)...\n");
        }
        catch (const std::exception& e) {
            ROS_ERROR("❌ Error during processing: %s", e.what());
        }
    }
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "quadric_test_node");
    
    try {
        QuadricTestNode node;
        ros::spin();
    }
    catch (const std::exception& e) {
        ROS_ERROR("Node failed: %s", e.what());
        return -1;
    }
    
    return 0;
}
