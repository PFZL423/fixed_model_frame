/**
 * @file interface_test.cpp
 * @brief 节点2：二次曲面检测节点（QuadricDetectNode）
 * @details 订阅平面检测后的点云，进行二次曲面检测，发布剩余点云和可视化
 */

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <geometry_msgs/Point.h>
#include <std_msgs/ColorRGBA.h>

#include "gpu_demo/QuadricDetect.h"

// QuadricDetect 类型别名
namespace QuadricTypes {
    using Params = quadric::DetectorParams;
    using Primitive = quadric::DetectedPrimitive;
}

class QuadricDetectNode
{
public:
    QuadricDetectNode(ros::NodeHandle &nh, ros::NodeHandle &pnh)
        : nh_(nh), pnh_(pnh), quadric_detector_(nullptr)
    {
        // 读取参数
        loadParameters();

        // 初始化二次曲面检测器
        initializeQuadricDetector();

        // 设置订阅和发布
        setupPubSub();

        ROS_INFO(" [Node2] QuadricDetectNode initialized successfully");
    }

private:
    ros::NodeHandle nh_, pnh_;
    
    // 订阅者
    ros::Subscriber plane_remaining_sub_;  // 订阅节点1的平面剩余点云
    
    // 发布者
    ros::Publisher quadric_remaining_pub_; // 发布二次曲面检测后的剩余点云（返回给节点1）
    ros::Publisher quadric_marker_pub_;    // 发布二次曲面可视化

    // 二次曲面检测器
    std::unique_ptr<QuadricDetect> quadric_detector_;
    QuadricTypes::Params quadric_params_;

    // 参数
    std::string output_frame_;
    int min_remaining_points_for_quadric_;
    bool enable_visualization_ = true;        // 全局可视化开关
    bool enable_quadric_visualization_ = true;// 二次曲面可视化开关

    void loadParameters()
    {
        // 坐标系参数
        pnh_.param<std::string>("output_frame", output_frame_, "map");

    // 二次曲面功能参数
        pnh_.param("min_remaining_points_for_quadric", min_remaining_points_for_quadric_, 300);

    // 可视化开关
    pnh_.param("enable_visualization", enable_visualization_, enable_visualization_);
    pnh_.param("enable_quadric_visualization", enable_quadric_visualization_, enable_quadric_visualization_);

    // QuadricDetect 算法参数
        pnh_.param("quadric_min_remaining_points_percentage", quadric_params_.min_remaining_points_percentage, 0.03);
        pnh_.param("quadric_distance_threshold", quadric_params_.quadric_distance_threshold, 0.02);
        pnh_.param("min_quadric_inlier_count_absolute", quadric_params_.min_quadric_inlier_count_absolute, 500);
        pnh_.param("quadric_max_iterations", quadric_params_.quadric_max_iterations, 5000);
        pnh_.param("min_quadric_inlier_percentage", quadric_params_.min_quadric_inlier_percentage, 0.05);
        pnh_.param("quadric_verbosity", quadric_params_.verbosity, 1);

        ROS_INFO("=== Node2 Parameters ===");
        ROS_INFO("  Output frame: %s", output_frame_.c_str());
        ROS_INFO("  Quadric distance threshold: %.4f", quadric_params_.quadric_distance_threshold);
        ROS_INFO("  Min inliers: %d", quadric_params_.min_quadric_inlier_count_absolute);
        ROS_INFO("  Min points threshold: %d", min_remaining_points_for_quadric_);
    }

    void initializeQuadricDetector()
    {
        quadric_detector_ = std::make_unique<QuadricDetect>(quadric_params_);
        ROS_INFO("ode2] QuadricDetector initialized with threshold=%.3f", 
                 quadric_params_.quadric_distance_threshold);
    }

    void setupPubSub()
    {
    //  订阅节点1发布的平面剩余点云（队列=1，最小延迟）
        plane_remaining_sub_ = nh_.subscribe("/plane_remaining", 1, 
                                             &QuadricDetectNode::planeRemainingCallback, this);
        
        //  发布二次曲面检测后的剩余点云（不使用 latched，实时返回给节点1）
        quadric_remaining_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("/node2_output", 1, false);
        
    // 发布二次曲面可视化
    quadric_marker_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("quadric_markers", 1, true);

        ROS_INFO("=== Node2: QuadricDetectNode ===");
        ROS_INFO(" Subscribed to: /plane_remaining");
        ROS_INFO(" Publishing quadric remaining to: /node2_output");
        ROS_INFO(" Publishing quadric markers to: quadric_markers");
    }

    // 回调：处理平面剩余点云，进行二次曲面检测
    void planeRemainingCallback(const sensor_msgs::PointCloud2::ConstPtr &msg)
    {
        ROS_INFO(" [Node2] Received plane remaining cloud: %d points (frame %u)", 
                 msg->width * msg->height, msg->header.seq);

        // 转换为PCL格式
        pcl::PointCloud<pcl::PointXYZI>::Ptr input_cloud(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::fromROSMsg(*msg, *input_cloud);

        if (input_cloud->empty())
        {
            ROS_WARN(" [Node2] Received empty point cloud, skipping");
            
            // 即使为空也要返回空点云给节点1（保持帧同步）
            sensor_msgs::PointCloud2 empty_msg;
            empty_msg.header = msg->header;  // 保留原始 header（包括 seq）
            empty_msg.header.frame_id = output_frame_;
            quadric_remaining_pub_.publish(empty_msg);
            return;
        }

        // 检查点数是否足够
        if (static_cast<int>(input_cloud->size()) < min_remaining_points_for_quadric_)
        {
            ROS_INFO(" [Node2] Insufficient points (%zu < %d), skipping quadric detection",
                     input_cloud->size(), min_remaining_points_for_quadric_);
            
            // 直接返回原始点云（未经处理）
            sensor_msgs::PointCloud2 output_msg;
            pcl::toROSMsg(*input_cloud, output_msg);
            output_msg.header = msg->header;  // 保留原始 header
            output_msg.header.frame_id = output_frame_;
            quadric_remaining_pub_.publish(output_msg);
            return;
        }

        ROS_INFO("\n=== QUADRIC DETECTION START ===");

        // 二次曲面检测
        bool success = quadric_detector_->processCloud(input_cloud);

        if (!success)
        {
            ROS_WARN("[Node2] Quadric detection failed");
            
            // 失败时返回原始点云
            sensor_msgs::PointCloud2 output_msg;
            pcl::toROSMsg(*input_cloud, output_msg);
            output_msg.header = msg->header;
            output_msg.header.frame_id = output_frame_;
            quadric_remaining_pub_.publish(output_msg);
            return;
        }

        // 获取检测结果
        const auto &detected_quadrics = quadric_detector_->getDetectedPrimitives();
        ROS_INFO("[Node2] Quadrics detected: %zu", detected_quadrics.size());

        // 可视化二次曲面
        if (!detected_quadrics.empty() && enable_visualization_ && enable_quadric_visualization_)
        {
            visualizeQuadrics(detected_quadrics, msg->header);
        }

        // 获取剩余点云
        auto remaining_cloud = quadric_detector_->getFinalCloud();
        ROS_INFO("[Node2] Points remaining after quadric: %zu", remaining_cloud->size());

        // 发布剩余点云（返回给节点1）
        sensor_msgs::PointCloud2 output_msg;
        pcl::toROSMsg(*remaining_cloud, output_msg);
        output_msg.header = msg->header;  // 🔑 保留原始 header（包括 seq，用于帧同步）
        output_msg.header.frame_id = output_frame_;
        quadric_remaining_pub_.publish(output_msg);

        ROS_INFO("[Node2] Published remaining cloud (%zu points) back to Node1 (frame %u)\n",
                 remaining_cloud->size(), output_msg.header.seq);
    }

    // 可视化二次曲面（从 plane_test.cpp 搬运过来）
    void visualizeQuadrics(const std::vector<QuadricTypes::Primitive,
                                           Eigen::aligned_allocator<QuadricTypes::Primitive>> &quadrics,
                          const std_msgs::Header &header)
    {
        visualization_msgs::MarkerArray marker_array;

        // 清除之前的二次曲面标记
        visualization_msgs::Marker clear_marker;
        clear_marker.header = header;
        clear_marker.header.frame_id = output_frame_;
        clear_marker.ns = "quadrics";
        clear_marker.action = visualization_msgs::Marker::DELETEALL;
        marker_array.markers.push_back(clear_marker);

        // 为每个二次曲面创建点云可视化
        for (size_t i = 0; i < quadrics.size(); ++i)
        {
            const auto &quadric = quadrics[i];

            if (!quadric.inliers || quadric.inliers->empty())
            {
                continue;
            }

            // 使用POINTS类型显示内点
            visualization_msgs::Marker quadric_marker;
            quadric_marker.header = header;
            quadric_marker.header.frame_id = output_frame_;
            quadric_marker.ns = "quadrics";
            quadric_marker.id = i;
            quadric_marker.type = visualization_msgs::Marker::POINTS;
            quadric_marker.action = visualization_msgs::Marker::ADD;

            // 添加内点
            for (const auto &pt : quadric.inliers->points)
            {
                geometry_msgs::Point p;
                p.x = pt.x;
                p.y = pt.y;
                p.z = pt.z;
                quadric_marker.points.push_back(p);
            }

            // 使用暖色调区分（与平面的冷色调区别）
            float hue = 30.0f + (float)i / (float)quadrics.size() * 60.0f; // 30-90度（红到黄）
            auto rgb = hsvToRgb(hue, 0.9f, 1.0f);

            quadric_marker.color.r = rgb[0];
            quadric_marker.color.g = rgb[1];
            quadric_marker.color.b = rgb[2];
            quadric_marker.color.a = 0.8f;

            quadric_marker.scale.x = 0.015; // 点大小
            quadric_marker.scale.y = 0.015;

            marker_array.markers.push_back(quadric_marker);
        }

        quadric_marker_pub_.publish(marker_array);
        ROS_INFO(" [Node2] Published %zu quadric markers", quadrics.size());
    }

    // HSV 转 RGB（从 plane_test.cpp 复制）
    std::array<float, 3> hsvToRgb(float h, float s, float v)
    {
        float c = v * s;
        float x = c * (1.0f - fabs(fmod(h / 60.0f, 2.0f) - 1.0f));
        float m = v - c;

        float r, g, b;
        if (h < 60.0f)
        {
            r = c;
            g = x;
            b = 0;
        }
        else if (h < 120.0f)
        {
            r = x;
            g = c;
            b = 0;
        }
        else if (h < 180.0f)
        {
            r = 0;
            g = c;
            b = x;
        }
        else if (h < 240.0f)
        {
            r = 0;
            g = x;
            b = c;
        }
        else if (h < 300.0f)
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
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "quadric_detect_node");
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");

    try
    {
        QuadricDetectNode node(nh, pnh);
        ROS_INFO("[Node2] QuadricDetectNode started, waiting for plane remaining clouds...");
        ros::spin();
    }
    catch (const std::exception &e)
    {
        ROS_FATAL("[Node2] QuadricDetectNode failed: %s", e.what());
        return -1;
    }

    return 0;
}
