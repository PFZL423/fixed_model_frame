#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <std_msgs/Header.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <geometry_msgs/TransformStamped.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <memory>
#include <chrono>

#include "gpu_demo/GPUPreprocessor.h"

class GPUPreprocessorTestNode
{
public:
    GPUPreprocessorTestNode() : nh_("~"), processor_(std::make_unique<GPUPreprocessor>()), processed_first_frame_(false)
    {
        loadParameters();

        // 🔒 强制禁用所有法线相关功能
        config_.compute_normals = false; // 以防万一有这个字段

        processor_->reserveMemory(max_points_);

        cloud_sub_ = nh_.subscribe(input_topic_, 1,
                                   &GPUPreprocessorTestNode::cloudCallback, this);
        cloud_pub_ = nh_.advertise<sensor_msgs::PointCloud2>(output_topic_,
                                                             queue_size_, latch_);

        // 创建定时器，用于持续发布处理后的点云
        publish_timer_ = nh_.createTimer(ros::Duration(0.1), // 10Hz发布频率
                                        &GPUPreprocessorTestNode::publishTimerCallback, this);

        // 🚀 发布静态tf变换，建立map到base_link的关系
        publishStaticTransform();

        ROS_INFO("[GPUPreprocessorTest] Initialized - NORMALS DISABLED");
        ROS_INFO("  - Input topic: %s", input_topic_.c_str());
        ROS_INFO("  - Output topic: %s", output_topic_.c_str());
        ROS_INFO("  - Voxel size: %.3f", config_.voxel_size);
        ROS_INFO("  - Frame ID: %s", frame_id_.c_str());
    }

    ~GPUPreprocessorTestNode()
    {
        processor_->clearMemory();
        ROS_INFO("[GPUPreprocessorTest] Cleanup completed");
    }

private:
    void loadParameters()
    {
        // 话题参数
        nh_.param<std::string>("input_topic", input_topic_, "/generated_cloud");
        nh_.param<std::string>("output_topic", output_topic_, "/processed_cloud");

        // 🎯 只加载基础预处理功能
        nh_.param<bool>("enable_voxel_filter", config_.enable_voxel_filter, true);
        nh_.param<float>("voxel_size", config_.voxel_size, 0.08f);

        nh_.param<bool>("enable_outlier_removal", config_.enable_outlier_removal, false);
        nh_.param<int>("statistical_k", config_.statistical_k, 50);
        nh_.param<float>("statistical_stddev", config_.statistical_stddev, 1.0f);

        nh_.param<bool>("enable_ground_removal", config_.enable_ground_removal, false);
        nh_.param<float>("ground_threshold", config_.ground_threshold, 0.02f);

        // 🔒 法线功能完全禁用
        config_.compute_normals = false;

        // 内存配置
        int max_points_int;
        nh_.param<int>("max_points", max_points_int, 6000); // 减少内存使用
        max_points_ = static_cast<size_t>(max_points_int);

        nh_.param<int>("queue_size", queue_size_, 1);
        nh_.param<bool>("latch", latch_, true);
        nh_.param<std::string>("frame_id", frame_id_, "base_link");
    }

    void publishStaticTransform()
    {
        // 🚀 发布静态tf变换：map -> base_link
        static tf2_ros::StaticTransformBroadcaster static_broadcaster;
        
        geometry_msgs::TransformStamped static_transformStamped;
        static_transformStamped.header.stamp = ros::Time::now();
        static_transformStamped.header.frame_id = "map";
        static_transformStamped.child_frame_id = frame_id_;
        
        // 设置为原点，无旋转
        static_transformStamped.transform.translation.x = 0.0;
        static_transformStamped.transform.translation.y = 0.0;
        static_transformStamped.transform.translation.z = 0.0;
        static_transformStamped.transform.rotation.x = 0.0;
        static_transformStamped.transform.rotation.y = 0.0;
        static_transformStamped.transform.rotation.z = 0.0;
        static_transformStamped.transform.rotation.w = 1.0;
        
        static_broadcaster.sendTransform(static_transformStamped);
        
        ROS_INFO("[GPUPreprocessorTest] Published static transform: map -> %s", frame_id_.c_str());
    }

    void cloudCallback(const sensor_msgs::PointCloud2::ConstPtr &msg)
    {
        auto start_time = std::chrono::high_resolution_clock::now();

        ROS_INFO("[GPUPreprocessorTest] Processing frame with %d points", msg->width * msg->height);

        try
        {
            // 转换为PCL
            pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud(new pcl::PointCloud<pcl::PointXYZ>);
            pcl::fromROSMsg(*msg, *input_cloud);

            // 🚀 GPU预处理（只做基础功能）
            auto process_start = std::chrono::high_resolution_clock::now();
            ProcessingResult result = processor_->process(input_cloud, config_);
            auto process_end = std::chrono::high_resolution_clock::now();

            float process_time = std::chrono::duration<float, std::milli>(
                                     process_end - process_start)
                                     .count();

            ROS_INFO("  ✅ GPU processing: %.2f ms, output: %zu points",
                     process_time, result.getPointCount());

            // 🔥 只下载基础点云（不涉及法线）
            std::vector<GPUPoint3f> cpu_points = result.downloadPoints();

            // 转换为PCL并保存
            processed_cloud_.reset(new pcl::PointCloud<pcl::PointXYZ>);
            processed_cloud_->reserve(cpu_points.size());

            for (const auto &gpu_pt : cpu_points)
            {
                processed_cloud_->emplace_back(gpu_pt.x, gpu_pt.y, gpu_pt.z);
            }

            processed_cloud_->width = processed_cloud_->size();
            processed_cloud_->height = 1;
            processed_cloud_->is_dense = true;

            // 更新header信息
            original_header_ = msg->header;
            original_header_.frame_id = frame_id_;

            // 标记已处理过至少一帧
            processed_first_frame_ = true;

            // 性能统计
            auto total_time = std::chrono::high_resolution_clock::now();
            float total_ms = std::chrono::duration<float, std::milli>(
                                 total_time - start_time)
                                 .count();

            if (processed_cloud_->size() > 0) {
                ROS_INFO("  📊 SUCCESS: %.2f ms total, compression: %.1f%% (%zu -> %zu points)",
                         total_ms, 100.0f * processed_cloud_->size() / input_cloud->size(),
                         input_cloud->size(), processed_cloud_->size());
            } else {
                ROS_WARN("  ⚠️  Processing failed - 0 output points! Check GPU processing errors above.");
            }
        }
        catch (const std::exception &e)
        {
            ROS_ERROR("[GPUPreprocessorTest] ❌ Error: %s", e.what());
        }
    }

    void publishTimerCallback(const ros::TimerEvent&)
    {
        // 发布最新处理的点云（如果有的话）
        if (!processed_first_frame_ || !processed_cloud_ || processed_cloud_->empty()) {
            return;
        }

        // 发布处理后的点云
        sensor_msgs::PointCloud2 output_msg;
        pcl::toROSMsg(*processed_cloud_, output_msg);
        output_msg.header = original_header_; // 使用最新处理帧的header
        output_msg.header.stamp = ros::Time::now(); // 更新时间戳为当前时间

        cloud_pub_.publish(output_msg);
        
        ROS_DEBUG_THROTTLE(2.0, "[GPUPreprocessorTest] Publishing %zu points", processed_cloud_->size());
    }

private:
    ros::NodeHandle nh_;
    ros::Subscriber cloud_sub_;
    ros::Publisher cloud_pub_;
    ros::Timer publish_timer_;

    std::unique_ptr<GPUPreprocessor> processor_;
    PreprocessConfig config_;

    std::string input_topic_;
    std::string output_topic_;
    std::string frame_id_;
    size_t max_points_;
    int queue_size_;
    bool latch_;

    // 🎯 新增：用于单帧处理和持续发布
    bool processed_first_frame_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr processed_cloud_;
    std_msgs::Header original_header_;
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "gpu_preprocessor_test");

    try
    {
        GPUPreprocessorTestNode node;
        ROS_INFO("🚀 [GPUPreprocessorTest] Ready! Waiting for clouds...");
        ros::spin();
    }
    catch (const std::exception &e)
    {
        ROS_ERROR("💥 [GPUPreprocessorTest] Failed: %s", e.what());
        return -1;
    }

    return 0;
}
