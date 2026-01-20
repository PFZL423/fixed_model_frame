#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <visualization_msgs/Marker.h>
#include <geometry_msgs/Point.h>
#include <std_msgs/ColorRGBA.h>
#include "super_voxel/supervoxel.h"
#include <iomanip>

class SupervoxelNode
{
public:
    SupervoxelNode(ros::NodeHandle &nh, ros::NodeHandle &pnh) 
        : processor_(nullptr)
    {
        // 从参数服务器读取配置
        pnh.param<std::string>("input_topic", input_topic_, std::string("/camera/rgb/points"));
        pnh.param<std::string>("output_topic", output_topic_, std::string("/supervoxel_cloud"));
        
        // 配置处理器参数
        super_voxel::SupervoxelParams params;
        pnh.param<double>("voxel_resolution", params.voxel_resolution, 0.04);
        pnh.param<double>("seed_resolution", params.seed_resolution, 0.15);
        pnh.param<double>("color_importance", params.color_importance, 0.2);
        pnh.param<double>("spatial_importance", params.spatial_importance, 0.4);
        pnh.param<double>("normal_importance", params.normal_importance, 1.0);
        pnh.param<bool>("enable_voxel_downsample", params.enable_voxel_downsample, true);
        pnh.param<double>("downsample_leaf_size", params.downsample_leaf_size, 0.02);
        pnh.param<bool>("use_2d_convex_hull", params.use_2d_convex_hull, true);

        // 创建处理器
        processor_.reset(new super_voxel::SupervoxelProcessor(params));

        pub_ = nh.advertise<sensor_msgs::PointCloud2>(output_topic_, 1);
        hull_marker_pub_ = nh.advertise<visualization_msgs::Marker>("supervoxel_hulls", 1);

        sub_ = nh.subscribe(input_topic_, 1, &SupervoxelNode::cloudCallback, this);
        last_time_ = ros::Time::now().toSec();

        ROS_INFO("Supervoxel node started. Subscribing to %s", input_topic_.c_str());
    }

private:
    ros::Subscriber sub_;
    ros::Publisher pub_;
    ros::Publisher hull_marker_pub_;
    std::string input_topic_, output_topic_;
    std::unique_ptr<super_voxel::SupervoxelProcessor> processor_;
    double last_time_;

    void cloudCallback(const sensor_msgs::PointCloud2ConstPtr &msg)
    {
        // 转换点云
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
        pcl::fromROSMsg(*msg, *cloud);

        if (cloud->empty()) {
            ROS_WARN("Empty cloud received.");
            return;
        }

        // 使用处理器进行处理
        if (!processor_->processPointCloud(cloud)) {
            ROS_ERROR("Failed to process point cloud");
            return;
        }

        // 计算FPS
        double current_time = ros::Time::now().toSec();
        double fps = 1.0 / (current_time - last_time_);
        last_time_ = current_time;

        // 获取处理结果
        const auto& stats = processor_->getProcessingStats();
        auto colored_cloud = processor_->getColoredCloud();
        const auto& convex_hulls = processor_->getConvexHulls();

        // 发布上色点云
        sensor_msgs::PointCloud2 output;
        pcl::toROSMsg(*colored_cloud, output);
        output.header = msg->header;
        
        // 修复frame_id
        std::string frame_id = msg->header.frame_id;
        if (!frame_id.empty() && frame_id.front() == '/') {
            frame_id = frame_id.substr(1);
        }
        output.header.frame_id = frame_id;
        pub_.publish(output);

        // 创建凸包可视化Marker
        visualization_msgs::Marker marker;
        marker.header.frame_id = frame_id;
        marker.header.stamp = ros::Time::now();
        marker.ns = "supervoxel_hulls";
        marker.id = 0;
        marker.type = visualization_msgs::Marker::LINE_LIST;
        marker.action = visualization_msgs::Marker::ADD;
        marker.scale.x = 0.02;
        marker.pose.orientation.w = 1.0;
        marker.points.clear();
        marker.colors.clear();

        // 预估容器大小
        size_t estimated_edges = convex_hulls.size() * 20 * 2;
        marker.points.reserve(estimated_edges);
        marker.colors.reserve(estimated_edges);

        // 从处理器获取凸包数据并生成可视化
        for (const auto& hull_data : convex_hulls) {
            // 生成颜色
            std_msgs::ColorRGBA c;
            c.a = 1.0;
            c.r = ((hull_data.supervoxel_id * 53) % 255) / 255.0f;
            c.g = ((hull_data.supervoxel_id * 97) % 255) / 255.0f;
            c.b = ((hull_data.supervoxel_id * 193) % 255) / 255.0f;

            // 显示凸包的边线
            for (const auto& polygon : hull_data.polygons) {
                for (size_t i = 0; i < polygon.vertices.size(); ++i) {
                    size_t next_i = (i + 1) % polygon.vertices.size();
                    
                    // 第一个顶点
                    geometry_msgs::Point p1;
                    p1.x = hull_data.hull_points->points[polygon.vertices[i]].x;
                    p1.y = hull_data.hull_points->points[polygon.vertices[i]].y;
                    p1.z = hull_data.hull_points->points[polygon.vertices[i]].z;
                    
                    // 第二个顶点
                    geometry_msgs::Point p2;
                    p2.x = hull_data.hull_points->points[polygon.vertices[next_i]].x;
                    p2.y = hull_data.hull_points->points[polygon.vertices[next_i]].y;
                    p2.z = hull_data.hull_points->points[polygon.vertices[next_i]].z;
                    
                    // 添加线段
                    marker.points.push_back(p1);
                    marker.points.push_back(p2);
                    marker.colors.push_back(c);
                    marker.colors.push_back(c);
                }
            }
        }

        // 输出性能统计
        ROS_INFO("=== PERFORMANCE SUMMARY ===");
        ROS_INFO("Total supervoxels: %zu", stats.total_supervoxels);
        ROS_INFO("Valid convex hulls: %zu", stats.valid_convex_hulls);
        ROS_INFO("Total input points: %zu", stats.total_input_points);
        ROS_INFO("Avg points per hull: %.1f", stats.getAvgPointsPerHull());
        ROS_INFO("Hull computation time: %.2f ms", stats.hull_computation_time_ms);
        ROS_INFO("Total processing time: %.2f ms", stats.total_processing_time_ms);
        ROS_INFO("Avg time per hull: %.2f ms", stats.getAvgTimePerHull());
        ROS_INFO("Processing FPS: %.2f", fps);
        ROS_INFO("Total marker points: %zu", marker.points.size());
        ROS_INFO("===============");

        hull_marker_pub_.publish(marker);
    }
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "supervoxel_node");
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");
    SupervoxelNode node(nh, pnh);
    ros::spin();
    return 0;
}
