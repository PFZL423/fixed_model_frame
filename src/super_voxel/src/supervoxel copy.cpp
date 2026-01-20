#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/segmentation/supervoxel_clustering.h>
#include <visualization_msgs/Marker.h>
#include <pcl/surface/convex_hull.h>
#include <map>
#include <mutex>
#include <iomanip>
#include <exception>
#include <chrono>

class SupervoxelNode
{
public:
    SupervoxelNode(ros::NodeHandle &nh, ros::NodeHandle &pnh)
    {
        // 从参数服务器读取配置
        pnh.param<std::string>("input_topic", input_topic_, std::string("/camera/rgb/points"));
        pnh.param<std::string>("output_topic", output_topic_, std::string("/supervoxel_cloud"));
        
        pnh.param<double>("voxel_resolution", voxel_resolution_, 0.04);
        pnh.param<double>("seed_resolution", seed_resolution_, 0.15);
        pnh.param<double>("color_importance", color_importance_, 0.2);
        pnh.param<double>("spatial_importance", spatial_importance_, 0.4);
        pnh.param<double>("normal_importance", normal_importance_, 1.0);
        
        // 体素下采样参数 (禁用)
        pnh.param<bool>("enable_voxel_downsample", enable_voxel_downsample_, false);
        pnh.param<double>("downsample_leaf_size", downsample_leaf_size_, 0.02);
        pnh.param<bool>("adaptive_supervoxel_params", adaptive_supervoxel_params_, false);

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
    double voxel_resolution_, seed_resolution_;
    double color_importance_, spatial_importance_, normal_importance_;
    bool enable_voxel_downsample_;
    double downsample_leaf_size_;
    bool adaptive_supervoxel_params_;
    std::mutex mutex_;
    double last_time_;

    void cloudCallback(const sensor_msgs::PointCloud2ConstPtr &msg)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        double start_time = ros::Time::now().toSec();

        typedef pcl::PointXYZRGBA PointT;
        pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>);
        pcl::fromROSMsg(*msg, *cloud);

        if (cloud->empty())
        {
            ROS_WARN("Empty cloud received.");
            return;
        }

        // 直接使用原始点云，不进行下采样
        ROS_INFO("Processing %zu points without downsampling", cloud->size());

        // 超体素提取
        pcl::SupervoxelClustering<PointT> super(voxel_resolution_, seed_resolution_);
        super.setInputCloud(cloud);
        super.setColorImportance(color_importance_);
        super.setSpatialImportance(spatial_importance_);
        super.setNormalImportance(normal_importance_);

        std::map<uint32_t, pcl::Supervoxel<PointT>::Ptr> supervoxel_clusters;
        
        try {
            super.extract(supervoxel_clusters);
        } catch (const std::exception& e) {
            ROS_ERROR("Supervoxel extraction failed: %s", e.what());
            return;
        }

        ROS_INFO("Extracted %zu supervoxels", supervoxel_clusters.size());

        pcl::PointCloud<pcl::PointXYZL>::Ptr labeled_cloud = super.getLabeledCloud();
        
        ROS_INFO("Original cloud size: %zu, Labeled cloud size: %zu", cloud->size(), labeled_cloud->size());

        double current_time = ros::Time::now().toSec();
        double fps = 1.0 / (current_time - last_time_);
        last_time_ = current_time;
        ROS_INFO_STREAM(" Processing FPS: " << std::fixed << std::setprecision(2) << fps);

        // 按label上色的点云
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
        colored_cloud->header = labeled_cloud->header;
        colored_cloud->width = labeled_cloud->width;
        colored_cloud->height = labeled_cloud->height;
        colored_cloud->is_dense = labeled_cloud->is_dense;
        colored_cloud->points.resize(labeled_cloud->points.size());

        for (size_t i = 0; i < labeled_cloud->points.size(); ++i)
        {
            pcl::PointXYZRGB point;
            point.x = labeled_cloud->points[i].x;
            point.y = labeled_cloud->points[i].y;
            point.z = labeled_cloud->points[i].z;

            uint32_t label = labeled_cloud->points[i].label;
            point.r = (label * 53) % 255;
            point.g = (label * 97) % 255;
            point.b = (label * 193) % 255;

            colored_cloud->points[i] = point;
        }

        sensor_msgs::PointCloud2 output;
        pcl::toROSMsg(*colored_cloud, output);
        output.header = msg->header;
        
        // 修复frame_id - 去掉前导斜杠
        std::string frame_id = msg->header.frame_id;
        if (frame_id.front() == '/') {
            frame_id = frame_id.substr(1);
        }
        output.header.frame_id = frame_id;
        
        pub_.publish(output);

        // --------- 凸包可视化Marker -------------
        visualization_msgs::Marker marker;
        marker.header.frame_id = frame_id;
        marker.header.stamp = ros::Time::now();
        marker.ns = "supervoxel_hulls";
        marker.id = 0;
        marker.type = visualization_msgs::Marker::LINE_LIST;
        marker.action = visualization_msgs::Marker::ADD;
        marker.scale.x = 0.02;  // LINE_LIST的线宽
        marker.pose.orientation.w = 1.0;
        marker.points.clear();
        marker.colors.clear();

        // ========== 性能分析 ==========
        auto hull_start = std::chrono::high_resolution_clock::now();
        
        // 遍历每个超体素，计算凸包并加到marker里
        int valid_hulls = 0;
        size_t total_hull_points = 0;
        double total_hull_time = 0.0;
        
        // 获取超体素包含的实际点云数据
        std::multimap<uint32_t, uint32_t> supervoxel_adjacency;
        super.getSupervoxelAdjacency(supervoxel_adjacency);
        
        for (const auto &kv : supervoxel_clusters)
        {
            uint32_t sv_label = kv.first;
            
            // 使用supervoxel的voxels_（这些是实际的空间点，不是那么稀疏）
            auto sv_voxels = kv.second->voxels_;
            
            // 另外，我们可以尝试获取原始点云中对应的所有点
            pcl::PointCloud<PointT>::Ptr sv_all_points(new pcl::PointCloud<PointT>);
            
            // 方法1：使用voxels（体素中心点）
            *sv_all_points = *sv_voxels;
            
            // 方法2：尝试从labeled_cloud中找到对应的点（如果大小匹配）
            if (labeled_cloud->size() == cloud->size()) {
                for (size_t i = 0; i < labeled_cloud->size(); ++i) {
                    if (labeled_cloud->points[i].label == sv_label) {
                        sv_all_points->points.push_back(cloud->points[i]);
                    }
                }
            }
                        
            if (sv_all_points->size() < 3) {  
                ROS_DEBUG("  - SKIPPING: Too few points (%zu < 3)", sv_all_points->size());
                continue;
            }

            // 单个凸包计算时间测量
            auto single_hull_start = std::chrono::high_resolution_clock::now();
            
            pcl::ConvexHull<PointT> chull;
            chull.setInputCloud(sv_all_points);
            // 使用3D凸包（默认行为）
            
            pcl::PointCloud<PointT> hull_points;
            std::vector<pcl::Vertices> polygons;
            chull.reconstruct(hull_points, polygons);
            
            auto single_hull_end = std::chrono::high_resolution_clock::now();
            double single_hull_time = std::chrono::duration<double, std::milli>(single_hull_end - single_hull_start).count();
            total_hull_time += single_hull_time;
            
            if (hull_points.empty() || polygons.empty()) {
                ROS_DEBUG("Empty convex hull for supervoxel %u", sv_label);
                continue;
            }
            
            valid_hulls++;
            total_hull_points += sv_all_points->size();
            
            std_msgs::ColorRGBA c;
            c.a = 1.0; // 改为不透明，更容易看到
            c.r = (sv_label * 53) % 255 / 255.0;
            c.g = (sv_label * 97) % 255 / 255.0;
            c.b = (sv_label * 193) % 255 / 255.0;
            
            ROS_DEBUG("  - Input points: %zu, Hull vertices: %zu, Time: %.2fms", 
                     sv_all_points->size(), hull_points.size(), single_hull_time);

            // 显示凸包的边线
            for (const auto& polygon : polygons)
            {
                // 为每个多边形的边创建线段
                for (size_t i = 0; i < polygon.vertices.size(); ++i)
                {
                    size_t next_i = (i + 1) % polygon.vertices.size();
                    
                    // 第一个顶点
                    geometry_msgs::Point p1;
                    p1.x = hull_points.points[polygon.vertices[i]].x;
                    p1.y = hull_points.points[polygon.vertices[i]].y;
                    p1.z = hull_points.points[polygon.vertices[i]].z;
                    
                    // 第二个顶点
                    geometry_msgs::Point p2;
                    p2.x = hull_points.points[polygon.vertices[next_i]].x;
                    p2.y = hull_points.points[polygon.vertices[next_i]].y;
                    p2.z = hull_points.points[polygon.vertices[next_i]].z;
                    
                    // 添加线段（每条线需要两个点）
                    marker.points.push_back(p1);
                    marker.points.push_back(p2);
                    marker.colors.push_back(c);
                    marker.colors.push_back(c);
                }
            }
        }

        auto hull_end = std::chrono::high_resolution_clock::now();
        double total_hull_processing = std::chrono::duration<double, std::milli>(hull_end - hull_start).count();
        
        ROS_INFO("=== PERFORMANCE SUMMARY ===");
        ROS_INFO("Total supervoxels: %zu", supervoxel_clusters.size());
        ROS_INFO("Valid convex hulls: %d", valid_hulls);
        ROS_INFO("Total input points: %zu", total_hull_points);
        ROS_INFO("Avg points per hull: %.1f", valid_hulls > 0 ? (double)total_hull_points/valid_hulls : 0);
        ROS_INFO("Hull computation time: %.2f ms", total_hull_time);
        ROS_INFO("Total hull processing: %.2f ms", total_hull_processing);
        ROS_INFO("Avg time per hull: %.2f ms", valid_hulls > 0 ? total_hull_time/valid_hulls : 0);
        ROS_INFO("Hull computation %% of total: %.1f%%", total_hull_processing > 0 ? (total_hull_time/total_hull_processing)*100 : 0);
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
