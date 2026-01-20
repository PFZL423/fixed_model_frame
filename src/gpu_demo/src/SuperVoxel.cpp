#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/segmentation/supervoxel_clustering.h>
#include <pcl/filters/random_sample.h>
#include <pcl/io/pcd_io.h>
#include <map>
#include <mutex>

class SupervoxelNode
{
public:
    SupervoxelNode(ros::NodeHandle &nh, ros::NodeHandle &pnh)
    {
        // pnh.param<std::string>("input_topic", input_topic_, std::string("/point_cloud/points"));
        pnh.param<std::string>("input_topic", input_topic_, std::string("/camera/rgb/points"));

        pnh.param<std::string>("output_topic", output_topic_, std::string("/points_supervoxel"));

        voxel_resolution_ = 0.04;
        seed_resolution_ = 0.15;
        color_importance_ = 0.2;
        spatial_importance_ = 0.4;
        normal_importance_ = 1.0;

        pub_ = nh.advertise<sensor_msgs::PointCloud2>(output_topic_, 1);
        sub_ = nh.subscribe(input_topic_, 1, &SupervoxelNode::cloudCallback, this);

        ROS_INFO("Supervoxel node started. Subscribing to %s", input_topic_.c_str());
    }

private:
    ros::Subscriber sub_;
    ros::Publisher pub_;
    std::string input_topic_, output_topic_;
    double voxel_resolution_, seed_resolution_;
    double color_importance_, spatial_importance_, normal_importance_;
    std::mutex mutex_;

    void cloudCallback(const sensor_msgs::PointCloud2ConstPtr &msg)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        double last_time = ros::Time::now().toSec();

        typedef pcl::PointXYZRGBA PointT;
        pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>);
        pcl::fromROSMsg(*msg, *cloud);

        if (cloud->empty())
        {
            ROS_WARN("Empty cloud received.");
            return;
        }

        pcl::SupervoxelClustering<PointT> super(voxel_resolution_, seed_resolution_);
        super.setInputCloud(cloud);
        super.setColorImportance(color_importance_);
        super.setSpatialImportance(spatial_importance_);
        super.setNormalImportance(normal_importance_);

        std::map<uint32_t, pcl::Supervoxel<PointT>::Ptr> supervoxel_clusters;
        super.extract(supervoxel_clusters);

        ROS_INFO("Extracted %zu supervoxels", supervoxel_clusters.size());

        pcl::PointCloud<pcl::PointXYZL>::Ptr labeled_cloud = super.getLabeledCloud();

        double current_time = ros::Time::now().toSec();
        double fps = 1.0 / (current_time - last_time);
        ROS_INFO_STREAM("FPS: " << fps);

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

            // 根据 label 生成颜色 (哈希映射)
            uint32_t label = labeled_cloud->points[i].label;
            uint8_t r = (label * 53) % 255;
            uint8_t g = (label * 97) % 255;
            uint8_t b = (label * 193) % 255;
            point.r = r;
            point.g = g;
            point.b = b;

            colored_cloud->points[i] = point;
        }

        sensor_msgs::PointCloud2 output;
        pcl::toROSMsg(*colored_cloud, output);
        output.header = msg->header;
        pub_.publish(output);
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