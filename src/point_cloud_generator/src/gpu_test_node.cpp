#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include "point_cloud_generator/MinimalSampleQuadric_GPU.h" // 你的GPU实现头文件

class GpuTestNode
{
public:
    GpuTestNode(ros::NodeHandle &nh, const ros::NodeHandle &pnh)
    {
        // 读取参数
        DetectorParams params;
        pnh.getParam("preprocessing/leaf_size_x", params.preprocessing.leaf_size_x);
        pnh.getParam("preprocessing/leaf_size_y", params.preprocessing.leaf_size_y);
        pnh.getParam("preprocessing/leaf_size_z", params.preprocessing.leaf_size_z);
        pnh.getParam("preprocessing/use_outlier_removal", params.preprocessing.use_outlier_removal);
        pnh.getParam("preprocessing/sor_mean_k", params.preprocessing.sor_mean_k);
        pnh.getParam("preprocessing/sor_stddev_mul_thresh", params.preprocessing.sor_stddev_mul_thresh);

        pnh.getParam("plane_detection/max_iterations", params.plane_max_iterations);
        pnh.getParam("plane_detection/distance_threshold", params.plane_distance_threshold);
        pnh.getParam("plane_detection/min_inlier_percentage", params.min_plane_inlier_percentage);

        pnh.getParam("quadric_detection/max_iterations", params.quadric_max_iterations);
        pnh.getParam("quadric_detection/distance_threshold", params.quadric_distance_threshold);
        pnh.getParam("quadric_detection/min_inlier_percentage", params.min_quadric_inlier_percentage);
        pnh.getParam("quadric_detection/min_quadric_inlier_count_absolute", params.min_quadric_inlier_count_absolute);

        pnh.getParam("main_loop/min_remaining_points_percentage", params.min_remaining_points_percentage);

        pnh.getParam("lo_ransac/enable_local_optimization",params.enable_local_optimization);
        pnh.getParam("lo_ransac/lo_min_inlier_ratio", params.lo_min_inlier_ratio);
        pnh.getParam("lo_ransac/lo_sample_size", params.lo_sample_size);
        pnh.getParam("lo_ransac/verbosity", params.verbosity);
        pnh.getParam("lo_ransac/desired_prob", params.desired_prob);

        // 创建 GPU 检测器
        detector_ = std::make_unique<MinimalSampleQuadric_GPU>(params);

        // 订阅点云
        sub_cloud_ = nh.subscribe("/generated_cloud", 1, &GpuTestNode::cloudCallback, this);
    }

private:
    void cloudCallback(const sensor_msgs::PointCloud2ConstPtr &msg)
    {
        std::cout << "[cloudCallback] 开始处理点云..." << std::endl;
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::fromROSMsg(*msg, *cloud);

        std::cout << "[cloudCallback] 调用 processCloud..." << std::endl;
        if (!detector_->processCloud(cloud))
        {
            ROS_WARN("Cloud processing failed.");
            return;
        }

        std::cout << "[cloudCallback] processCloud 完成，获取结果..." << std::endl;
        const auto &results = detector_->getDetectedPrimitives();
        ROS_INFO("Detected %zu primitives", results.size());

        for (size_t i = 0; i < results.size(); ++i)
        {
            Eigen::IOFormat fmt(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n", "[", "]");
            std::ostringstream oss;
            oss << "[" << i << "] Type: " << results[i].type << ", Model:\n"
                << Eigen::Matrix4f(results[i].model_coefficients).format(fmt);

            ROS_INFO_STREAM(oss.str());
        }
        std::cout << "[cloudCallback] 回调函数完成" << std::endl;
    }

    ros::Subscriber sub_cloud_;
    std::unique_ptr<MinimalSampleQuadric_GPU> detector_;
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "gpu_test_node");
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~"); // 读取私有命名空间参数

    GpuTestNode node(nh, pnh);

    ros::spin();

    return 0;
}
