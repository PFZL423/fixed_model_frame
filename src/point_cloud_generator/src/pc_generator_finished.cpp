#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <random>
#include <vector>
#include <string>
#include <sstream>

// 定义点云类型别名
using PointT = pcl::PointXYZ;
using PointCloudT = pcl::PointCloud<PointT>;

// ====== 函数声明 ======
PointCloudT::Ptr generateArbitraryQuadric(const std::vector<double> &coeffs, int num_points, float box_size, float tolerance);
void addGaussianNoise(PointCloudT::Ptr cloud, double std_dev);
void addOutliers(PointCloudT::Ptr cloud, int num_outliers, float bounding_box_size);
void translatePointCloud(PointCloudT::Ptr cloud, float dx, float dy, float dz);

// ====== 二次曲面点云生成 ======
PointCloudT::Ptr generateArbitraryQuadric(const std::vector<double> &coeffs, int num_points, float box_size, float tolerance)
{
    if (coeffs.size() != 10)
    {
        ROS_ERROR("Coefficients vector must have 10 elements.");
        return nullptr;
    }

    PointCloudT::Ptr cloud(new PointCloudT);
    cloud->reserve(num_points); // 预分配内存提高效率

    std::mt19937 generator(std::random_device{}());
    std::uniform_real_distribution<float> rand_coord(-box_size / 2.0f, box_size / 2.0f);

    while ((int)cloud->points.size() < num_points)
    {
        // 1. 随机坐标
        float x = rand_coord(generator);
        float y = rand_coord(generator);
        float z = rand_coord(generator);

        // 2. 曲面方程值
        double value = coeffs[0] * x * x + coeffs[1] * y * y + coeffs[2] * z * z +
                       coeffs[3] * x * y + coeffs[4] * y * z + coeffs[5] * x * z +
                       coeffs[6] * x + coeffs[7] * y + coeffs[8] * z +
                       coeffs[9];

        // 3. 判断容差
        if (std::abs(value) < tolerance)
        {
            cloud->points.emplace_back(x, y, z);
        }
    }

    cloud->width = cloud->points.size();
    cloud->height = 1;
    cloud->is_dense = true;
    return cloud;
}

// ====== 添加高斯噪声 ======
void addGaussianNoise(PointCloudT::Ptr cloud, double std_dev)
{
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0.0, std_dev);
    for (auto &point : cloud->points)
    {
        point.x += distribution(generator);
        point.y += distribution(generator);
        point.z += distribution(generator);
    }
}

// ====== 添加离群点 ======
void addOutliers(PointCloudT::Ptr cloud, int num_outliers, float bounding_box_size)
{
    std::default_random_engine generator;
    std::uniform_real_distribution<float> distribution(-bounding_box_size / 2.0f, bounding_box_size / 2.0f);
    for (int i = 0; i < num_outliers; ++i)
    {
        PointT outlier;
        outlier.x = distribution(generator);
        outlier.y = distribution(generator);
        outlier.z = distribution(generator);
        cloud->points.push_back(outlier);
    }
    cloud->width = cloud->points.size();
    cloud->height = 1;
}

// ====== 平移点云 ======
void translatePointCloud(PointCloudT::Ptr cloud, float dx, float dy, float dz)
{
    for (auto &point : cloud->points)
    {
        point.x += dx;
        point.y += dy;
        point.z += dz;
    }
}
int main(int argc, char **argv)
{
    ros::init(argc, argv, "point_cloud_generator_node");
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");

    ros::Publisher cloud_pub = nh.advertise<sensor_msgs::PointCloud2>("/generated_cloud", 1, true);

    // ====== 参数读取 ======
    int quadric_count;
    float bounding_box_size;
    float tolerance;
    double noise_std_dev;
    int num_outliers;
    double publish_rate;

    pnh.param("quadric_count", quadric_count, 1);
    pnh.param("bounding_box_size", bounding_box_size, 3.0f);
    pnh.param("tolerance", tolerance, 0.05f);
    pnh.param("noise_std_dev", noise_std_dev, 0.01);
    pnh.param("num_outliers", num_outliers, 200);
    pnh.param("publish_rate", publish_rate, 1.0);

    ROS_INFO("Generating %d quadrics at %.1f Hz...", quadric_count, publish_rate);

    // 🔧 关键修复：只生成一次点云，然后重复发布
    PointCloudT::Ptr final_cloud(new PointCloudT);

    // 🎯 一次性生成所有曲面
    for (int i = 1; i <= quadric_count; ++i)
    {
        std::string coeff_name = "coefficients_" + std::to_string(i);
        std::string num_name = "num_points_" + std::to_string(i);
        std::string offset_name = "offset_" + std::to_string(i);

        // 读取参数
        std::string coeff_str;
        pnh.param<std::string>(coeff_name, coeff_str, "1 1 1 0 0 0 0 0 0 -1");
        std::stringstream ss(coeff_str);
        std::vector<double> coeffs;
        double val;
        while (ss >> val)
            coeffs.push_back(val);

        if (coeffs.size() != 10)
        {
            ROS_FATAL("Quadric %d has wrong coefficient count.", i);
            return -1;
        }

        int num_points;
        pnh.param(num_name, num_points, 1000);

        std::string offset_str;
        pnh.param<std::string>(offset_name, offset_str, "0 0 0");
        std::stringstream ss_offset(offset_str);
        float dx, dy, dz;
        ss_offset >> dx >> dy >> dz;

        ROS_INFO("Generating Quadric %d: points=%d, offset=(%.2f, %.2f, %.2f)",
                 i, num_points, dx, dy, dz);

        // 生成曲面
        PointCloudT::Ptr quadric_cloud = generateArbitraryQuadric(coeffs, num_points, bounding_box_size, tolerance);
        if (!quadric_cloud)
        {
            ROS_ERROR("Failed to generate quadric %d", i);
            continue;
        }

        addGaussianNoise(quadric_cloud, noise_std_dev);
        translatePointCloud(quadric_cloud, dx, dy, dz);
        addOutliers(quadric_cloud, num_outliers, bounding_box_size);

        *final_cloud += *quadric_cloud;
    }

    ROS_INFO("Generated FIXED point cloud with %zu points. Will publish same data repeatedly.",
             final_cloud->size());

    // 🔧 现在只是重复发布相同的点云
    ros::Rate rate(publish_rate);
    int publish_count = 0;

    while (ros::ok())
    {
        // 每次发布相同的点云数据
        sensor_msgs::PointCloud2 cloud_msg;
        pcl::toROSMsg(*final_cloud, cloud_msg);
        cloud_msg.header.stamp = ros::Time::now(); // 时间戳更新，但数据不变
        cloud_msg.header.frame_id = "map";

        cloud_pub.publish(cloud_msg);

        publish_count++;
        if (publish_count == 1)
        {
            ROS_INFO("✅ Started publishing FIXED point cloud. Same data will repeat forever.");
        }
        else if (publish_count % 100 == 0)
        {
            ROS_INFO("📡 Published same point cloud %d times.", publish_count);
        }

        rate.sleep();
        ros::spinOnce();
    }

    return 0;
}
