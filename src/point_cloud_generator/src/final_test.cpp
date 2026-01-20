#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <chrono>

// 确保CMakeLists.txt中的include_directories设置正确
#include <point_cloud_generator/MinimalSampleQuadric.h>
#include <point_cloud_generator/Point_cloud_preprocessor.h>

/**
 * @brief 从ROS参数服务器加载参数到DetectorParams结构体中
 * @param nh ROS节点句柄，用于访问参数服务器
 * @param params 需要被填充的参数结构体对象
 * @return 总是返回true，如果参数未找到，ROS会打印警告并使用结构体中的默认值
 */
bool loadParamsFromROS(const ros::NodeHandle &nh, DetectorParams &params)
{
    ROS_INFO("Loading parameters from ROS Parameter Server...");

    // ros::NodeHandle pnh(nh, "detector"); // 如果launch文件中用了<group ns="detector">

    // 加载预处理参数
    nh.getParam("preprocessing/leaf_size_x", params.preprocessing.leaf_size_x);
    nh.getParam("preprocessing/leaf_size_y", params.preprocessing.leaf_size_y);
    nh.getParam("preprocessing/leaf_size_z", params.preprocessing.leaf_size_z);
    nh.getParam("preprocessing/use_outlier_removal", params.preprocessing.use_outlier_removal);
    nh.getParam("preprocessing/sor_mean_k", params.preprocessing.sor_mean_k);
    nh.getParam("preprocessing/sor_stddev_mul_thresh", params.preprocessing.sor_stddev_mul_thresh);

    // 加载平面检测参数
    nh.getParam("plane_detection/max_iterations", params.plane_max_iterations);
    nh.getParam("plane_detection/distance_threshold", params.plane_distance_threshold);
    nh.getParam("plane_detection/min_inlier_percentage", params.min_plane_inlier_percentage);

    // 加载二次曲面检测参数 (请确保您的 DetectorParams 结构体有这些成员)
    nh.getParam("quadric_detection/max_iterations", params.quadric_max_iterations);
    nh.getParam("quadric_detection/distance_threshold", params.quadric_distance_threshold);
    nh.getParam("quadric_detection/min_inlier_percentage", params.min_quadric_inlier_percentage);
    nh.getParam("quadric_detection/voting_bin_size", params.voting_bin_size);
    nh.getParam("main_loop/min_remaining_points_percentage", params.min_remaining_points_percentage);

    ROS_INFO("Parameters loaded successfully.");
    return true;
}

// 封装节点逻辑的类
class DetectorNode
{
public:
    // 构造函数：初始化检测器实例和订阅者
    DetectorNode(const ros::NodeHandle &nh, const DetectorParams &params)
        : nh_(nh), detector_(params) // 直接用加载好的参数初始化检测器
    {
        // 初始化订阅者，订阅/generated_cloud话题，队列大小为1
        // 当有消息到达时，调用 cloudCallback 方法
        cloud_sub_ = nh_.subscribe("/generated_cloud", 1, &DetectorNode::cloudCallback, this);

        ROS_INFO("Detector node initialized. Waiting for point clouds on topic '/generated_cloud'...");
    }

    // 点云消息的回调函数
    void cloudCallback(const sensor_msgs::PointCloud2ConstPtr &cloud_msg)
    {
        ROS_INFO("=======================================================");
        ROS_INFO("Received a point cloud message with %d points.", cloud_msg->width * cloud_msg->height);

        // 1. 将ROS消息格式转换为PCL点云格式 (pcl::PointXYZ)
        // 这是 processCloud 方法需要的输入类型
        pcl::PointCloud<pcl::PointXYZ>::ConstPtr raw_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::fromROSMsg(*cloud_msg, *const_cast<pcl::PointCloud<pcl::PointXYZ> *>(raw_cloud.get()));

        if (raw_cloud->empty())
        {
            ROS_WARN("Received an empty point cloud. Skipping processing.");
            return;
        }

        // 2. 直接调用核心处理函数，传入原始点云
        // 所有的预处理和检测逻辑都封装在 processCloud 内部
        ROS_INFO("Handing raw cloud to the detector...");
        if (detector_.processCloud(raw_cloud))
        {
            // 3. 获取并打印结果
            const std::vector<DetectedPrimitive> &primitives = detector_.getDetectedPrimitives();
            ROS_INFO_STREAM("Detection finished. Found " << primitives.size() << " primitives.");

            if (primitives.empty())
            {
                ROS_INFO("No primitives were detected that met the criteria.");
            }
            else
            {
                int primitive_count = 0;
                for (const auto &primitive : primitives)
                {
                    primitive_count++;
                    ROS_INFO("--- Primitive #%d ---", primitive_count);
                    ROS_INFO("  Type: %s", primitive.type.c_str());
                    ROS_INFO("  Inlier points: %zu", primitive.inliers->size());

                    std::stringstream ss;
                    ss << "" << primitive.model_coefficients;
                        ROS_INFO("  Model Coefficients Matrix: %s", ss.str().c_str());
                }
            }

            // 打印剩余点云信息
            auto remaining_cloud = detector_.getFinalCloud();
            ROS_INFO("Final remaining points not assigned to any primitive: %zu", remaining_cloud->size());
        }
        else
        {
            ROS_ERROR("An error occurred during the detection process.");
        }
        ROS_INFO("=======================================================");
    }

private:
    ros::NodeHandle nh_;
    ros::Subscriber cloud_sub_;
    MinimalSampleQuadric detector_; // 核心算法类实例，它内部包含了预处理器
};

int main(int argc, char **argv)
{
    auto start_ = std::chrono::high_resolution_clock::now();

    // 初始化ROS节点
    ros::init(argc, argv, "final_test_node");
    ros::NodeHandle nh("~"); // 使用私有节点句柄，它会自动处理launch文件中的命名空间

    // 1. 创建参数对象并从ROS参数服务器加载
    DetectorParams params;
    if (!loadParamsFromROS(nh, params))
    {
        ROS_ERROR("Failed to load parameters. Shutting down.");
        return -1;
    }

    // 2. 创建节点逻辑处理对象，它会自动开始订阅话题
    DetectorNode node(nh, params);

    // 3. 在这里等待并处理回调函数
    ros::spin();
    auto end_ = std::chrono::high_resolution_clock::now();
    auto duration_ = std::chrono::duration_cast<std::chrono::microseconds>(end_ - start_);
    std::cout << "Total:" << duration_.count() / 1000 << "ms"<<std::endl;
    return 0;
}
