#ifndef MINIMAL_SAMPLE_QUADRIC_GPU_H
#define MINIMAL_SAMPLE_QUADRIC_GPU_H
#include <vector>
#include <memory>
#include <string>

// 包含PCL和自定义结构体
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <Eigen/Core>
#include "point_cloud_generator/Point_cloud_preprocessor.h"

struct DetectorParams
{
    // --- 预处理参数 ---
    PreprocessingParams preprocessing;

    // --- 平面检测(RANSAC)参数 ---
    int plane_max_iterations = 1000;
    double plane_distance_threshold = 0.02;
    double min_plane_inlier_percentage = 0.10;

    // --- 二次曲面检测(RANSAC)参数 ---
    int quadric_max_iterations = 5000;
    double quadric_distance_threshold = 0.02;
    double min_quadric_inlier_percentage = 0.05;
    int min_quadric_inlier_count_absolute = 500; // 至少500个内点

    // RANSAC动态迭代次数计算所需的概率
    double ransac_probability = 0.99;

    // --- 循环控制参数 ---
    double min_remaining_points_percentage = 0.03;

    // LO-RANSAC参数
    bool enable_local_optimization = true; // 开局就开LO
    double lo_min_inlier_ratio = 0.6;      // 内点比例大于此值触发局部优化
    double desired_prob = 0.99;            // 成功概率
    int lo_sample_size = 15;               // 局部优化采样点数
    int verbosity = 1;                     // 控制打印，0=关, 1=简略, 2=详细
};
// 2用于存储结果的结构体
struct DetectedPrimitive
{
    using PointCloudPtr = pcl::PointCloud<pcl::PointXYZ>::Ptr;
    using PointIndicesPtr = pcl::PointIndices::Ptr;
    std::string type; // "plane" 或 "quadric"

    // 模型参数，统一使用4x4 Q矩阵存储。对于平面，可以只用前几项。
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Eigen::Matrix4f model_coefficients;

    // 内点（只包含位置信息，避免Eigen对齐问题）
    pcl::PointCloud<pcl::PointXYZ>::Ptr inliers;

    DetectedPrimitive() 
    {
        // 使用安全的方式初始化 PCL 智能指针，避免拷贝构造问题
        inliers.reset(new pcl::PointCloud<pcl::PointXYZ>());
        
        // 初始化 Eigen 矩阵
        model_coefficients.setZero();
    }
};

class MinimalSampleQuadric_GPU
{

public:
    using PointCloud = pcl::PointCloud<pcl::PointXYZ>;
    using PointCloudPtr = PointCloud::Ptr;
    using PointCloudConstPtr = PointCloud::ConstPtr;

    /**
     * @brief 构造函数。接收统一的参数结构体。
     * @param params 包含所有检测器所需参数的结构体。
     */
    MinimalSampleQuadric_GPU(const DetectorParams &params);
    /**
     * @brief 析构函数。需要自定义以正确处理PIMPL指针。
     */
    ~MinimalSampleQuadric_GPU();
    // 禁用拷贝构造和拷贝赋值，因为PIMPL指针和GPU资源不易拷贝
    MinimalSampleQuadric_GPU(const MinimalSampleQuadric_GPU &) = delete;
    MinimalSampleQuadric_GPU &operator=(const MinimalSampleQuadric_GPU &) = delete;
    // 允许移动构造和移动赋值，以支持所有权转移
    MinimalSampleQuadric_GPU(MinimalSampleQuadric_GPU &&) noexcept;
    MinimalSampleQuadric_GPU &operator=(MinimalSampleQuadric_GPU &&) noexcept;
    /**
     * @brief 核心处理函数。接收一个点云，并进行平面和二次曲面的检测。
     *        此函数将触发数据上传、GPU RANSAC循环等一系列操作。
     * @param input_cloud 待处理的输入点云。
     * @return 如果成功处理则返回true，否则返回false。
     */
    bool processCloud(const PointCloudConstPtr &input_cloud);
    /**
     * @brief 获取检测到的所有几何图元（平面和二次曲面）。
     * @return 一个包含所有检测结果的常量引用。
     */
    const std::vector<DetectedPrimitive, Eigen::aligned_allocator<DetectedPrimitive>> &
    getDetectedPrimitives() const;
    /**
     * @brief 获取经过多次分割后，最终剩余的点云。
     *        此函数会触发一次从GPU到CPU的数据下载。
     * @return 指向最终剩余点云的智能指针。
     */
    PointCloudPtr getFinalCloud() const;

private:
    // --- PIMPL (Pointer to Implementation) ---
    // 这是隐藏所有CUDA/GPU细节的关键。
    // `MinimalSampleQuadric_GPU_Impl` 将在 .cu 文件中被完整定义。
    class MinimalSampleQuadric_GPU_Impl;

    // --- 结果存储 ---
    std::unique_ptr<MinimalSampleQuadric_GPU_Impl> impl_;
    // 检测结果在GPU上计算完成后，会被下载并存储在这里。

    std::vector<DetectedPrimitive, Eigen::aligned_allocator<DetectedPrimitive>> detected_primitives_;
};

#endif