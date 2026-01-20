#ifndef POINT_CLOUD_PROCESSOR
#define POINT_CLOUD_PROCESSOR
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/statistical_outlier_removal.h>

struct PreprocessingParams
{
    // VoxelGrid downsampling parameters
    float leaf_size_x = 0.05f;
    float leaf_size_y = 0.05f;
    float leaf_size_z = 0.05f;

    // StatisticalOutlierRemoval parameters
    bool use_outlier_removal = true; // Switch to turn this step on/off
    int sor_mean_k = 50;
    double sor_stddev_mul_thresh = 1.0;
};

class PointCloudPreprocessor
{
    using PointT = pcl::PointXYZ;
    using PointCloud = pcl::PointCloud<PointT>;
    using PointCloudPtr = PointCloud::Ptr;
    using PointCloudConstPtr = PointCloud::ConstPtr;

public:
    /**
     * @brief 构造函数，接收参数并初始化滤波器
     * @param params 预处理参数结构体
     */
    explicit PointCloudPreprocessor (const PreprocessingParams &params);
    /**
     * @brief 执行完整的预处理流水线（仅下采样和离群点移除，不计算法线）
     * @param raw_cloud 输入的原始点云
     * @param processed_cloud_out 输出的处理后的点云（下采样+去离群点）
     * @return 如果处理成功返回true，否则返回false
     */
    bool process(const PointCloudConstPtr &raw_cloud,
                PointCloudPtr &processed_cloud_out);
private:
    // 预处理参数
    PreprocessingParams params_;
    // PCL滤波器对象，作为类的成员变量
    pcl::VoxelGrid<PointT> voxel_filter_;
    pcl::StatisticalOutlierRemoval<PointT> sor_filter_;
};

#endif 