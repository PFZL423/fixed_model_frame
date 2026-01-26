#ifndef SUPERVOXEL_PROCESSOR_H
#define SUPERVOXEL_PROCESSOR_H

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/segmentation/supervoxel_clustering.h>
#include <pcl/surface/convex_hull.h>
#include <pcl/filters/voxel_grid.h>
#include <map>
#include <unordered_map>
#include <vector>
#include <mutex>
#include <memory>
#include <limits>

namespace super_voxel {

// 凸包数据结构
struct ConvexHullData {
    uint32_t supervoxel_id;
    pcl::PointCloud<pcl::PointXYZI>::Ptr hull_points;
    std::vector<pcl::Vertices> polygons;
    size_t input_point_count;
    double computation_time_ms;
    
    ConvexHullData() : supervoxel_id(0), input_point_count(0), computation_time_ms(0.0) {
        hull_points.reset(new pcl::PointCloud<pcl::PointXYZI>);
    }
};

// 处理参数结构
struct SupervoxelParams {
    double voxel_resolution = 0.04;
    double seed_resolution = 0.15;
    double color_importance = 0.2;
    double spatial_importance = 0.4;
    double normal_importance = 1.0;
    
    bool enable_voxel_downsample = false;
    double downsample_leaf_size = 0.02;
    
    bool use_2d_convex_hull = true;  // 默认使用2D凸包
    size_t min_points_for_hull = 3;
};

// 处理结果统计
struct ProcessingStats {
    size_t total_supervoxels = 0;
    size_t valid_convex_hulls = 0;
    size_t total_input_points = 0;
    double total_processing_time_ms = 0.0;
    double supervoxel_time_ms = 0.0;
    double hull_computation_time_ms = 0.0;
    
    double getAvgPointsPerHull() const {
        return valid_convex_hulls > 0 ? (double)total_input_points / valid_convex_hulls : 0.0;
    }
    
    double getAvgTimePerHull() const {
        return valid_convex_hulls > 0 ? hull_computation_time_ms / valid_convex_hulls : 0.0;
    }
};

/**
 * @brief 超体素分割和凸包计算处理器
 * 
 * 这个类封装了超体素分割和凸包计算的完整流程，提供可复用的接口
 */
class SupervoxelProcessor {
public:
    using PointT = pcl::PointXYZI;
    using PointCloudPtr = pcl::PointCloud<PointT>::Ptr;
    
    /**
     * @brief 构造函数
     * @param params 处理参数
     */
    explicit SupervoxelProcessor(const SupervoxelParams& params = SupervoxelParams());
    
    /**
     * @brief 析构函数
     */
    ~SupervoxelProcessor() = default;
    
    /**
     * @brief 处理点云，执行超体素分割和凸包计算
     * @param input_cloud 输入点云
     * @return 是否处理成功
     */
    bool processPointCloud(const PointCloudPtr& input_cloud);
    
    /**
     * @brief 获取所有凸包数据
     * @return 凸包数据向量的常引用
     */
    const std::vector<ConvexHullData>& getConvexHulls() const;
    
    /**
     * @brief 获取处理统计信息
     * @return 处理统计信息的常引用
     */
    const ProcessingStats& getProcessingStats() const;
    
    /**
     * @brief 获取标记点云（按supervoxel上色）
     * @return 上色后的点云指针
     */
    pcl::PointCloud<pcl::PointXYZI>::Ptr getColoredCloud() const;
    
    /**
     * @brief 清除所有处理结果
     */
    void clear();
    
    /**
     * @brief 更新处理参数
     * @param params 新的处理参数
     */
    void updateParams(const SupervoxelParams& params);
    
    /**
     * @brief 获取当前参数
     * @return 当前处理参数的常引用
     */
    const SupervoxelParams& getParams() const;

private:
    SupervoxelParams params_;
    std::vector<ConvexHullData> convex_hulls_;
    ProcessingStats stats_;
    pcl::PointCloud<pcl::PointXYZI>::Ptr colored_cloud_;
    mutable std::mutex mutex_;
    
    // 内部处理方法
    PointCloudPtr preprocessPointCloud(const PointCloudPtr& input_cloud);
    bool performSupervoxelSegmentation(const PointCloudPtr& cloud, 
                                       std::map<uint32_t, pcl::Supervoxel<PointT>::Ptr>& supervoxel_clusters,
                                       pcl::PointCloud<pcl::PointXYZL>::Ptr& labeled_cloud);
    void buildLabelIndex(const pcl::PointCloud<pcl::PointXYZL>::Ptr& labeled_cloud,
                        const PointCloudPtr& processed_cloud,
                        std::unordered_map<uint32_t, std::vector<size_t>>& label_to_points);
    bool computeConvexHull(uint32_t supervoxel_id, 
                          const PointCloudPtr& points, 
                          ConvexHullData& hull_data);
    void generateColoredCloud(const pcl::PointCloud<pcl::PointXYZL>::Ptr& labeled_cloud);
    bool isPointSetDegenerate(const PointCloudPtr& points) const;
};

} // namespace super_voxel

#endif // SUPERVOXEL_PROCESSOR_H