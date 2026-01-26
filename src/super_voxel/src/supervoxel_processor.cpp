#include "super_voxel/supervoxel.h"
#include <chrono>
#include <iostream>
#include <unistd.h>  // for dup, dup2, close
#include <fcntl.h>   // for open, O_WRONLY

namespace super_voxel {

SupervoxelProcessor::SupervoxelProcessor(const SupervoxelParams& params) 
    : params_(params) {
    colored_cloud_.reset(new pcl::PointCloud<pcl::PointXYZI>);
}

bool SupervoxelProcessor::processPointCloud(const PointCloudPtr& input_cloud) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (!input_cloud || input_cloud->empty()) {
        return false;
    }
    
    // 清除之前的结果
    convex_hulls_.clear();
    stats_ = ProcessingStats();
    
    auto total_start = std::chrono::high_resolution_clock::now();
    
    // 1. 预处理点云（可选的体素下采样）
    PointCloudPtr processed_cloud = preprocessPointCloud(input_cloud);
    
    // 2. 超体素分割
    std::map<uint32_t, pcl::Supervoxel<PointT>::Ptr> supervoxel_clusters;
    pcl::PointCloud<pcl::PointXYZL>::Ptr labeled_cloud;
    
    auto sv_start = std::chrono::high_resolution_clock::now();
    if (!performSupervoxelSegmentation(processed_cloud, supervoxel_clusters, labeled_cloud)) {
        return false;
    }
    auto sv_end = std::chrono::high_resolution_clock::now();
    stats_.supervoxel_time_ms = std::chrono::duration<double, std::milli>(sv_end - sv_start).count();
    
    // 3. 构建标签索引
    std::unordered_map<uint32_t, std::vector<size_t>> label_to_points;
    buildLabelIndex(labeled_cloud, processed_cloud, label_to_points);
    
    // 4. 生成上色点云
    generateColoredCloud(labeled_cloud);
    
    // 5. 计算凸包
    auto hull_start = std::chrono::high_resolution_clock::now();
    
    stats_.total_supervoxels = supervoxel_clusters.size();
    convex_hulls_.reserve(supervoxel_clusters.size());
    
    for (const auto& kv : supervoxel_clusters) {
        uint32_t sv_label = kv.first;
        auto sv_voxels = kv.second->voxels_;
        
        // 构建该supervoxel的所有点
        PointCloudPtr sv_all_points(new pcl::PointCloud<PointT>);
        
        // 方法1: 添加voxel中心点
        sv_all_points->points.insert(sv_all_points->points.end(),
                                     sv_voxels->points.begin(),
                                     sv_voxels->points.end());
        
        // 方法2: 添加原始点云中的对应点
        auto it = label_to_points.find(sv_label);
        if (it != label_to_points.end()) {
            const auto& point_indices = it->second;
            sv_all_points->points.reserve(sv_all_points->size() + point_indices.size());
            for (size_t idx : point_indices) {
                sv_all_points->points.push_back(processed_cloud->points[idx]);
            }
        }
        
        // 检查点数是否足够
        if (sv_all_points->size() < params_.min_points_for_hull) {
            continue;
        }
        
        // 检查是否退化
        if (isPointSetDegenerate(sv_all_points)) {
            continue;
        }
        
        // 计算凸包
        ConvexHullData hull_data;
        hull_data.supervoxel_id = sv_label;
        hull_data.input_point_count = sv_all_points->size();
        
        if (computeConvexHull(sv_label, sv_all_points, hull_data)) {
            convex_hulls_.push_back(std::move(hull_data));
            stats_.valid_convex_hulls++;
            stats_.total_input_points += hull_data.input_point_count;
            stats_.hull_computation_time_ms += hull_data.computation_time_ms;
        }
    }
    
    auto hull_end = std::chrono::high_resolution_clock::now();
    auto total_end = std::chrono::high_resolution_clock::now();
    
    stats_.total_processing_time_ms = std::chrono::duration<double, std::milli>(total_end - total_start).count();
    
    return true;
}

SupervoxelProcessor::PointCloudPtr SupervoxelProcessor::preprocessPointCloud(const PointCloudPtr& input_cloud) {
    if (!params_.enable_voxel_downsample) {
        return input_cloud;
    }
    
    PointCloudPtr downsampled_cloud(new pcl::PointCloud<PointT>);
    pcl::VoxelGrid<PointT> voxel_filter;
    voxel_filter.setInputCloud(input_cloud);
    voxel_filter.setLeafSize(params_.downsample_leaf_size, 
                            params_.downsample_leaf_size, 
                            params_.downsample_leaf_size);
    voxel_filter.filter(*downsampled_cloud);
    
    return downsampled_cloud;
}

bool SupervoxelProcessor::performSupervoxelSegmentation(
    const PointCloudPtr& cloud,
    std::map<uint32_t, pcl::Supervoxel<PointT>::Ptr>& supervoxel_clusters,
    pcl::PointCloud<pcl::PointXYZL>::Ptr& labeled_cloud) {
    
    // 将 PointXYZI 转换为 PointXYZRGB（SupervoxelClustering 需要 RGB 类型）
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr rgb_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    rgb_cloud->resize(cloud->size());
    rgb_cloud->width = cloud->width;
    rgb_cloud->height = cloud->height;
    rgb_cloud->is_dense = cloud->is_dense;
    rgb_cloud->header = cloud->header;
    
    for (size_t i = 0; i < cloud->size(); ++i) {
        rgb_cloud->points[i].x = cloud->points[i].x;
        rgb_cloud->points[i].y = cloud->points[i].y;
        rgb_cloud->points[i].z = cloud->points[i].z;
        
        // 将 intensity 值映射到 RGB（方案1：使用 intensity 值）
        float intensity = cloud->points[i].intensity;
        uint8_t intensity_byte = static_cast<uint8_t>(std::min(255.0f, std::max(0.0f, intensity)));
        rgb_cloud->points[i].r = intensity_byte;
        rgb_cloud->points[i].g = intensity_byte;
        rgb_cloud->points[i].b = intensity_byte;
    }
    
    // 使用 PointXYZRGB 类型进行 SupervoxelClustering
    pcl::SupervoxelClustering<pcl::PointXYZRGB> super(params_.voxel_resolution, params_.seed_resolution);
    super.setInputCloud(rgb_cloud);
    super.setColorImportance(params_.color_importance);
    super.setSpatialImportance(params_.spatial_importance);
    super.setNormalImportance(params_.normal_importance);
    
    // 提取 supervoxel（使用 RGB 类型）
    std::map<uint32_t, pcl::Supervoxel<pcl::PointXYZRGB>::Ptr> rgb_supervoxel_clusters;
    
    try {
        super.extract(rgb_supervoxel_clusters);
        labeled_cloud = super.getLabeledCloud();
        
        // 将 RGB supervoxel 转换为 PointT 类型
        supervoxel_clusters.clear();
        for (const auto& kv : rgb_supervoxel_clusters) {
            pcl::Supervoxel<PointT>::Ptr sv(new pcl::Supervoxel<PointT>);
            sv->centroid_ = kv.second->centroid_;
            sv->normal_ = kv.second->normal_;
            sv->voxels_.reset(new pcl::PointCloud<PointT>);
            sv->voxels_->resize(kv.second->voxels_->size());
            for (size_t i = 0; i < kv.second->voxels_->size(); ++i) {
                sv->voxels_->points[i].x = kv.second->voxels_->points[i].x;
                sv->voxels_->points[i].y = kv.second->voxels_->points[i].y;
                sv->voxels_->points[i].z = kv.second->voxels_->points[i].z;
                // 从 RGB 恢复 intensity（取平均值）
                uint8_t r = kv.second->voxels_->points[i].r;
                uint8_t g = kv.second->voxels_->points[i].g;
                uint8_t b = kv.second->voxels_->points[i].b;
                sv->voxels_->points[i].intensity = static_cast<float>((r + g + b) / 3);
            }
            supervoxel_clusters[kv.first] = sv;
        }
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Supervoxel extraction failed: " << e.what() << std::endl;
        return false;
    }
}

void SupervoxelProcessor::buildLabelIndex(
    const pcl::PointCloud<pcl::PointXYZL>::Ptr& labeled_cloud,
    const PointCloudPtr& processed_cloud,
    std::unordered_map<uint32_t, std::vector<size_t>>& label_to_points) {
    
    if (labeled_cloud->size() == processed_cloud->size()) {
        for (size_t i = 0; i < labeled_cloud->size(); ++i) {
            uint32_t label = labeled_cloud->points[i].label;
            label_to_points[label].push_back(i);
        }
    }
}

bool SupervoxelProcessor::computeConvexHull(uint32_t supervoxel_id, 
                                           const PointCloudPtr& points, 
                                           ConvexHullData& hull_data) {
    auto start = std::chrono::high_resolution_clock::now();
    
    pcl::ConvexHull<PointT> chull;
    chull.setInputCloud(points);
    
    if (params_.use_2d_convex_hull) {
        chull.setDimension(2);
    }
    
    bool success = false;
    
    // 临时抑制 qhull 的 stderr 输出(避免大量无用的错误日志)
    int stderr_backup = dup(STDERR_FILENO);
    int devnull = open("/dev/null", O_WRONLY);
    dup2(devnull, STDERR_FILENO);
    close(devnull);
    
    try {
        chull.reconstruct(*hull_data.hull_points, hull_data.polygons);
        success = !hull_data.hull_points->empty() && !hull_data.polygons.empty();
    } catch (const std::exception& e) {
        success = false;
    } catch (...) {
        success = false;
    }
    
    // 恢复 stderr
    dup2(stderr_backup, STDERR_FILENO);
    close(stderr_backup);
    
    auto end = std::chrono::high_resolution_clock::now();
    hull_data.computation_time_ms = std::chrono::duration<double, std::milli>(end - start).count();
    
    return success;
}

void SupervoxelProcessor::generateColoredCloud(const pcl::PointCloud<pcl::PointXYZL>::Ptr& labeled_cloud) {
    colored_cloud_->clear();
    colored_cloud_->header = labeled_cloud->header;
    colored_cloud_->width = labeled_cloud->width;
    colored_cloud_->height = labeled_cloud->height;
    colored_cloud_->is_dense = labeled_cloud->is_dense;
    colored_cloud_->points.resize(labeled_cloud->points.size());
    
    for (size_t i = 0; i < labeled_cloud->points.size(); ++i) {
        pcl::PointXYZI& point = colored_cloud_->points[i];
        point.x = labeled_cloud->points[i].x;
        point.y = labeled_cloud->points[i].y;
        point.z = labeled_cloud->points[i].z;
        
        uint32_t label = labeled_cloud->points[i].label;
        // 使用 intensity 存储标签值（用于可视化）
        point.intensity = static_cast<float>(label);
    }
}

bool SupervoxelProcessor::isPointSetDegenerate(const PointCloudPtr& points) const {
    if (points->size() < params_.min_points_for_hull) {
        return true;
    }
    
    // 计算包围盒
    float min_x = std::numeric_limits<float>::max();
    float max_x = std::numeric_limits<float>::lowest();
    float min_y = std::numeric_limits<float>::max();
    float max_y = std::numeric_limits<float>::lowest();
    float min_z = std::numeric_limits<float>::max();
    float max_z = std::numeric_limits<float>::lowest();
    
    for (const auto& pt : points->points) {
        min_x = std::min(min_x, pt.x);
        max_x = std::max(max_x, pt.x);
        min_y = std::min(min_y, pt.y);
        max_y = std::max(max_y, pt.y);
        min_z = std::min(min_z, pt.z);
        max_z = std::max(max_z, pt.z);
    }
    
    // 检查是否所有维度都太小（完全退化）
    float tolerance = 1e-6f;
    float x_range = max_x - min_x;
    float y_range = max_y - min_y;
    float z_range = max_z - min_z;
    
    return (x_range < tolerance && y_range < tolerance && z_range < tolerance);
}

const std::vector<ConvexHullData>& SupervoxelProcessor::getConvexHulls() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return convex_hulls_;
}

const ProcessingStats& SupervoxelProcessor::getProcessingStats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_;
}

pcl::PointCloud<pcl::PointXYZI>::Ptr SupervoxelProcessor::getColoredCloud() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return colored_cloud_;
}

void SupervoxelProcessor::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    convex_hulls_.clear();
    colored_cloud_->clear();
    stats_ = ProcessingStats();
}

void SupervoxelProcessor::updateParams(const SupervoxelParams& params) {
    std::lock_guard<std::mutex> lock(mutex_);
    params_ = params;
}

const SupervoxelParams& SupervoxelProcessor::getParams() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return params_;
}

} // namespace super_voxel
