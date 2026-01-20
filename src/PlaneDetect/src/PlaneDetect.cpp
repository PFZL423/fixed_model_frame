#include "PlaneDetect/PlaneDetect.h"
#include "PlaneDetect/PlaneDetect.cuh"
#include <pcl/common/io.h>
#include <thrust/copy.h>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <iomanip>
template <typename pointT>
PlaneDetect<pointT>::PlaneDetect(const DetectorParams &params) : params_(params)
{
}
template <typename pointT>
PlaneDetect<pointT>::~PlaneDetect(){
}

template <typename PointT>
bool PlaneDetect<PointT>::processCloud(const typename pcl::PointCloud<PointT>::ConstPtr &input_cloud)
{
    if (!input_cloud || input_cloud->empty())
        return false;

    auto total_start = std::chrono::high_resolution_clock::now();

    detected_primitives_.clear();

    // Step 1: PCL转换和GPU上传
    auto convert_start = std::chrono::high_resolution_clock::now();
    convertPCLtoGPU(input_cloud);
    auto convert_end = std::chrono::high_resolution_clock::now();
    float convert_time = std::chrono::duration<float, std::milli>(convert_end - convert_start).count();

    // Step 2: 主要的平面检测
    auto detect_start = std::chrono::high_resolution_clock::now();
    findPlanes_BatchGPU();
    auto detect_end = std::chrono::high_resolution_clock::now();
    float detect_time = std::chrono::duration<float, std::milli>(detect_end - detect_start).count();

    auto total_end = std::chrono::high_resolution_clock::now();
    float total_time = std::chrono::duration<float, std::milli>(total_end - total_start).count();

    // 关键修复：确保所有 GPU 操作完成后再返回
    // 避免与后续的 QuadricDetect GPU 操作产生冲突
    cudaDeviceSynchronize();

    std::cout << "[PlaneDetect] Timing breakdown:" << std::endl;
    std::cout << "  PCL->GPU convert: " << convert_time << " ms" << std::endl;
    std::cout << "  Plane detection: " << detect_time << " ms" << std::endl;
    std::cout << "  Total: " << total_time << " ms" << std::endl;

    return true;
}
template <typename PointT>
void PlaneDetect<PointT>::convertPCLtoGPU(const typename pcl::PointCloud<PointT>::ConstPtr &cloud)
{
    auto total_start = std::chrono::high_resolution_clock::now();

    // Step 1: CPU数据转换
    auto cpu_convert_start = std::chrono::high_resolution_clock::now();
    std::vector<GPUPoint3f> h_points;
    h_points.reserve(cloud->size());

    for (const auto &pt : cloud->points)
    {
        h_points.push_back(GPUPoint3f{pt.x, pt.y, pt.z, pt.rgb});  // 保存RGB信息
    }
    auto cpu_convert_end = std::chrono::high_resolution_clock::now();
    float cpu_convert_time = std::chrono::duration<float, std::milli>(cpu_convert_end - cpu_convert_start).count();

    // Step 2: 上传到GPU
    auto gpu_upload_start = std::chrono::high_resolution_clock::now();
    uploadPointsToGPU(h_points);
    auto gpu_upload_end = std::chrono::high_resolution_clock::now();
    float gpu_upload_time = std::chrono::duration<float, std::milli>(gpu_upload_end - gpu_upload_start).count();

    auto total_end = std::chrono::high_resolution_clock::now();
    float total_time = std::chrono::duration<float, std::milli>(total_end - total_start).count();

    if (params_.verbosity > 0)
    {
        std::cout << "[convertPCLtoGPU] CPU convert: " << cpu_convert_time << " ms" << std::endl;
        std::cout << "[convertPCLtoGPU] GPU upload: " << gpu_upload_time << " ms" << std::endl;
        std::cout << "[convertPCLtoGPU] Total time: " << total_time << " ms" << std::endl;
        std::cout << "[convertPCLtoGPU] Uploaded " << cloud->size() << " points to GPU." << std::endl;
    }
}

template <typename PointT>
void PlaneDetect<PointT>::findPlanes_BatchGPU()
{
    auto total_detect_start = std::chrono::high_resolution_clock::now();
    const int batch_size = params_.batch_size;
    const int max_iterations = 10;

    // Step 1: 初始化GPU内存
    auto init_start = std::chrono::high_resolution_clock::now();
    initializeGPUMemory(batch_size);
    launchInitCurandStates(batch_size);
    auto init_end = std::chrono::high_resolution_clock::now();
    float init_time = std::chrono::duration<float, std::milli>(init_end - init_start).count();

    size_t remaining_points = d_remaining_indices_.size();
    size_t min_points = static_cast<size_t>(params_.min_remaining_points_percentage * d_all_points_.size());

    int iteration = 0;

    if (params_.verbosity > 0)
    {
        std::cout << "[findPlanes_BatchGPU] 开始检测，总点数: " << d_all_points_.size()
                  << ", 最小剩余点数: " << min_points << std::endl;
    }

    float total_sampling_time = 0.0f;
    float total_inlier_count_time = 0.0f;
    float total_best_model_time = 0.0f;
    float total_extract_inliers_time = 0.0f;
    float total_remove_points_time = 0.0f;

    while (remaining_points >= min_points && iteration < max_iterations)
    {
        if (params_.verbosity > 0)
        {
            std::cout << "== 第 " << iteration + 1 << " 次迭代，剩余点数: " << remaining_points << " ==" << std::endl;
        }

        // Step 2: 采样3点并直接拟合平面 (简化版，无需矩阵构建)
        auto sampling_start = std::chrono::high_resolution_clock::now();
        launchSampleAndFitPlanes(batch_size);
        auto sampling_end = std::chrono::high_resolution_clock::now();
        float sampling_time = std::chrono::duration<float, std::milli>(sampling_end - sampling_start).count();
        total_sampling_time += sampling_time;

        // Step 3: 计算内点数 (跳过反幂迭代，直接验证)
        auto inlier_count_start = std::chrono::high_resolution_clock::now();
        launchCountInliersBatch(batch_size);
        auto inlier_count_end = std::chrono::high_resolution_clock::now();
        float inlier_count_time = std::chrono::duration<float, std::milli>(inlier_count_end - inlier_count_start).count();
        total_inlier_count_time += inlier_count_time;

        // Step 4: 找最优模型
        auto best_model_start = std::chrono::high_resolution_clock::now();
        launchFindBestModel(batch_size);
        auto best_model_end = std::chrono::high_resolution_clock::now();
        float best_model_time = std::chrono::duration<float, std::milli>(best_model_end - best_model_start).count();
        total_best_model_time += best_model_time;

        // 获取最优结果
        thrust::host_vector<int> h_best_index(1);
        thrust::host_vector<int> h_best_count(1);
        getBestModelResults(h_best_index, h_best_count);

        int best_count = h_best_count[0];
        int best_model_idx = h_best_index[0];

        if (best_count < params_.min_plane_inlier_count_absolute)
        {
            if (params_.verbosity > 0)
            {
                std::cout << "最优模型内点数不足 (" << best_count << " < " 
                         << params_.min_plane_inlier_count_absolute << ")，结束检测" << std::endl;
            }
            break;
        }

        // Step 5: 获取最优平面模型
        thrust::host_vector<GPUPlaneModel> h_best_model(1);
        thrust::copy_n(d_batch_models_.begin() + best_model_idx, 1, h_best_model.begin());
        GPUPlaneModel best_gpu_model = h_best_model[0];

        // 输出最优模型详情
        if (params_.verbosity > 0)
        {
            std::cout << "最优平面 [" << best_model_idx << "]: 内点数=" << best_count
                      << ", 系数=[" << std::fixed << std::setprecision(4)
                      << best_gpu_model.coeffs[0] << ", " << best_gpu_model.coeffs[1] << ", "
                      << best_gpu_model.coeffs[2] << ", " << best_gpu_model.coeffs[3] << "]" << std::endl;
        }

        // Step 6: 提取内点索引
        auto extract_inliers_start = std::chrono::high_resolution_clock::now();
        launchExtractInliers(&best_gpu_model);
        auto extract_inliers_end = std::chrono::high_resolution_clock::now();
        float extract_inliers_time = std::chrono::duration<float, std::milli>(extract_inliers_end - extract_inliers_start).count();
        total_extract_inliers_time += extract_inliers_time;

        // Step 7: 构建内点点云
        typename pcl::PointCloud<PointT>::Ptr inlier_cloud = extractInlierCloud();

        // Step 8: 保存检测结果
        DetectedPrimitive<PointT> detected_plane;
        detected_plane.type = "plane";
        // 复制平面系数
        for (int i = 0; i < 4; i++) {
            detected_plane.model_coefficients[i] = best_gpu_model.coeffs[i];
        }
        detected_plane.inliers = inlier_cloud;
        detected_primitives_.push_back(detected_plane);

        if (params_.verbosity > 0)
        {
            std::cout << "已保存第 " << detected_primitives_.size() << " 个平面" << std::endl;
        }

        // Step 9: 移除内点
        auto remove_points_start = std::chrono::high_resolution_clock::now();
        std::vector<int> dummy_vector; // 实际使用GPU数据
        removeFoundPoints(dummy_vector);
        auto remove_points_end = std::chrono::high_resolution_clock::now();
        float remove_points_time = std::chrono::duration<float, std::milli>(remove_points_end - remove_points_start).count();
        total_remove_points_time += remove_points_time;

        // 输出本次迭代的详细计时
        if (params_.verbosity > 0)
        {
            std::cout << "[Iteration " << iteration + 1 << "] Timing breakdown:" << std::endl;
            std::cout << "  Sampling & fitting: " << sampling_time << " ms" << std::endl;
            std::cout << "  Inlier counting: " << inlier_count_time << " ms" << std::endl;
            std::cout << "  Best model finding: " << best_model_time << " ms" << std::endl;
            std::cout << "  Extract inliers: " << extract_inliers_time << " ms" << std::endl;
            std::cout << "  Remove points: " << remove_points_time << " ms" << std::endl;
            float iteration_total = sampling_time + inlier_count_time + 
                                  best_model_time + extract_inliers_time + remove_points_time;
            std::cout << "  Iteration total: " << iteration_total << " ms" << std::endl;
        }

        // 更新循环条件
        remaining_points = d_remaining_indices_.size();
        iteration++;
    }

    auto total_detect_end = std::chrono::high_resolution_clock::now();
    float total_detect_time = std::chrono::duration<float, std::milli>(total_detect_end - total_detect_start).count();

    if (params_.verbosity > 0)
    {
        std::cout << "\n[findPlanes_BatchGPU] Final timing summary:" << std::endl;
        std::cout << "  Initialization: " << init_time << " ms" << std::endl;
        std::cout << "  Total sampling: " << total_sampling_time << " ms" << std::endl;
        std::cout << "  Total inlier counting: " << total_inlier_count_time << " ms" << std::endl;
        std::cout << "  Total best model finding: " << total_best_model_time << " ms" << std::endl;
        std::cout << "  Total extract inliers: " << total_extract_inliers_time << " ms" << std::endl;
        std::cout << "  Total remove points: " << total_remove_points_time << " ms" << std::endl;
        std::cout << "  Total detection time: " << total_detect_time << " ms" << std::endl;
        std::cout << "== 检测完成，共找到 " << detected_primitives_.size() << " 个平面 ==" << std::endl;
    }
}

template <typename PointT>
typename pcl::PointCloud<PointT>::Ptr PlaneDetect<PointT>::extractInlierCloud() const
{
    auto cloud = boost::make_shared<pcl::PointCloud<PointT>>();
    
    // 从GPU获取内点索引
    thrust::host_vector<int> h_inlier_indices(current_inlier_count_);
    thrust::copy_n(d_temp_inlier_indices_.begin(), current_inlier_count_, h_inlier_indices.begin());
    
    // 从GPU获取所有点
    thrust::host_vector<GPUPoint3f> h_all_points(d_all_points_.size());
    thrust::copy(d_all_points_.begin(), d_all_points_.end(), h_all_points.begin());
    
    cloud->points.reserve(current_inlier_count_);
    
    for (int i = 0; i < current_inlier_count_; i++) {
        int idx = h_inlier_indices[i];
        if (idx >= 0 && idx < static_cast<int>(h_all_points.size())) {
            PointT pt;
            pt.x = h_all_points[idx].x;
            pt.y = h_all_points[idx].y;
            pt.z = h_all_points[idx].z;
            pt.rgb = h_all_points[idx].rgb;  // 恢复RGB信息
            cloud->points.push_back(pt);
        }
    }
    
    cloud->width = cloud->points.size();
    cloud->height = 1;
    cloud->is_dense = true;
    
    return cloud;
}

template <typename PointT>
void PlaneDetect<PointT>::removeFoundPoints(const std::vector<int> &indices_to_remove)
{
    auto total_start = std::chrono::high_resolution_clock::now();

    if (d_temp_inlier_indices_.empty() || current_inlier_count_ == 0)
    {
        return;
    }

    if (params_.verbosity > 0)
    {
        std::cout << "[removeFoundPoints] 移除前剩余点数: " << d_remaining_indices_.size() << std::endl;
    }

    // 方案：使用自定义CUDA内核，完全避免Thrust set_difference
    auto kernel_start = std::chrono::high_resolution_clock::now();
    launchRemovePointsKernel();
    auto kernel_end = std::chrono::high_resolution_clock::now();
    float kernel_time = std::chrono::duration<float, std::milli>(kernel_end - kernel_start).count();

    auto total_end = std::chrono::high_resolution_clock::now();
    float total_time = std::chrono::duration<float, std::milli>(total_end - total_start).count();

    if (params_.verbosity > 0)
    {
        std::cout << "[removeFoundPoints] Remove kernel: " << kernel_time << " ms" << std::endl;
        std::cout << "[removeFoundPoints] Total time: " << total_time << " ms" << std::endl;
        std::cout << "[removeFoundPoints] 移除了 " << current_inlier_count_
                  << " 个内点，剩余 " << d_remaining_indices_.size() << " 个点" << std::endl;
    }
}
template <typename PointT>
const std::vector<DetectedPrimitive<PointT>> &PlaneDetect<PointT>::getDetectedPrimitives() const
{
    return detected_primitives_;
}

template <typename PointT>
size_t PlaneDetect<PointT>::getDetectedPlaneCount() const
{
    return detected_primitives_.size();
}

template <typename PointT>
std::vector<float> PlaneDetect<PointT>::getPlaneCoefficients(size_t index) const
{
    std::vector<float> coeffs;
    if (index < detected_primitives_.size()) {
        const auto& primitive = detected_primitives_[index];
        coeffs = {primitive.model_coefficients[0], 
                  primitive.model_coefficients[1], 
                  primitive.model_coefficients[2], 
                  primitive.model_coefficients[3]};
    }
    return coeffs;
}

template <typename PointT>
std::vector<std::vector<float>> PlaneDetect<PointT>::getAllPlaneCoefficients() const
{
    std::vector<std::vector<float>> all_coeffs;
    all_coeffs.reserve(detected_primitives_.size());
    
    for (const auto& primitive : detected_primitives_) {
        std::vector<float> coeffs = {primitive.model_coefficients[0], 
                                     primitive.model_coefficients[1], 
                                     primitive.model_coefficients[2], 
                                     primitive.model_coefficients[3]};
        all_coeffs.push_back(coeffs);
    }
    return all_coeffs;
}

template <typename PointT>
typename pcl::PointCloud<PointT>::Ptr PlaneDetect<PointT>::getPlaneInliers(size_t index) const
{
    typename pcl::PointCloud<PointT>::Ptr empty_cloud(new pcl::PointCloud<PointT>());
    if (index < detected_primitives_.size()) {
        return detected_primitives_[index].inliers;
    }
    return empty_cloud;
}

template <typename PointT>
size_t PlaneDetect<PointT>::getPlaneInlierCount(size_t index) const
{
    if (index < detected_primitives_.size()) {
        return detected_primitives_[index].inliers->size();
    }
    return 0;
}

template <typename PointT>
size_t PlaneDetect<PointT>::getRemainingPointCount() const
{
    return d_remaining_indices_.size();
}
template <typename PointT>
typename pcl::PointCloud<PointT>::Ptr PlaneDetect<PointT>::getFinalCloud() const
{
    // 关键修复：确保所有 GPU 操作完成后再复制数据
    // 因为 thrust 的 device_vector 复制可能是异步的
    cudaDeviceSynchronize();
    
    typename pcl::PointCloud<PointT>::Ptr final_cloud(new pcl::PointCloud<PointT>());
    
    // 如果没有剩余点的索引信息，返回空点云
    if (d_remaining_indices_.empty() || d_all_points_.empty()) {
        return final_cloud;
    }
    
    // 将剩余点的索引从GPU复制到CPU
    thrust::host_vector<int> h_remaining_indices = d_remaining_indices_;
    thrust::host_vector<GPUPoint3f> h_all_points = d_all_points_;
    
    // 再次同步，确保 thrust 的复制完成
    cudaDeviceSynchronize();
    
    // 预分配空间
    final_cloud->points.reserve(h_remaining_indices.size());
    
    // 根据剩余索引构建最终点云
    for (int idx : h_remaining_indices) {
        if (idx >= 0 && idx < (int)h_all_points.size()) {
            PointT point;
            point.x = h_all_points[idx].x;
            point.y = h_all_points[idx].y;  
            point.z = h_all_points[idx].z;
            point.rgb = h_all_points[idx].rgb;  // 恢复RGB信息（关键！）
            final_cloud->points.push_back(point);
        }
    }
    
    // 设置点云属性
    final_cloud->width = final_cloud->points.size();
    final_cloud->height = 1;
    final_cloud->is_dense = true;
    
    return final_cloud;
}

// 显式模板实例化
// template class PlaneDetect<pcl::PointXYZ>;
// template class PlaneDetect<pcl::PointXYZI>;
template class PlaneDetect<pcl::PointXYZRGB>;
template class PlaneDetect<pcl::PointXYZRGBA>;
