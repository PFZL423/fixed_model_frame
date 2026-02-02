#include "gpu_demo/QuadricDetect.h"
#include "gpu_demo/QuadricDetect_kernels.cuh"
#include <pcl/common/io.h>
#include <thrust/copy.h>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <iomanip>

using DetectorParams = quadric::DetectorParams;
using DetectedPrimitive = quadric::DetectedPrimitive;
using GPUtimer=quadric::GPUTimer;

QuadricDetect::QuadricDetect(const DetectorParams &params) : params_(params)
{
    cudaStreamCreate(&stream_);
}

QuadricDetect::~QuadricDetect()
{
    cudaStreamDestroy(stream_);
}

bool QuadricDetect::processCloud(const pcl::PointCloud<pcl::PointXYZI>::ConstPtr &input_cloud)
{
    if (!input_cloud || input_cloud->empty())
        return false;

    //  关键修复：完全同步所有CUDA设备并清除任何潜在错误
    // cudaDeviceSynchronize();  // 等待所有之前的CUDA操作完成
    // cudaGetLastError();        // 清除之前可能存在的CUDA错误状态
    
    auto total_start = std::chrono::high_resolution_clock::now();

    //  关键修复：清空所有GPU状态（防止多帧复用时的数据残留）
    detected_primitives_.clear();
    d_batch_inlier_counts_.clear();
    d_batch_models_.clear();
    d_best_model_index_.clear();
    d_best_model_count_.clear();
    
    // Step 1: PCL转换和GPU上传
    auto convert_start = std::chrono::high_resolution_clock::now();
    convertPCLtoGPU(input_cloud);
    auto convert_end = std::chrono::high_resolution_clock::now();
    float convert_time = std::chrono::duration<float, std::milli>(convert_end - convert_start).count();

    // Step 2: 主要的二次曲面检测
    auto detect_start = std::chrono::high_resolution_clock::now();
    findQuadrics_BatchGPU();
    auto detect_end = std::chrono::high_resolution_clock::now();
    float detect_time = std::chrono::duration<float, std::milli>(detect_end - detect_start).count();

    auto total_end = std::chrono::high_resolution_clock::now();
    float total_time = std::chrono::duration<float, std::milli>(total_end - total_start).count();

    //  关键修复：确保所有 GPU 操作完成
    cudaStreamSynchronize(stream_);
    cudaDeviceSynchronize();
    
    if (params_.verbosity > 0) {
        std::cout << "[QuadricDetect] Timing breakdown:" << std::endl;
        std::cout << "  PCL->GPU convert: " << convert_time << " ms" << std::endl;
        std::cout << "  Quadric detection: " << detect_time << " ms" << std::endl;
        std::cout << "  Total: " << total_time << " ms" << std::endl;
    }

    return true;
}

void QuadricDetect::convertPCLtoGPU(const pcl::PointCloud<pcl::PointXYZI>::ConstPtr &cloud)
{
    auto total_start = std::chrono::high_resolution_clock::now();

    // Step 1: CPU数据转换
    auto cpu_convert_start = std::chrono::high_resolution_clock::now();
    std::vector<GPUPoint3f> h_points;
    h_points.reserve(cloud->size());

    for (const auto &pt : cloud->points)
    {
            // 关键修复：过滤NaN/Inf点
        if (std::isfinite(pt.x) && std::isfinite(pt.y) && std::isfinite(pt.z))
        {
            GPUPoint3f gpu_pt;
            gpu_pt.x = pt.x;
            gpu_pt.y = pt.y;
            gpu_pt.z = pt.z;
            gpu_pt.intensity = pt.intensity;  // 保存强度信息
            h_points.push_back(gpu_pt);
        }
    }
    auto cpu_convert_end = std::chrono::high_resolution_clock::now();
    float cpu_convert_time = std::chrono::duration<float, std::milli>(cpu_convert_end - cpu_convert_start).count();

    // Step 2: GPU上传
    auto gpu_upload_start = std::chrono::high_resolution_clock::now();
    uploadPointsToGPU(h_points);
    auto gpu_upload_end = std::chrono::high_resolution_clock::now();
    float gpu_upload_time = std::chrono::duration<float, std::milli>(gpu_upload_end - gpu_upload_start).count();

    auto total_end = std::chrono::high_resolution_clock::now();
    float total_time = std::chrono::duration<float, std::milli>(total_end - total_start).count();

    if (params_.verbosity > 1)
    {
        std::cout << "[QuadricDetect] PCL转换时间: " << cpu_convert_time << " ms" << std::endl;
        std::cout << "[QuadricDetect] GPU上传时间: " << gpu_upload_time << " ms" << std::endl;
        std::cout << "[QuadricDetect] 转换总时间: " << total_time << " ms" << std::endl;
    }
}

Eigen::Matrix4f QuadricDetect::convertGPUModelToEigen(const GPUQuadricModel &gpu_model)
{
    Eigen::Matrix4f eigen_mat;
    for (int i = 0; i < 16; ++i)
    {
        eigen_mat(i / 4, i % 4) = gpu_model.coeffs[i];
    }
    return eigen_mat;
}

void QuadricDetect::findQuadrics_BatchGPU()
{
    auto total_detect_start = std::chrono::high_resolution_clock::now();

    const int batch_size = 1024;
    const int max_iterations = 3;  // 降低主循环迭代次数，避免剩余点数少时空转

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
        std::cout << "[QuadricDetect] 开始检测，总点数: " << d_all_points_.size()
                  << ", 最小剩余点数: " << min_points << std::endl;
        std::cout << "[QuadricDetect] 初始化GPU内存: " << init_time << " ms" << std::endl;
    }

    float total_sampling_time = 0.0f;
    float total_inverse_power_time = 0.0f;
    float total_inlier_count_time = 0.0f;
    float total_best_model_time = 0.0f;
    float total_extract_inliers_time = 0.0f;
    float total_extract_cloud_time = 0.0f;
    float total_remove_points_time = 0.0f;

    while (remaining_points >= min_points && iteration < max_iterations)
    {
        if (params_.verbosity > 0)
        {
            std::cout << "[QuadricDetect] == 第 " << iteration + 1 << " 次迭代，剩余点数: " << remaining_points << " ==" << std::endl;
        }

        // Step 2: 采样和构建矩阵
        auto sampling_start = std::chrono::high_resolution_clock::now();
        launchSampleAndBuildMatrices(batch_size);
        auto sampling_end = std::chrono::high_resolution_clock::now();
        float sampling_time = std::chrono::duration<float, std::milli>(sampling_end - sampling_start).count();
        total_sampling_time += sampling_time;
        
        // Step 3: 批量反幂迭代
        auto inverse_power_start = std::chrono::high_resolution_clock::now();
        performBatchInversePowerIteration(batch_size);
        auto inverse_power_end = std::chrono::high_resolution_clock::now();
        float inverse_power_time = std::chrono::duration<float, std::milli>(inverse_power_end - inverse_power_start).count();
        total_inverse_power_time += inverse_power_time;

        // Step 4: 计算内点数
        auto inlier_count_start = std::chrono::high_resolution_clock::now();
        launchCountInliersBatch(batch_size);
        auto inlier_count_end = std::chrono::high_resolution_clock::now();
        float inlier_count_time = std::chrono::duration<float, std::milli>(inlier_count_end - inlier_count_start).count();
        total_inlier_count_time += inlier_count_time;

        // Step 5: 找最优模型
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

        // 如果剩余点数已经很少，且内点数不足，立即停止
        if (remaining_points < min_points * 2 && best_count < params_.min_quadric_inlier_count_absolute) {
            if (params_.verbosity > 0) {
                std::cout << "[QuadricDetect] 剩余点数过少且内点不足，提前结束检测" << std::endl;
            }
            break;
        }

        if (best_count < params_.min_quadric_inlier_count_absolute)
        {
            if (params_.verbosity > 0)
            {
                std::cout << "[QuadricDetect] 最优模型内点数不足 (" << best_count 
                          << " < " << params_.min_quadric_inlier_count_absolute << ")，结束检测" << std::endl;
            }
            break;
        }

        // Step 6: 获取最优模型
        thrust::host_vector<GPUQuadricModel> h_best_model(1);
        thrust::copy_n(d_batch_models_.begin() + best_model_idx, 1, h_best_model.begin());
        GPUQuadricModel best_gpu_model = h_best_model[0];

        // 添加：输出最优模型详情
        if (params_.verbosity > 0)
        {
            outputBestModelDetails(best_gpu_model, best_count, best_model_idx, iteration + 1);
        }

        // Step 7: 提取内点索引
        auto extract_inliers_start = std::chrono::high_resolution_clock::now();
        launchExtractInliers(&best_gpu_model);
        auto extract_inliers_end = std::chrono::high_resolution_clock::now();
        float extract_inliers_time = std::chrono::duration<float, std::milli>(extract_inliers_end - extract_inliers_start).count();
        total_extract_inliers_time += extract_inliers_time;

        // Step 8: 构建内点点云
        auto extract_cloud_start = std::chrono::high_resolution_clock::now();
        pcl::PointCloud<pcl::PointXYZI>::Ptr inlier_cloud = extractInlierCloud();
        auto extract_cloud_end = std::chrono::high_resolution_clock::now();
        float extract_cloud_time = std::chrono::duration<float, std::milli>(extract_cloud_end - extract_cloud_start).count();
        total_extract_cloud_time += extract_cloud_time;

        // Step 9: 保存检测结果
        DetectedPrimitive detected_quadric;
        detected_quadric.type = "quadric";
        detected_quadric.model_coefficients = convertGPUModelToEigen(best_gpu_model);
        detected_quadric.inliers = inlier_cloud;
        detected_primitives_.push_back(detected_quadric);

        // Step 10: 移除内点
        auto remove_points_start = std::chrono::high_resolution_clock::now();
        std::vector<int> dummy_vector; // 实际使用GPU数据
        removeFoundPoints(dummy_vector);
        auto remove_points_end = std::chrono::high_resolution_clock::now();
        float remove_points_time = std::chrono::duration<float, std::milli>(remove_points_end - remove_points_start).count();
        total_remove_points_time += remove_points_time;

        if (params_.verbosity > 0)
        {
            float iteration_total = sampling_time + inverse_power_time + inlier_count_time + 
                                  best_model_time + extract_inliers_time + extract_cloud_time + remove_points_time;
            std::cout << "[QuadricDetect] 已保存第 " << detected_primitives_.size() << " 个二次曲面" << std::endl;
            std::cout << "[QuadricDetect] 迭代 " << iteration + 1 << " 时间: " << iteration_total << " ms" << std::endl;
            std::cout << "  - 采样和构建矩阵: " << sampling_time << " ms" << std::endl;
            std::cout << "  - 反幂迭代: " << inverse_power_time << " ms" << std::endl;
            std::cout << "  - 计算内点数: " << inlier_count_time << " ms" << std::endl;
            std::cout << "  - 找最优模型: " << best_model_time << " ms" << std::endl;
            std::cout << "  - 提取内点索引: " << extract_inliers_time << " ms" << std::endl;
            std::cout << "  - 构建内点点云: " << extract_cloud_time << " ms" << std::endl;
            std::cout << "  - 移除内点: " << remove_points_time << " ms" << std::endl;
        }


        // 更新循环条件
        remaining_points = d_remaining_indices_.size();
        iteration++;
    }

    auto total_detect_end = std::chrono::high_resolution_clock::now();
    float total_detect_time = std::chrono::duration<float, std::milli>(total_detect_end - total_detect_start).count();

    if (params_.verbosity > 0)
    {
        std::cout << "[QuadricDetect] == 检测完成，共找到 " << detected_primitives_.size() << " 个二次曲面 ==" << std::endl;
        std::cout << "[QuadricDetect] 总时间统计:" << std::endl;
        std::cout << "  - 初始化: " << init_time << " ms" << std::endl;
        std::cout << "  - 采样和构建矩阵: " << total_sampling_time << " ms" << std::endl;
        std::cout << "  - 反幂迭代: " << total_inverse_power_time << " ms" << std::endl;
        std::cout << "  - 计算内点数: " << total_inlier_count_time << " ms" << std::endl;
        std::cout << "  - 找最优模型: " << total_best_model_time << " ms" << std::endl;
        std::cout << "  - 提取内点索引: " << total_extract_inliers_time << " ms" << std::endl;
        std::cout << "  - 构建内点点云: " << total_extract_cloud_time << " ms" << std::endl;
        std::cout << "  - 移除内点: " << total_remove_points_time << " ms" << std::endl;
        std::cout << "  - 总检测时间: " << total_detect_time << " ms" << std::endl;
    }
}

void QuadricDetect::performBatchInversePowerIteration(int batch_size)
{
    auto total_start = std::chrono::high_resolution_clock::now();

    if (params_.verbosity > 1)
    {
        std::cout << "[QuadricDetect] 启动批量反幂迭代，batch_size=" << batch_size << std::endl;
    }

    // Step 1: 从9×10矩阵计算10×10的A^T*A矩阵
    auto ata_start = std::chrono::high_resolution_clock::now();
    launchComputeATA(batch_size);
    auto ata_end = std::chrono::high_resolution_clock::now();
    float ata_time = std::chrono::duration<float, std::milli>(ata_end - ata_start).count();

    // Step 2: 对A^T*A进行QR分解
    auto qr_start = std::chrono::high_resolution_clock::now();
    launchBatchQR(batch_size);
    auto qr_end = std::chrono::high_resolution_clock::now();
    float qr_time = std::chrono::duration<float, std::milli>(qr_end - qr_start).count();

    // Step 3: 反幂迭代求最小特征向量
    auto power_start = std::chrono::high_resolution_clock::now();
    launchBatchInversePower(batch_size);
    auto power_end = std::chrono::high_resolution_clock::now();
    float power_time = std::chrono::duration<float, std::milli>(power_end - power_start).count();

    // Step 4: 将特征向量转换为二次曲面模型
    auto extract_start = std::chrono::high_resolution_clock::now();
    launchExtractQuadricModels(batch_size);
    auto extract_end = std::chrono::high_resolution_clock::now();
    float extract_time = std::chrono::duration<float, std::milli>(extract_end - extract_start).count();

    auto total_end = std::chrono::high_resolution_clock::now();
    float total_time = std::chrono::duration<float, std::milli>(total_end - total_start).count();

    if (params_.verbosity > 1)
    {
        std::cout << "[QuadricDetect] 反幂迭代详细时间:" << std::endl;
        std::cout << "  - Compute A^T*A: " << ata_time << " ms" << std::endl;
        std::cout << "  - QR decomposition: " << qr_time << " ms" << std::endl;
        std::cout << "  - Inverse power iteration: " << power_time << " ms" << std::endl;
        std::cout << "  - Extract quadric models: " << extract_time << " ms" << std::endl;
        std::cout << "  - Total: " << total_time << " ms" << std::endl;
    }

    // 验证反幂迭代结果（仅在详细模式下）
    if (params_.verbosity > 1)
    {
        validateInversePowerResults(batch_size);
    }
}


void QuadricDetect::removeFoundPoints(const std::vector<int> &indices_to_remove)
{
    auto total_start = std::chrono::high_resolution_clock::now();

    if (d_temp_inlier_indices_.empty() || current_inlier_count_ == 0)
    {
        return;
    }

    if (params_.verbosity > 1)
    {
        std::cout << "[QuadricDetect] 移除前剩余点数: " << d_remaining_indices_.size() << std::endl;
    }

    // 🚀 方案：使用自定义CUDA内核，完全避免Thrust set_difference
    auto kernel_start = std::chrono::high_resolution_clock::now();
    launchRemovePointsKernel();
    auto kernel_end = std::chrono::high_resolution_clock::now();
    float kernel_time = std::chrono::duration<float, std::milli>(kernel_end - kernel_start).count();

    auto total_end = std::chrono::high_resolution_clock::now();
    float total_time = std::chrono::duration<float, std::milli>(total_end - total_start).count();

    if (params_.verbosity > 1)
    {
        std::cout << "[QuadricDetect] 移除内点时间: " << kernel_time << " ms" << std::endl;
        std::cout << "[QuadricDetect] 移除了 " << current_inlier_count_
                  << " 个内点，剩余 " << d_remaining_indices_.size() << " 个点" << std::endl;
    }
}


const std::vector<DetectedPrimitive, Eigen::aligned_allocator<DetectedPrimitive>> &
QuadricDetect::getDetectedPrimitives() const
{
    return detected_primitives_;
}

pcl::PointCloud<pcl::PointXYZI>::Ptr QuadricDetect::getFinalCloud() const
{
    pcl::PointCloud<pcl::PointXYZI>::Ptr final_cloud(new pcl::PointCloud<pcl::PointXYZI>());

    //  关键修复：确保所有 GPU 操作完成后再复制数据到 Host
    cudaStreamSynchronize(stream_);
    cudaDeviceSynchronize();  // 全局同步，确保所有设备操作完成
    
    //  检查 CUDA 错误状态
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "[getFinalCloud]  CUDA错误在同步后检测到: " 
                  << cudaGetErrorString(err) << std::endl;
        return final_cloud;
    }
    
    if (d_remaining_indices_.empty()) {
        return final_cloud;
    }
    
    //  安全的 Device→Host 拷贝，带错误检查
    thrust::host_vector<int> h_remaining_indices;
    thrust::host_vector<GPUPoint3f> h_all_points;
    
    try {
        h_remaining_indices = d_remaining_indices_;
        h_all_points = d_all_points_;
    } catch (const thrust::system_error &e) {
        std::cerr << "[getFinalCloud]  Thrust拷贝失败: " << e.what() << std::endl;
        err = cudaGetLastError();
        std::cerr << "[getFinalCloud] CUDA错误: " << cudaGetErrorString(err) << std::endl;
        return final_cloud;
    }
    
    final_cloud->reserve(h_remaining_indices.size());
    
    for (int idx : h_remaining_indices) {
        if (idx >= 0 && idx < h_all_points.size()) {
            pcl::PointXYZI pt;
            pt.x = h_all_points[idx].x;
            pt.y = h_all_points[idx].y;
            pt.z = h_all_points[idx].z;
            pt.intensity = h_all_points[idx].intensity;  // 恢复强度信息
            final_cloud->push_back(pt);
        }
    }
    
    final_cloud->width = final_cloud->size();
    final_cloud->height = 1;
    final_cloud->is_dense = true;
    
    
    return final_cloud;
}

pcl::PointCloud<pcl::PointXYZI>::Ptr QuadricDetect::extractInlierCloud() const
{
    pcl::PointCloud<pcl::PointXYZI>::Ptr inlier_cloud(new pcl::PointCloud<pcl::PointXYZI>());

    if (d_temp_inlier_indices_.empty() || current_inlier_count_ == 0)
    {
        return inlier_cloud;
    }

    //  确保 GPU 操作完成
    cudaStreamSynchronize(stream_);
    cudaDeviceSynchronize();
    
    //  检查 CUDA 错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "[extractInlierCloud]  CUDA错误: " 
                  << cudaGetErrorString(err) << std::endl;
        return inlier_cloud;
    }

    // 从GPU拷贝内点索引到CPU（带异常处理）
    thrust::host_vector<int> h_inlier_indices;
    thrust::host_vector<GPUPoint3f> h_all_points;
    
    try {
        h_inlier_indices = d_temp_inlier_indices_;
        h_all_points = d_all_points_;
    } catch (const thrust::system_error &e) {
        std::cerr << "[extractInlierCloud]  Thrust拷贝失败: " << e.what() << std::endl;
        err = cudaGetLastError();
        std::cerr << "[extractInlierCloud] CUDA错误: " << cudaGetErrorString(err) << std::endl;
        return inlier_cloud;
    }

    inlier_cloud->reserve(current_inlier_count_);

    // 根据索引构建点云
    for (int i = 0; i < current_inlier_count_; ++i)
    {
        int idx = h_inlier_indices[i];
        if (idx >= 0 && idx < h_all_points.size())
        {
            pcl::PointXYZI pt;
            pt.x = h_all_points[idx].x;
            pt.y = h_all_points[idx].y;
            pt.z = h_all_points[idx].z;
            pt.intensity = h_all_points[idx].intensity;  // 恢复强度信息（关键！）
            inlier_cloud->push_back(pt);
        }
    }

    inlier_cloud->width = inlier_cloud->size();
    inlier_cloud->height = 1;
    inlier_cloud->is_dense = true;

    if (params_.verbosity > 1)
    {
        std::cout << "[QuadricDetect] 构建了包含 " << inlier_cloud->size() << " 个内点的点云" << std::endl;
    }

    return inlier_cloud;
}






//  新增函数：验证反幂迭代结果
void QuadricDetect::validateInversePowerResults(int batch_size)
{
    std::cout << "[QuadricDetect] 验证反幂迭代结果..." << std::endl;

    // 检查前几个特征向量和模型
    int check_count = std::min(3, batch_size);

    // 1. 检查特征向量
    thrust::host_vector<float> h_eigenvectors(check_count * 10);
    thrust::copy_n(d_batch_eigenvectors_.begin(), check_count * 10, h_eigenvectors.begin());

    // 2. 检查生成的模型
    thrust::host_vector<GPUQuadricModel> h_models(check_count);
    thrust::copy_n(d_batch_models_.begin(), check_count, h_models.begin());

    bool all_valid = true;

    for (int i = 0; i < check_count; ++i)
    {
        std::cout << "[QuadricDetect] 模型 " << i << ":" << std::endl;

        // 检查特征向量
        float *eigenvec = &h_eigenvectors[i * 10];
        float norm_sq = 0.0f;
        bool has_nan = false;

        for (int j = 0; j < 10; ++j)
        {
            norm_sq += eigenvec[j] * eigenvec[j];
            if (!std::isfinite(eigenvec[j]) || std::isnan(eigenvec[j]))
            {
                has_nan = true;
            }
        }

        float norm = std::sqrt(norm_sq);

        if (has_nan)
        {
            std::cout << "[QuadricDetect]    特征向量包含NaN/Inf值" << std::endl;
            all_valid = false;
        }
        else if (norm < 1e-12f)
        {
            std::cout << "[QuadricDetect]    特征向量模长过小: " << norm << std::endl;
            all_valid = false;
        }
        else
        {
            std::cout << "[QuadricDetect]    特征向量正常，模长: " << norm << std::endl;
        }

        // 检查模型系数
        const GPUQuadricModel &model = h_models[i];
        bool model_valid = true;
        float coeff_sum = 0.0f;

        for (int j = 0; j < 16; ++j)
        {
            coeff_sum += std::abs(model.coeffs[j]);
            if (!std::isfinite(model.coeffs[j]) || std::isnan(model.coeffs[j]))
            {
                model_valid = false;
                break;
            }
        }

        if (!model_valid)
        {
            std::cout << "[QuadricDetect]    模型系数包含NaN/Inf值" << std::endl;
            all_valid = false;
        }
        else if (coeff_sum < 1e-12f)
        {
            std::cout << "[QuadricDetect]    模型系数全为零" << std::endl;
            all_valid = false;
        }
        else
        {
            std::cout << "[QuadricDetect]    模型系数正常，系数和: " << coeff_sum << std::endl;
        }

        // 显示前几个系数
        if (params_.verbosity > 1)
        {
            std::cout << "[QuadricDetect]    前6个系数: [";
            for (int j = 0; j < 6; ++j)
            {
                std::cout << model.coeffs[j];
                if (j < 5)
                    std::cout << ", ";
            }
            std::cout << "]" << std::endl;
        }
    }

    if (all_valid)
    {
        std::cout << "[QuadricDetect] 反幂迭代结果验证通过" << std::endl;
    }
    else
    {
        std::cout << "[QuadricDetect] 反幂迭代结果存在问题，请检查算法实现" << std::endl;
    }
}

// 🆕 新增函数：输出最优模型详情
void QuadricDetect::outputBestModelDetails(const GPUQuadricModel &best_model, int inlier_count, int model_idx, int iteration)
{
    std::cout << "\n[QuadricDetect] ========== 第" << iteration << "次迭代最优模型详情 ==========" << std::endl;
    std::cout << "[QuadricDetect] 模型索引: " << model_idx << " (在1024个候选中)" << std::endl;
    std::cout << "[QuadricDetect] 内点数量: " << inlier_count << std::endl;
    std::cout << "[QuadricDetect] 内点比例: " << std::fixed << std::setprecision(2) 
              << (100.0 * inlier_count / d_remaining_indices_.size()) << "%" << std::endl;

    // 转换为Eigen矩阵便于显示（仅在详细模式下）
    if (params_.verbosity > 1)
    {
        Eigen::Matrix4f Q = convertGPUModelToEigen(best_model);
        std::cout << "[QuadricDetect] 二次曲面矩阵 Q:" << std::endl;
        for (int i = 0; i < 4; ++i)
        {
            std::cout << "[QuadricDetect]   [";
            for (int j = 0; j < 4; ++j)
            {
                std::cout << std::setw(10) << std::setprecision(6) << std::fixed << Q(i, j);
                if (j < 3)
                    std::cout << ", ";
            }
            std::cout << "]" << std::endl;
        }

        // 分析二次曲面类型（简单判断）
        float det = Q.determinant();
        std::cout << "[QuadricDetect] 矩阵行列式: " << det << std::endl;

        // 检查对角线元素符号
        int pos_diag = 0, neg_diag = 0, zero_diag = 0;
        for (int i = 0; i < 3; ++i) // 只看前3×3部分
        {
            if (Q(i, i) > 1e-6f)
                pos_diag++;
            else if (Q(i, i) < -1e-6f)
                neg_diag++;
            else
                zero_diag++;
        }

        std::cout << "[QuadricDetect] 对角线符号分布: +" << pos_diag << " / -" << neg_diag << " / 0:" << zero_diag;

        // 简单的曲面类型推断
        if (pos_diag == 3 || neg_diag == 3)
        {
            std::cout << " → 可能是椭球面" << std::endl;
        }
        else if ((pos_diag == 2 && neg_diag == 1) || (pos_diag == 1 && neg_diag == 2))
        {
            std::cout << " → 可能是双曲面" << std::endl;
        }
        else if (zero_diag > 0)
        {
            std::cout << " → 可能是抛物面或退化曲面" << std::endl;
        }
        else
        {
            std::cout << " → 曲面类型待进一步分析" << std::endl;
        }
    }

    std::cout << "[QuadricDetect] ================================================" << std::endl;
}



//重载实现
// bool QuadricDetect::processCloud(const thrust::device_vector<GPUPoint3f> &input_cloud)
// {
//     if (input_cloud.empty())
//         return false;

//     auto total_start = std::chrono::high_resolution_clock::now();

//     detected_primitives_.clear();

//     // Step 1: GPU数据直接赋值 (无CPU-GPU传输)
//     auto convert_start = std::chrono::high_resolution_clock::now();
//     uploadPointsToGPU(input_cloud);
//     auto convert_end = std::chrono::high_resolution_clock::now();
//     float convert_time = std::chrono::duration<float, std::milli>(convert_end - convert_start).count();

//     // Step 2: 主要的二次曲面检测
//     auto detect_start = std::chrono::high_resolution_clock::now();
//     findQuadrics_BatchGPU();
//     auto detect_end = std::chrono::high_resolution_clock::now();
//     float detect_time = std::chrono::duration<float, std::milli>(detect_end - detect_start).count();

//     auto total_end = std::chrono::high_resolution_clock::now();
//     float total_time = std::chrono::duration<float, std::milli>(total_end - total_start).count();

//     std::cout << "[QuadricDetect] GPU-Direct Timing breakdown:" << std::endl;
//     std::cout << "  GPU data assignment: " << convert_time << " ms" << std::endl;
//     std::cout << "  Quadric detection: " << detect_time << " ms" << std::endl;
//     std::cout << "  Total: " << total_time << " ms" << std::endl;

//     return true;
// }