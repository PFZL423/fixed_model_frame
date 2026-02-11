#include "gpu_demo/QuadricDetect.h"
#include "gpu_demo/QuadricDetect_kernels.cuh"
#include <pcl/common/io.h>
#include <thrust/copy.h>
#include <thrust/device_ptr.h>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <numeric>
#include <geometry_msgs/Point.h>
#include <ros/ros.h>

using DetectorParams = quadric::DetectorParams;
using DetectedPrimitive = quadric::DetectedPrimitive;
using GPUtimer=quadric::GPUTimer;

QuadricDetect::QuadricDetect(const DetectorParams &params) : params_(params)
{
    cudaStreamCreate(&stream_);
    owns_stream_ = true;
    is_external_memory_ = false;
    d_external_points_ = nullptr;
    d_valid_mask_ = nullptr;
    max_points_capacity_ = 0;
    original_total_count_ = 0;
    cusolver_handle_ = nullptr;
}

QuadricDetect::~QuadricDetect()
{
    if (owns_stream_ && stream_ != nullptr)
    {
        cudaStreamDestroy(stream_);
    }
    if (cusolver_handle_ != nullptr)
    {
        cusolverDnDestroy(cusolver_handle_);
    }
    if (d_valid_mask_ != nullptr)
    {
        cudaFree(d_valid_mask_);
        d_valid_mask_ = nullptr;
    }
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

    // Step 1: 初始化GPU内存（仅在需要时）
    auto init_start = std::chrono::high_resolution_clock::now();
    
    // 检查是否已经初始化（通过检查 d_batch_models_ 的大小）
    bool needs_init = (d_batch_models_.size() != static_cast<size_t>(batch_size));
    if (needs_init)
    {
        initializeGPUMemory(batch_size);
    }
    
    launchInitCurandStates(batch_size);
    auto init_end = std::chrono::high_resolution_clock::now();
    float init_time = std::chrono::duration<float, std::milli>(init_end - init_start).count();

    size_t remaining_points = d_remaining_indices_.size();
    // 获取点云总数（支持外部内存）
    size_t total_points = is_external_memory_ ? d_remaining_indices_.size() : d_all_points_.size();
    size_t min_points = static_cast<size_t>(params_.min_remaining_points_percentage * total_points);

    int iteration = 0;

    if (params_.verbosity > 0)
    {
        std::cout << "[QuadricDetect] 开始检测，总点数: " << total_points
                  << ", 最小剩余点数: " << min_points << std::endl;
        std::cout << "[QuadricDetect] 初始化GPU内存: " << init_time << " ms" << std::endl;
    }

    float total_sampling_time = 0.0f;
    float total_inlier_count_time = 0.0f;
    float total_coarse_time = 0.0f;
    float total_topk_time = 0.0f;
    float total_fine_time = 0.0f;
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

        // Step 3: 两阶段RANSAC竞速
        // 3.1 粗筛阶段：对batch_size个模型进行子采样计数（2%采样率）
        auto inlier_count_start = std::chrono::high_resolution_clock::now();
        const int coarse_stride = 50;  // 2%采样率
        launchCountInliersBatch(batch_size, coarse_stride);  // 粗筛，得到coarse_score
        auto coarse_end = std::chrono::high_resolution_clock::now();
        float coarse_time = std::chrono::duration<float, std::milli>(coarse_end - inlier_count_start).count();

        // 3.2 Top-K选择：选出coarse_score最高的k个模型索引
        const int fine_k = 20;  // 精选阶段候选数量（可从params_读取）
        launchSelectTopKModels(fine_k);
        auto topk_end = std::chrono::high_resolution_clock::now();
        float topk_time = std::chrono::duration<float, std::milli>(topk_end - coarse_end).count();

        // 3.3 精选阶段：对前k个模型进行全量计数
        launchFineCountInliersBatch(fine_k);  // 精选，得到fine_score
        auto fine_end = std::chrono::high_resolution_clock::now();
        float fine_time = std::chrono::duration<float, std::milli>(fine_end - topk_end).count();

        float inlier_count_time = coarse_time + topk_time + fine_time;
        total_inlier_count_time += inlier_count_time;
        total_coarse_time += coarse_time;
        total_topk_time += topk_time;
        total_fine_time += fine_time;

        // 调试信息：计算并打印前几个模型的距离统计
        if (params_.verbosity > 1)
        {
            const int debug_model_count = std::min(3, batch_size);
            std::cout << "[QuadricDetect] 计算前 " << debug_model_count << " 个模型的距离统计..." << std::endl;
            
            // 获取前几个模型
            thrust::host_vector<GPUQuadricModel> h_models(debug_model_count);
            thrust::copy_n(d_batch_models_.begin(), debug_model_count, h_models.begin());
            
            // 获取一些采样点来计算距离
            const int sample_point_count = std::min(100, static_cast<int>(d_remaining_indices_.size()));
            thrust::host_vector<GPUPoint3f> h_sample_points(sample_point_count);
            GPUPoint3f* points_ptr = getPointsPtr();
            cudaMemcpy(h_sample_points.data(),
                       &points_ptr[0],
                       sample_point_count * sizeof(GPUPoint3f),
                       cudaMemcpyDeviceToHost);
            
            for (int model_id = 0; model_id < debug_model_count; ++model_id)
            {
                float min_dist = 1e10f;
                float max_dist = 0.0f;
                float sum_dist = 0.0f;
                int valid_count = 0;
                
                for (int i = 0; i < sample_point_count; ++i)
                {
                    const GPUPoint3f& pt = h_sample_points[i];
                    float x = pt.x, y = pt.y, z = pt.z;
                    
                    // 计算距离（简化版本，使用 evaluateQuadricDistance 的逻辑）
                    float result = 0.0f;
                    float coords[4] = {x, y, z, 1.0f};
                    for (int row = 0; row < 4; ++row)
                    {
                        for (int col = 0; col < 4; ++col)
                        {
                            int idx = row * 4 + col;
                            if (idx >= 0 && idx < 16)
                            {
                                float coeff = h_models[model_id].coeffs[idx];
                                float term = coords[row] * coeff * coords[col];
                                if (std::isfinite(term) && !std::isnan(term) && !std::isinf(term))
                                {
                                    result += term;
                                }
                            }
                        }
                    }
                    
                    float dist = std::abs(result);
                    if (std::isfinite(dist) && !std::isnan(dist) && !std::isinf(dist))
                    {
                        min_dist = std::min(min_dist, dist);
                        max_dist = std::max(max_dist, dist);
                        sum_dist += dist;
                        valid_count++;
                    }
                }
                
                if (valid_count > 0)
                {
                    float avg_dist = sum_dist / valid_count;
                    std::cout << "  模型 " << model_id << " 距离统计 (基于 " << valid_count << " 个采样点):" << std::endl;
                    std::cout << "    最小距离: " << min_dist << " m" << std::endl;
                    std::cout << "    最大距离: " << max_dist << " m" << std::endl;
                    std::cout << "    平均距离: " << avg_dist << " m" << std::endl;
                    std::cout << "    阈值: " << params_.quadric_distance_threshold << " m" << std::endl;
                    std::cout << "    内点计数: " << (thrust::host_vector<int>(d_batch_inlier_counts_)[model_id]) << std::endl;
                }
            }
        }

        // Step 5: 从精选结果中找最优模型
        auto best_model_start = std::chrono::high_resolution_clock::now();
        // 从d_fine_inlier_counts_中找出最大值及其索引
        thrust::host_vector<int> h_fine_counts(fine_k);
        thrust::copy_n(d_fine_inlier_counts_.begin(), fine_k, h_fine_counts.begin());

        int best_fine_count = 0;
        int best_fine_idx = -1;
        for (int i = 0; i < fine_k; ++i)
        {
            if (h_fine_counts[i] > best_fine_count)
            {
                best_fine_count = h_fine_counts[i];
                best_fine_idx = i;
            }
        }

        // 获取最优模型在原始batch中的索引
        thrust::host_vector<int> h_top_k_indices(fine_k);
        thrust::copy_n(d_top_k_indices_.begin(), fine_k, h_top_k_indices.begin());
        int best_model_idx = (best_fine_idx >= 0) ? h_top_k_indices[best_fine_idx] : -1;
        int best_count = best_fine_count;

        auto best_model_end = std::chrono::high_resolution_clock::now();
        float best_model_time = std::chrono::duration<float, std::milli>(best_model_end - best_model_start).count();
        total_best_model_time += best_model_time;
        
        // 调试信息：验证最优模型结果
        if (params_.verbosity > 1)
        {
            std::cout << "[QuadricDetect] 最优模型选择结果:" << std::endl;
            std::cout << "  最优候选索引: " << best_fine_idx << std:: endl;
            std::cout << "  最优模型索引（原始batch）: " << best_model_idx << std::endl;
            std::cout << "  最优模型内点数: " << best_count << std::endl;
            
            // 验证索引有效性
            if (best_model_idx < 0 || best_model_idx >= batch_size)
            {
                std::cerr << "[QuadricDetect] 警告：最优模型索引无效！" << std::endl;
            }
        }

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

        // Step 6: 获取最优模型（从候选模型中获取）
        thrust::host_vector<GPUQuadricModel> h_candidate_models(fine_k);
        thrust::copy_n(d_candidate_models_.begin(), fine_k, h_candidate_models.begin());
        GPUQuadricModel best_gpu_model = h_candidate_models[best_fine_idx];

        // 添加：输出最优模型详情
        if (params_.verbosity > 0)
        {
            outputBestModelDetails(best_gpu_model, best_count, best_model_idx, iteration + 1);
            
            // 新增：最终检查日志
            std::cout << "[Final Check] Best model count: " << best_count 
                      << ", Index: " << best_model_idx << std::endl;
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
        
        // 🆕 从GPU缓冲区读取最优模型的显式系数和变换矩阵
        if (best_model_idx >= 0 && best_model_idx < batch_size) {
            thrust::host_vector<float> h_explicit_coeffs(6);
            thrust::host_vector<float> h_transform(12);
            thrust::copy_n(d_batch_explicit_coeffs_.begin() + best_model_idx * 6, 6, h_explicit_coeffs.begin());
            thrust::copy_n(d_batch_transforms_.begin() + best_model_idx * 12, 12, h_transform.begin());
            
            // 保存到DetectedPrimitive
            for (int i = 0; i < 6; ++i) {
                detected_quadric.explicit_coeffs[i] = h_explicit_coeffs[i];
            }
            for (int i = 0; i < 12; ++i) {
                detected_quadric.transform[i] = h_transform[i];
            }
            detected_quadric.has_visualization_data = true;
            
            if (params_.verbosity > 0) {
                ROS_INFO("[QuadricDetect] 已保存可视化数据: best_model_idx=%d, explicit_coeffs=[%.3f,%.3f,%.3f,%.3f,%.3f,%.3f]", 
                         best_model_idx, h_explicit_coeffs[0], h_explicit_coeffs[1], h_explicit_coeffs[2],
                         h_explicit_coeffs[3], h_explicit_coeffs[4], h_explicit_coeffs[5]);
            }
        } else {
            // 如果索引无效，标记为无可视化数据
            detected_quadric.has_visualization_data = false;
            ROS_WARN("[QuadricDetect] best_model_idx无效 (%d)，无法保存可视化数据", best_model_idx);
        }
        
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
            float iteration_total = sampling_time + inlier_count_time + 
                                  best_model_time + extract_inliers_time + extract_cloud_time + remove_points_time;
            std::cout << "[QuadricDetect] 已保存第 " << detected_primitives_.size() << " 个二次曲面" << std::endl;
            std::cout << "[QuadricDetect] 迭代 " << iteration + 1 << " 时间: " << iteration_total << " ms" << std::endl;
            std::cout << "  - 采样和构建矩阵: " << sampling_time << " ms" << std::endl;
            std::cout << "  - 计算内点数: " << inlier_count_time << " ms" << std::endl;
            std::cout << "    - 粗筛阶段: " << coarse_time << " ms" << std::endl;
            std::cout << "    - Top-K选择: " << topk_time << " ms" << std::endl;
            std::cout << "    - 精选阶段: " << fine_time << " ms" << std::endl;
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
        std::cout << "  - 计算内点数: " << total_inlier_count_time << " ms" << std::endl;
        std::cout << "    - 粗筛阶段: " << total_coarse_time << " ms" << std::endl;
        std::cout << "    - Top-K选择: " << total_topk_time << " ms" << std::endl;
        std::cout << "    - 精选阶段: " << total_fine_time << " ms" << std::endl;
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

    // Step 1: 从6×10矩阵（填充为9×10）计算10×10的A^T*A矩阵
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
    
    size_t remaining_count = d_remaining_indices_.size();
    
    // 优化：在 GPU 内部使用 gather 聚集剩余点到连续缓冲区（在 .cu 文件中实现）
    gatherRemainingToCompact();
    
    // 检查 gather 操作错误
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "[getFinalCloud]  gather操作错误: " 
                  << cudaGetErrorString(err) << std::endl;
        return final_cloud;
    }

    // 单次拷贝整块数据：从 GPU 连续缓冲区到 CPU
    thrust::host_vector<GPUPoint3f> h_compact_points;
    try {
        h_compact_points = d_compact_inliers_;
    } catch (const thrust::system_error &e) {
        std::cerr << "[getFinalCloud]  Thrust拷贝失败: " << e.what() << std::endl;
        err = cudaGetLastError();
        std::cerr << "[getFinalCloud] CUDA错误: " << cudaGetErrorString(err) << std::endl;
        return final_cloud;
    }
    
    // 转换为 PCL 点云
    final_cloud->reserve(remaining_count);
    for (size_t i = 0; i < h_compact_points.size(); ++i)
    {
        const GPUPoint3f& gpu_pt = h_compact_points[i];
        pcl::PointXYZI pt;
        pt.x = gpu_pt.x;
        pt.y = gpu_pt.y;
        pt.z = gpu_pt.z;
        pt.intensity = gpu_pt.intensity;
        final_cloud->push_back(pt);
    }
    
    final_cloud->width = final_cloud->size();
    final_cloud->height = 1;
    final_cloud->is_dense = true;
    
    return final_cloud;
}

void QuadricDetect::setStream(cudaStream_t stream)
{
    // 如果已有 stream 且拥有所有权，先销毁旧 stream
    if (stream_ != nullptr && owns_stream_)
    {
        cudaStreamDestroy(stream_);
    }
    
    stream_ = stream;
    owns_stream_ = false;  // 外部管理流生命周期
    
    // 如果 cusolver_handle_ 已初始化，绑定流
    if (cusolver_handle_ != nullptr)
    {
        cusolverDnSetStream(cusolver_handle_, stream_);
    }
}

bool QuadricDetect::processCloudDirect(GPUPoint3f* d_points, size_t count)
{
    if (d_points == nullptr || count == 0)
    {
        std::cerr << "[processCloudDirect] 错误：输入参数无效" << std::endl;
        return false;
    }

    if (stream_ == nullptr)
    {
        std::cerr << "[processCloudDirect] 错误：CUDA流未初始化" << std::endl;
        return false;
    }

    auto total_start = std::chrono::high_resolution_clock::now();

    // 重置内部状态（防止上一帧数据污染）
    detected_primitives_.clear();
    d_batch_inlier_counts_.clear();
    d_batch_models_.clear();
    d_best_model_index_.clear();
    d_best_model_count_.clear();
    d_remaining_indices_.clear();

    // 零拷贝指针赋值
    d_external_points_ = d_points;
    is_external_memory_ = true;
    
    // 记录初始总点数（用于掩码缓冲区分配）
    original_total_count_ = count;

    // 初始化 d_remaining_indices_ 为 0..count-1（假设输入就是剩余点，已压实）
    initializeRemainingIndices(count);

    // 确保掩码缓冲区已分配并初始化为全1
    initializeGPUMemory(1024);  // batch_size=1024
    if (d_valid_mask_ != nullptr)
    {
        cudaMemsetAsync(d_valid_mask_, 1, count * sizeof(uint8_t), stream_);
    }

    // 执行二次曲面检测
    auto detect_start = std::chrono::high_resolution_clock::now();
    findQuadrics_BatchGPU();
    auto detect_end = std::chrono::high_resolution_clock::now();
    float detect_time = std::chrono::duration<float, std::milli>(detect_end - detect_start).count();

    auto total_end = std::chrono::high_resolution_clock::now();
    float total_time = std::chrono::duration<float, std::milli>(total_end - total_start).count();

    if (params_.verbosity > 0) {
        std::cout << "[processCloudDirect] Timing breakdown:" << std::endl;
        std::cout << "  Quadric detection: " << detect_time << " ms" << std::endl;
        std::cout << "  Total: " << total_time << " ms" << std::endl;
    }

    // 确保所有 GPU 操作完成
    cudaStreamSynchronize(stream_);

    // 注意：不重置外部内存标志，保持 is_external_memory_ 和 d_external_points_
    // 以便后续 getFinalCloud() 和 extractInlierCloud() 能正确识别零拷贝模式
    // 标志位由调用方在适当时候管理（如调用 getFinalCloud() 之后）

    return true;
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
    
    //  检查 CUDA 错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "[extractInlierCloud]  CUDA错误: " 
                  << cudaGetErrorString(err) << std::endl;
        return inlier_cloud;
    }

    // 优化：在 GPU 内部使用 gather 聚集内点到连续缓冲区（在 .cu 文件中实现）
    gatherInliersToCompact();
    
    // 检查 gather 操作错误
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "[extractInlierCloud]  gather操作错误: " 
                  << cudaGetErrorString(err) << std::endl;
        return inlier_cloud;
    }

    // 单次拷贝整块数据：从 GPU 连续缓冲区到 CPU
    thrust::host_vector<GPUPoint3f> h_compact_inliers;
    try {
        h_compact_inliers = d_compact_inliers_;
    } catch (const thrust::system_error &e) {
        std::cerr << "[extractInlierCloud]  Thrust拷贝失败: " << e.what() << std::endl;
        err = cudaGetLastError();
        std::cerr << "[extractInlierCloud] CUDA错误: " << cudaGetErrorString(err) << std::endl;
        return inlier_cloud;
    }

    // 转换为 PCL 点云
    inlier_cloud->reserve(current_inlier_count_);
    for (size_t i = 0; i < h_compact_inliers.size(); ++i)
    {
        const GPUPoint3f& gpu_pt = h_compact_inliers[i];
        pcl::PointXYZI pt;
        pt.x = gpu_pt.x;
        pt.y = gpu_pt.y;
        pt.z = gpu_pt.z;
        pt.intensity = gpu_pt.intensity;
        inlier_cloud->push_back(pt);
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

//  新增函数：输出最优模型详情
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

// ========================================
// 可视化函数实现
// ========================================

// 辅助函数：将全局点变换到局部坐标系
static GPUPoint3f transformToLocal(const pcl::PointXYZI &pt_global, const float transform[12])
{
    // 对应 .cu 中的 T[i*4 + j] 存储方式：
    // transform[0-2] = R的第0行前3列 [X[0], Y[0], Z[0]]
    // transform[3] = p.x
    // transform[4-6] = R的第1行前3列 [X[1], Y[1], Z[1]]
    // transform[7] = p.y
    // transform[8-10] = R的第2行前3列 [X[2], Y[2], Z[2]]
    // transform[11] = p.z
    float R[9] = {transform[0], transform[1], transform[2],  // Row 0
                  transform[4], transform[5], transform[6],  // Row 1
                  transform[8], transform[9], transform[10]}; // Row 2
    float p[3] = {transform[3], transform[7], transform[11]}; // Translation
    
    // P - p
    float dx = pt_global.x - p[0];
    float dy = pt_global.y - p[1];
    float dz = pt_global.z - p[2];
    
    // R^T * (P - p)
    GPUPoint3f pt_local;
    pt_local.x = R[0]*dx + R[3]*dy + R[6]*dz;  // R^T的第一行
    pt_local.y = R[1]*dx + R[4]*dy + R[7]*dz;  // R^T的第二行
    pt_local.z = R[2]*dx + R[5]*dy + R[8]*dz;  // R^T的第三行
    
    return pt_local;
}

// 辅助函数：将局部点变换到全局坐标系
static GPUPoint3f transformToGlobal(const GPUPoint3f &pt_local, const float transform[12])
{
    // 对应 .cu 中的 T[i*4 + j] 存储方式：
    // transform[0-2] = R的第0行前3列 [X[0], Y[0], Z[0]]
    // transform[3] = p.x
    // transform[4-6] = R的第1行前3列 [X[1], Y[1], Z[1]]
    // transform[7] = p.y
    // transform[8-10] = R的第2行前3列 [X[2], Y[2], Z[2]]
    // transform[11] = p.z
    float R[9] = {transform[0], transform[1], transform[2],  // Row 0
                  transform[4], transform[5], transform[6],  // Row 1
                  transform[8], transform[9], transform[10]}; // Row 2
    float p[3] = {transform[3], transform[7], transform[11]}; // Translation
    
    // R * P_local + p
    GPUPoint3f pt_global;
    pt_global.x = R[0]*pt_local.x + R[1]*pt_local.y + R[2]*pt_local.z + p[0];
    pt_global.y = R[3]*pt_local.x + R[4]*pt_local.y + R[5]*pt_local.z + p[1];
    pt_global.z = R[6]*pt_local.x + R[7]*pt_local.y + R[8]*pt_local.z + p[2];
    
    return pt_global;
}

// Graham Scan凸包算法
struct Point2D {
    float x, y;
    int idx;
};

static std::vector<Point2D> grahamScan(std::vector<Point2D> &points)
{
    if (points.size() < 3) return points;
    
    // 1. 找最下方的点（y最小，相同则x最小）
    int bottom_idx = 0;
    for (size_t i = 1; i < points.size(); ++i) {
        if (points[i].y < points[bottom_idx].y ||
            (points[i].y == points[bottom_idx].y && points[i].x < points[bottom_idx].x)) {
            bottom_idx = i;
        }
    }
    std::swap(points[0], points[bottom_idx]);
    Point2D pivot = points[0];
    
    // 2. 按极角排序（相对于pivot）
    std::sort(points.begin() + 1, points.end(), [&pivot](const Point2D &a, const Point2D &b) {
        float cross = (a.x - pivot.x) * (b.y - pivot.y) - (a.y - pivot.y) * (b.x - pivot.x);
        if (fabsf(cross) < 1e-6f) {
            float dist_a = (a.x - pivot.x) * (a.x - pivot.x) + (a.y - pivot.y) * (a.y - pivot.y);
            float dist_b = (b.x - pivot.x) * (b.x - pivot.x) + (b.y - pivot.y) * (b.y - pivot.y);
            return dist_a < dist_b;
        }
        return cross > 0;
    });
    
    // 3. 构建凸包栈
    std::vector<Point2D> hull;
    hull.push_back(points[0]);
    if (points.size() > 1) hull.push_back(points[1]);
    
    for (size_t i = 2; i < points.size(); ++i) {
        while (hull.size() > 1) {
            Point2D &p1 = hull[hull.size() - 2];
            Point2D &p2 = hull[hull.size() - 1];
            Point2D &p3 = points[i];
            float cross = (p2.x - p1.x) * (p3.y - p1.y) - (p2.y - p1.y) * (p3.x - p1.x);
            if (cross > 0) break;
            hull.pop_back();
        }
        hull.push_back(points[i]);
    }
    
    return hull;
}

// 射线法判断点是否在凸包内
static bool isPointInConvexHull(const Point2D &pt, const std::vector<Point2D> &hull)
{
    if (hull.size() < 3) return false;
    
    // 从点向右发射射线，计算与凸包边界的交点数量
    int intersections = 0;
    for (size_t i = 0; i < hull.size(); ++i) {
        size_t j = (i + 1) % hull.size();
        Point2D &p1 = const_cast<Point2D&>(hull[i]);
        Point2D &p2 = const_cast<Point2D&>(hull[j]);
        
        // 检查射线是否与边相交
        if ((p1.y > pt.y) != (p2.y > pt.y)) {
            // 避免除零错误
            float dy = p2.y - p1.y;
            if (fabsf(dy) > 1e-6f) {
                float x_intersect = (pt.y - p1.y) * (p2.x - p1.x) / dy + p1.x;
                if (x_intersect > pt.x) {
                    intersections++;
                }
            }
        }
    }
    
    return (intersections % 2) == 1;
}

void QuadricDetect::computeVisualizationMarkers(
    const quadric::DetectedPrimitive &primitive,
    visualization_msgs::MarkerArray &marker_array,
    const std_msgs::Header &header,
    float grid_step,
    float alpha,
    bool clip_to_hull) const
{
    if (!primitive.has_visualization_data) {
        ROS_WARN("[computeVisualizationMarkers] primitive.has_visualization_data = false，跳过可视化");
        return;
    }
    
    if (primitive.inliers->empty()) {
        ROS_WARN("[computeVisualizationMarkers] primitive.inliers为空，跳过可视化");
        return;
    }
    
    ROS_INFO("[computeVisualizationMarkers] 开始处理，内点数: %zu", primitive.inliers->size());
    
    // ========================================
    // 1. 3σ离群点剔除
    // ========================================
    std::vector<GPUPoint3f> local_points;
    std::vector<float> distances;
    
    for (const auto &pt : primitive.inliers->points) {
        GPUPoint3f pt_local = transformToLocal(pt, primitive.transform);
        float dist = sqrtf(pt_local.x * pt_local.x + pt_local.y * pt_local.y);
        distances.push_back(dist);
        local_points.push_back(pt_local);
    }
    
    // 计算均值和标准差
    float mean = 0.0f;
    for (float d : distances) {
        mean += d;
    }
    mean /= distances.size();
    
    float variance = 0.0f;
    for (float d : distances) {
        variance += (d - mean) * (d - mean);
    }
    float std_dev = sqrtf(variance / distances.size());
    
    // 过滤：d < μ + 3σ
    float threshold = mean + 3.0f * std_dev;
    std::vector<GPUPoint3f> filtered_local_points;
    for (size_t i = 0; i < local_points.size(); ++i) {
        if (distances[i] < threshold) {
            filtered_local_points.push_back(local_points[i]);
        }
    }
    
    ROS_INFO("[computeVisualizationMarkers] 3σ过滤: 原始内点数=%zu, 过滤后=%zu, mean=%.3f, std_dev=%.3f, threshold=%.3f", 
             local_points.size(), filtered_local_points.size(), mean, std_dev, threshold);
    
    if (filtered_local_points.size() < 3) {
        ROS_WARN("[computeVisualizationMarkers] 过滤后点数太少 (%zu < 3)，无法生成凸包", filtered_local_points.size());
        return; // 点太少，无法生成凸包
    }
    
    // ========================================
    // 2. Graham Scan凸包生成
    // ========================================
    std::vector<Point2D> points_2d;
    for (size_t i = 0; i < filtered_local_points.size(); ++i) {
        points_2d.push_back({filtered_local_points[i].x, filtered_local_points[i].y, static_cast<int>(i)});
    }
    
    std::vector<Point2D> hull_2d = grahamScan(points_2d);
    
    ROS_INFO("[computeVisualizationMarkers] 凸包生成: 输入点数=%zu, 凸包点数=%zu", points_2d.size(), hull_2d.size());
    
    if (hull_2d.size() < 3) {
        ROS_WARN("[computeVisualizationMarkers] 凸包点数太少 (%zu < 3)，无法继续", hull_2d.size());
        return;
    }
    
    // 保存凸包点到primitive（注意：这里需要修改primitive，但函数是const，所以暂时跳过）
    // primitive.hull_points_local.clear();
    // for (const auto &pt : hull_2d) {
    //     primitive.hull_points_local.push_back({pt.x, pt.y, 0.0f});
    // }
    
    // ========================================
    // 3. 计算凸包的XY Bounding Box
    // ========================================
    float min_x = hull_2d[0].x, max_x = hull_2d[0].x;
    float min_y = hull_2d[0].y, max_y = hull_2d[0].y;
    for (const auto &pt : hull_2d) {
        min_x = std::min(min_x, pt.x);
        max_x = std::max(max_x, pt.x);
        min_y = std::min(min_y, pt.y);
        max_y = std::max(max_y, pt.y);
    }
    
    float bbox_dx = max_x - min_x;
    float bbox_dy = max_y - min_y;
    ROS_INFO("[computeVisualizationMarkers] Bounding Box: min_x=%.3f, max_x=%.3f, min_y=%.3f, max_y=%.3f, 范围: dx=%.3f, dy=%.3f",
             min_x, max_x, min_y, max_y, bbox_dx, bbox_dy);
    
    // ========================================
    // 4. 生成网格点并判断是否在凸包内
    // ========================================
    std::vector<geometry_msgs::Point> triangle_vertices;
    std::vector<geometry_msgs::Point> triangle_normals;
    
    // 自动调整网格步长：如果步长太大（超过bounding box的10%），则缩小
    float adjusted_grid_step = grid_step;
    if (grid_step > bbox_dx * 0.1f || grid_step > bbox_dy * 0.1f) {
        adjusted_grid_step = std::min(bbox_dx, bbox_dy) * 0.05f; // 使用bounding box的5%作为步长
        ROS_WARN("[computeVisualizationMarkers] 网格步长太大 (%.4f)，自动调整为 %.4f", grid_step, adjusted_grid_step);
    }
    
    // 生成网格点
    std::vector<std::vector<int>> grid_indices; // 存储网格点的索引映射
    int grid_width = static_cast<int>((max_x - min_x) / adjusted_grid_step) + 1;
    int grid_height = static_cast<int>((max_y - min_y) / adjusted_grid_step) + 1;
    grid_indices.resize(grid_height, std::vector<int>(grid_width, -1));
    
    ROS_INFO("[computeVisualizationMarkers] 网格参数: 原始grid_step=%.4f, 调整后=%.4f, grid_width=%d, grid_height=%d, clip_to_hull=%d",
             grid_step, adjusted_grid_step, grid_width, grid_height, clip_to_hull);
    
    int vertex_count = 0;
    int points_in_hull = 0;
    int points_out_hull = 0;
    for (int i = 0; i < grid_height; ++i) {
        for (int j = 0; j < grid_width; ++j) {
            float x = min_x + j * adjusted_grid_step;
            float y = min_y + i * adjusted_grid_step;
            Point2D grid_pt = {x, y, 0};
            
            // 射线法判断是否在凸包内
            bool in_hull = !clip_to_hull || isPointInConvexHull(grid_pt, hull_2d);
            
            if (in_hull) {
                points_in_hull++;
                // 显式映射：z = ax² + bxy + cy² + dx + ey + f
                float z = primitive.explicit_coeffs[0] * x * x +
                          primitive.explicit_coeffs[1] * x * y +
                          primitive.explicit_coeffs[2] * y * y +
                          primitive.explicit_coeffs[3] * x +
                          primitive.explicit_coeffs[4] * y +
                          primitive.explicit_coeffs[5];
                
                // 计算法向量：n = [-(2ax+by+d), -(bx+2cy+e), 1]
                float nx = -(2.0f * primitive.explicit_coeffs[0] * x + 
                             primitive.explicit_coeffs[1] * y + 
                             primitive.explicit_coeffs[3]);
                float ny = -(primitive.explicit_coeffs[1] * x + 
                             2.0f * primitive.explicit_coeffs[2] * y + 
                             primitive.explicit_coeffs[4]);
                float nz = 1.0f;
                float norm = sqrtf(nx*nx + ny*ny + nz*nz);
                nx /= norm; ny /= norm; nz /= norm;
                
                // 变换回全局坐标系
                GPUPoint3f local_vertex = {x, y, z};
                GPUPoint3f global_vertex = transformToGlobal(local_vertex, primitive.transform);
                
                geometry_msgs::Point v;
                v.x = global_vertex.x;
                v.y = global_vertex.y;
                v.z = global_vertex.z;
                triangle_vertices.push_back(v);
                
                // 法向量也需要变换到全局坐标系
                // 对应 .cu 中的 T[i*4 + j] 存储方式
                float R[9] = {primitive.transform[0], primitive.transform[1], primitive.transform[2],  // Row 0
                             primitive.transform[4], primitive.transform[5], primitive.transform[6],  // Row 1
                             primitive.transform[8], primitive.transform[9], primitive.transform[10]}; // Row 2
                geometry_msgs::Point n;
                n.x = R[0]*nx + R[1]*ny + R[2]*nz;
                n.y = R[3]*nx + R[4]*ny + R[5]*nz;
                n.z = R[6]*nx + R[7]*ny + R[8]*nz;
                triangle_normals.push_back(n);
                
                grid_indices[i][j] = vertex_count++;
            } else {
                points_out_hull++;
            }
        }
    }
    
    ROS_INFO("[computeVisualizationMarkers] 网格点统计: 总网格点数=%d, 在凸包内=%d, 在凸包外=%d, 生成顶点数=%d",
             grid_width * grid_height, points_in_hull, points_out_hull, vertex_count);
    
    // ========================================
    // 5. 生成三角形（每个网格单元2个三角形）
    // ========================================
    visualization_msgs::Marker marker;
    marker.header = header;
    marker.ns = "quadric_surfaces";
    marker.id = static_cast<int>(marker_array.markers.size());
    marker.type = visualization_msgs::Marker::TRIANGLE_LIST;
    marker.action = visualization_msgs::Marker::ADD;
    marker.pose.orientation.w = 1.0;
    marker.scale.x = 1.0;
    marker.scale.y = 1.0;
    marker.scale.z = 1.0;
    marker.color.r = 0.0f;
    marker.color.g = 0.5f;
    marker.color.b = 1.0f;
    marker.color.a = alpha;
    
    // 生成三角形
    int triangles_generated = 0;
    for (int i = 0; i < grid_height - 1; ++i) {
        for (int j = 0; j < grid_width - 1; ++j) {
            int idx00 = grid_indices[i][j];
            int idx01 = grid_indices[i][j+1];
            int idx10 = grid_indices[i+1][j];
            int idx11 = grid_indices[i+1][j+1];
            
            // 检查四个顶点是否都在凸包内
            if (idx00 >= 0 && idx01 >= 0 && idx10 >= 0) {
                // 第一个三角形
                marker.points.push_back(triangle_vertices[idx00]);
                marker.points.push_back(triangle_vertices[idx01]);
                marker.points.push_back(triangle_vertices[idx10]);
                triangles_generated++;
            }
            
            if (idx01 >= 0 && idx10 >= 0 && idx11 >= 0) {
                // 第二个三角形
                marker.points.push_back(triangle_vertices[idx01]);
                marker.points.push_back(triangle_vertices[idx11]);
                marker.points.push_back(triangle_vertices[idx10]);
                triangles_generated++;
            }
        }
    }
    
    ROS_INFO("[computeVisualizationMarkers] 三角形生成: 生成了 %d 个三角形, marker包含 %zu 个顶点", 
             triangles_generated, marker.points.size());
    
    if (!marker.points.empty()) {
        marker_array.markers.push_back(marker);
        ROS_INFO("[computeVisualizationMarkers] ✓ 成功生成marker，包含 %zu 个顶点", marker.points.size());
    } else {
        ROS_WARN("[computeVisualizationMarkers] ✗ 生成的marker为空（没有顶点），可能原因：1) 网格步长太大 2) 没有网格点在凸包内 3) 三角形生成失败");
    }
}