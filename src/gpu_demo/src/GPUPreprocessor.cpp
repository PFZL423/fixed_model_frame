#include "gpu_demo/GPUPreprocessor.h"
#include <iostream>
#include <chrono>
#include <stdexcept>
#include <cmath>

// ========== 构造函数与析构函数 ==========
GPUPreprocessor::GPUPreprocessor()
{
    int device_count;
    cudaError_t error = cudaGetDeviceCount(&device_count);
    if (error != cudaSuccess || device_count == 0)
    {
        throw std::runtime_error("No CUDA-capable device found!");
    }

    error = cudaSetDevice(0);
    if (error != cudaSuccess)
    {
        throw std::runtime_error("Failed to set CUDA device!");
    }

    std::cout << "[GPUPreprocessor] Initialized with CUDA device 0" << std::endl;
}

GPUPreprocessor::~GPUPreprocessor()
{
    clearMemory();
}

// ========== 主要处理接口 ==========
ProcessingResult GPUPreprocessor::process(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cpu_cloud,
                                          const PreprocessConfig &config)
{
    auto start_time = std::chrono::high_resolution_clock::now();
    last_stats_ = PerformanceStats{};

    // Step 1: 上传PCL点云到GPU
    auto upload_start = std::chrono::high_resolution_clock::now();
    std::vector<GPUPoint3f> gpu_points = convertPCLToGPU(cpu_cloud); 
    auto upload_end = std::chrono::high_resolution_clock::now();
    last_stats_.upload_time_ms = std::chrono::duration<float, std::milli>(upload_end - upload_start).count();
    std::cout<<last_stats_.upload_time_ms<<"!!!!!!!!!!!!!"<<std::endl;
    cuda_uploadGPUPoints(gpu_points);
    auto upload_end2 = std::chrono::high_resolution_clock::now();
    last_stats_.upload_time_ms = std::chrono::duration<float, std::milli>(upload_end2 - upload_end).count();

    // Step 2: GPU预处理
    preprocessOnGPU(config);

    // Step 3: 创建结果对象
    ProcessingResult result = createResult(config);

    auto end_time = std::chrono::high_resolution_clock::now();
    last_stats_.total_time_ms = std::chrono::duration<float, std::milli>(end_time - start_time).count();

    std::cout << "[GPUPreprocessor] Processing completed:" << std::endl;
    std::cout << "  - Upload: " << last_stats_.upload_time_ms << "ms" << std::endl;
    std::cout << "  - Voxel Filter: " << last_stats_.voxel_filter_time_ms << "ms" << std::endl;
    std::cout << "  - Outlier Removal: " << last_stats_.outlier_removal_time_ms << "ms" << std::endl;
    std::cout << "  - Normal Estimation: " << last_stats_.normal_estimation_time_ms << "ms" << std::endl;
    std::cout << "  - Total: " << last_stats_.total_time_ms << "ms" << std::endl;

    return result;
}
void GPUPreprocessor::preprocessOnGPU(const PreprocessConfig &config)
{
    // 初始化工作点云
    d_temp_points_ = d_input_points_;

    // Step 1: 体素下采样
    if (config.enable_voxel_filter)
    {
        auto start = std::chrono::high_resolution_clock::now();
        cuda_launchVoxelFilter(config.voxel_size);
        auto end = std::chrono::high_resolution_clock::now();
        last_stats_.voxel_filter_time_ms = std::chrono::duration<float, std::milli>(end - start).count();
    }

    // Step 2: 离群点移除
    if (config.enable_outlier_removal)
    {
        auto start = std::chrono::high_resolution_clock::now();
        cuda_launchOutlierRemoval(config);
        auto end = std::chrono::high_resolution_clock::now();
        last_stats_.outlier_removal_time_ms = std::chrono::duration<float, std::milli>(end - start).count();
    }

    // Step 3: 地面移除 (可选)
    if (config.enable_ground_removal)
    {
        cuda_launchGroundRemoval(config.ground_threshold);
    }

    // // Step 4: 法线估计 (根据开关决定)
    // if (config.compute_normals)
    // {
    //     auto start = std::chrono::high_resolution_clock::now();
    //     launchNormalEstimation(config.normal_radius, config.normal_k);
    //     auto end = std::chrono::high_resolution_clock::now();
    //     last_stats_.normal_estimation_time_ms = std::chrono::duration<float, std::milli>(end - start).count();
    // }
}

ProcessingResult GPUPreprocessor::createResult(const PreprocessConfig &config)
{
    ProcessingResult result;
    size_t final_point_count = getCurrentPointCount();

    result.setPointsRef(&d_output_points_);
    result.setPointCount(final_point_count);

    // if (config.compute_normals)
    // {
    //     result.setPointsNormalRef(&d_output_points_normal_);
    // }

    return result;
}

size_t GPUPreprocessor::getCurrentPointCount() const
{
    return d_output_points_.size();
}

// ========== 算法包装函数 (调用.cu中的实现) ==========


// ========== 修改launchNormalEstimation函数 ==========
void GPUPreprocessor::launchNormalEstimation(float radius, int k)
{
    // std::cout << "[GPUPreprocessor] Starting normal estimation" << std::endl;

    // size_t point_count = d_temp_points_.size();
    // if (point_count == 0)
    //     return;

    // // ✅ 避开resize，用clear+reserve+手动构造
    // d_output_points_normal_.clear();
    // d_output_points_normal_.reserve(point_count);

    // // 创建临时的host_vector来构造数据
    // std::vector<GPUPointNormal3f> h_temp(point_count);
    // d_output_points_normal_ = h_temp; // 通过赋值避免resize

    // // 调用.cu文件中的CUDA实现
    // cuda_performNormalEstimation(
    //     thrust::raw_pointer_cast(d_temp_points_.data()),
    //     thrust::raw_pointer_cast(d_output_points_normal_.data()),
    //     point_count, radius, k);

    // std::cout << "[GPUPreprocessor] Normal estimation completed for " << point_count << " points" << std::endl;
}


// ========== 工具函数实现 ==========



// ========== ProcessingResult 实现 ==========
std::vector<GPUPoint3f> ProcessingResult::downloadPoints() const
{
    if (!d_points_)
        return {};

    thrust::host_vector<GPUPoint3f> host_points = *d_points_;
    return std::vector<GPUPoint3f>(host_points.begin(), host_points.end());
}

std::vector<GPUPointNormal3f> ProcessingResult::downloadPointsWithNormals() const
{
    // // if (!d_points_normal_)
    // //     return {};

    // // thrust::host_vector<GPUPointNormal3f> host_points = *d_points_normal_;
    // // return std::vector<GPUPointNormal3f>(host_points.begin(), host_points.end());
    // if (!has_normals_ || !d_points_normal_ || point_count_ == 0)
    // {
    //     return {};
    // }

    // // 使用 cudaMemcpy 替代 thrust::copy
    // std::vector<GPUPointNormal3f> result(point_count_);

    // cudaError_t error = cudaMemcpy(
    //     result.data(),                                      // 目标：CPU内存
    //     thrust::raw_pointer_cast(d_points_normal_->data()), // 源：GPU内存
    //     point_count_ * sizeof(GPUPointNormal3f),            // 大小
    //     cudaMemcpyDeviceToHost                              // 方向
    // );

    // if (error != cudaSuccess)
    // {
    //     throw std::runtime_error("CUDA memcpy failed: " + std::string(cudaGetErrorString(error)));
    // }

    // return result;
}

std::vector<GPUPoint3f> GPUPreprocessor::convertPCLToGPU(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cpu_cloud)
{
    std::vector<GPUPoint3f> h_points;
    h_points.reserve(cpu_cloud->size());

    for (const auto &pt : cpu_cloud->points)
    {
        // ✅ 添加有效性检查
        if (std::isfinite(pt.x) && std::isfinite(pt.y) && std::isfinite(pt.z))
        {
            GPUPoint3f gpu_pt;
            gpu_pt.x = pt.x;
            gpu_pt.y = pt.y;
            gpu_pt.z = pt.z;
            h_points.push_back(gpu_pt);
        }
    }

    std::cout << "[GPUPreprocessor] Converted " << h_points.size() << " valid points (filtered "
              << (cpu_cloud->size() - h_points.size()) << " invalid points)" << std::endl;

    return h_points; // ✅ 返回转换结果，不执行上传
}