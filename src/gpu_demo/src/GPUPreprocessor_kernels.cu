#include <cuda_runtime.h>

#include <ros/ros.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <thrust/unique.h>
#include <thrust/gather.h>
#include <thrust/functional.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <thrust/system/cuda/execution_policy.h>
#include <iostream>
#include <numeric>
#include <chrono>

#include "gpu_demo/GPUPreprocessor_kernels.cuh"
#include "gpu_demo/GPUPreprocessor.h"

// ========== GPU Kernel实现 (保持不变) ==========
namespace VoxelFilter
{
    __device__ inline uint64_t computeVoxelHash(float x, float y, float z, float voxel_size)
    {
        // ✅ 添加输入验证
        if (!isfinite(x) || !isfinite(y) || !isfinite(z) || voxel_size <= 0.0f)
        {
            return 0; // 返回安全的默认值
        }

        int vx = __float2int_rd(x / voxel_size);
        int vy = __float2int_rd(y / voxel_size);
        int vz = __float2int_rd(z / voxel_size);

        // ✅ 限制范围，避免溢出
        vx = max(-1048576, min(1048575, vx)); // ±2^20
        vy = max(-1048576, min(1048575, vy)); // ±2^20
        vz = max(-512, min(511, vz));         // ±2^9

        uint32_t ux = static_cast<uint32_t>(vx + (1 << 20));
        uint32_t uy = static_cast<uint32_t>(vy + (1 << 20));
        uint32_t uz = static_cast<uint32_t>(vz + (1 << 9));

        uint64_t hash = (static_cast<uint64_t>(ux) << 32) |
                        (static_cast<uint64_t>(uy) << 10) |
                        static_cast<uint64_t>(uz);
        return hash;
    }

    __global__ void computeVoxelKeysKernel(
        const GPUPoint3f *points,
        uint64_t *voxel_keys,
        float voxel_size,
        int point_count)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= point_count)
            return;

        const GPUPoint3f &point = points[idx];
        voxel_keys[idx] = computeVoxelHash(point.x, point.y, point.z, voxel_size);
    }
}

namespace OutlierRemoval
{
    __device__ inline float computeDistance(const GPUPoint3f &p1, const GPUPoint3f &p2);
    __device__ inline float computeDistance(const GPUPoint3f &p1, const GPUPoint3f &p2)
    {
        float dx = p1.x - p2.x;
        float dy = p1.y - p2.y;
        float dz = p1.z - p2.z;
        return sqrtf(dx * dx + dy * dy + dz * dz);
    }

    // 已弃用：O(N²)暴力实现，被空间哈希替代
    __global__ void radiusOutlierKernel(
        const GPUPoint3f *points,
        bool *valid_flags,
        int point_count,
        float radius,
        int min_neighbors)
    {
        // 空实现，不再使用
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= point_count)
            return;
        valid_flags[idx] = true; // 默认所有点有效
    }

    // 已弃用：统计离群点移除，未实现
    __global__ void statisticalOutlierKernel(
        const GPUPoint3f *points,
        bool *valid_flags,
        int point_count,
        int k,
        float std_dev_multiplier)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= point_count)
            return;
        valid_flags[idx] = true; // 默认所有点有效
    }
}

namespace GroundRemoval
{
    __global__ void ransacGroundDetectionKernel(
        const GPUPoint3f *points,
        bool *ground_flags,
        int point_count,
        float threshold,
        int max_iterations)
    {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= point_count)
            return;
        ground_flags[idx] = (points[idx].z < threshold);
    }
}

// ========== 在.cu文件末尾添加所有GPU内存管理函数 ==========

void GPUPreprocessor::cuda_initializeMemory(size_t max_points)
{
    // 在.cu文件中，所有resize都是安全的
    // 只调整大小，不初始化数据，等待后续填充
    if (d_voxel_keys_.size() < max_points)
    {
        d_voxel_keys_.resize(max_points);
    }
    if (d_valid_flags_.size() < max_points)
    {
        d_valid_flags_.resize(max_points);
    }
    if (d_neighbor_counts_.size() < max_points)
    {
        d_neighbor_counts_.resize(max_points);
    }
    if (d_knn_indices_.size() < max_points * 20)
    {
        d_knn_indices_.resize(max_points * 20);
    }
    if (d_knn_distances_.size() < max_points * 20)
    {
        d_knn_distances_.resize(max_points * 20);
    }
    if (d_voxel_boundaries_.size() < max_points)
    {
        d_voxel_boundaries_.resize(max_points);
    }
    if (d_unique_keys_.size() < max_points)
    {
        d_unique_keys_.resize(max_points);
    }

    // POD结构体只需要reserve即可，大小会在使用时正确设置
    d_temp_points_.reserve(max_points);
    d_output_points_.reserve(max_points);

    // 预分配桶排序缓冲区
    const int NUM_BUCKETS = 1024;  // 桶数量（可配置，1024足够处理大部分情况）
    if (d_bucket_indices_.size() < max_points)
    {
        d_bucket_indices_.resize(max_points);
    }
    if (d_temp_points_sort_.size() < max_points)
    {
        d_temp_points_sort_.resize(max_points);
    }
    if (d_temp_keys_sort_.size() < max_points)
    {
        d_temp_keys_sort_.resize(max_points);
    }
    if (d_bucket_counts_.size() < NUM_BUCKETS)
    {
        d_bucket_counts_.resize(NUM_BUCKETS);
    }
    if (d_bucket_offsets_.size() < NUM_BUCKETS)
    {
        d_bucket_offsets_.resize(NUM_BUCKETS);
    }
    if (d_bucket_positions_.size() < NUM_BUCKETS)
    {
        d_bucket_positions_.resize(NUM_BUCKETS);
    }
    if (d_min_max_keys_.size() < 2)
    {
        d_min_max_keys_.resize(2);
    }
}

void GPUPreprocessor::copyTempPointsFromInput()
{
    d_temp_points_ = d_input_points_;
}

void GPUPreprocessor::cuda_launchVoxelFilter(float voxel_size)
{
    auto total_start = std::chrono::high_resolution_clock::now();
    std::cout << "[GPUPreprocessor] Starting voxel filter with size " << voxel_size << std::endl;

    size_t input_count = d_temp_points_.size();
    if (input_count == 0)
        return;

    // Step 1: 准备内存
    auto memory_start = std::chrono::high_resolution_clock::now();
    d_voxel_keys_.clear();
    d_voxel_keys_.resize(input_count);
    auto memory_end = std::chrono::high_resolution_clock::now();
    float memory_time = std::chrono::duration<float, std::milli>(memory_end - memory_start).count();

    // Step 2: 计算体素keys
    auto kernel_start = std::chrono::high_resolution_clock::now();
    dim3 block(256);
    dim3 grid((input_count + block.x - 1) / block.x);

    VoxelFilter::computeVoxelKeysKernel<<<grid, block, 0, stream_>>>(
        thrust::raw_pointer_cast(d_temp_points_.data()),
        thrust::raw_pointer_cast(d_voxel_keys_.data()),
        voxel_size,
        static_cast<int>(input_count));

    cudaError_t kernel_error = cudaGetLastError();
    if (kernel_error != cudaSuccess)
    {
        std::cerr << "[ERROR] Voxel kernel failed: " << cudaGetErrorString(kernel_error) << std::endl;
        return;
    }
    cudaStreamSynchronize(stream_);
    auto kernel_end = std::chrono::high_resolution_clock::now();
    float kernel_time = std::chrono::duration<float, std::milli>(kernel_end - kernel_start).count();

    // Step 3: 大小检查
    auto check_start = std::chrono::high_resolution_clock::now();
    if (d_voxel_keys_.size() != input_count)
    {
        std::cerr << "[ERROR] Voxel keys size mismatch: " << d_voxel_keys_.size()
                  << " vs " << input_count << std::endl;
        d_voxel_keys_.resize(input_count);
    }

    if (d_temp_points_.size() != input_count)
    {
        std::cerr << "[ERROR] Temp points size mismatch: " << d_temp_points_.size()
                  << " vs " << input_count << std::endl;
        return;
    }
    auto check_end = std::chrono::high_resolution_clock::now();
    float check_time = std::chrono::duration<float, std::milli>(check_end - check_start).count();

    // ========== Step 4: GPU Bucket Sort (全GPU流程) ==========
    auto sort_start = std::chrono::high_resolution_clock::now();

    const int NUM_BUCKETS = 1024;  // 桶数量

    // 确保临时缓冲区大小足够
    if (d_bucket_indices_.size() < input_count)
    {
        d_bucket_indices_.resize(input_count);
    }
    if (d_temp_points_sort_.size() < input_count)
    {
        d_temp_points_sort_.resize(input_count);
    }
    if (d_temp_keys_sort_.size() < input_count)
    {
        d_temp_keys_sort_.resize(input_count);
    }
    if (d_bucket_counts_.size() < NUM_BUCKETS)
    {
        d_bucket_counts_.resize(NUM_BUCKETS);
    }
    if (d_bucket_offsets_.size() < NUM_BUCKETS)
    {
        d_bucket_offsets_.resize(NUM_BUCKETS);
    }
    if (d_bucket_positions_.size() < NUM_BUCKETS)
    {
        d_bucket_positions_.resize(NUM_BUCKETS);
    }
    if (d_min_max_keys_.size() < 2)
    {
        d_min_max_keys_.resize(2);
    }

    // Step A: 范围分析 - 计算 min/max key
    // block 已在前面声明，直接使用
    dim3 grid_range((input_count + block.x - 1) / block.x);

    // 初始化 min/max
    uint64_t init_min = UINT64_MAX;
    uint64_t init_max = 0;
    cudaMemcpyAsync(thrust::raw_pointer_cast(d_min_max_keys_.data()), &init_min, 
                    sizeof(uint64_t), cudaMemcpyHostToDevice, stream_);
    cudaMemcpyAsync(thrust::raw_pointer_cast(d_min_max_keys_.data()) + 1, &init_max, 
                    sizeof(uint64_t), cudaMemcpyHostToDevice, stream_);

    GPUBucketSort::analyzeKeyRangeKernel<<<grid_range, block, 0, stream_>>>(
        thrust::raw_pointer_cast(d_voxel_keys_.data()),
        static_cast<int>(input_count),
        thrust::raw_pointer_cast(d_min_max_keys_.data()),
        thrust::raw_pointer_cast(d_min_max_keys_.data()) + 1);

    // 下载 min/max (需要同步，因为后续步骤依赖)
    cudaStreamSynchronize(stream_);
    uint64_t min_key, max_key;
    cudaMemcpy(&min_key, thrust::raw_pointer_cast(d_min_max_keys_.data()), 
               sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&max_key, thrust::raw_pointer_cast(d_min_max_keys_.data()) + 1, 
               sizeof(uint64_t), cudaMemcpyDeviceToHost);

    uint64_t key_range = (max_key > min_key) ? (max_key - min_key) : 1;

    // Step B: 计算桶索引
    dim3 grid_bucket((input_count + block.x - 1) / block.x);
    GPUBucketSort::computeBucketIndicesKernel<<<grid_bucket, block, 0, stream_>>>(
        thrust::raw_pointer_cast(d_voxel_keys_.data()),
        thrust::raw_pointer_cast(d_bucket_indices_.data()),
        static_cast<int>(input_count),
        min_key,
        key_range,
        NUM_BUCKETS);

    // Step C: 统计桶大小
    thrust::fill(thrust::cuda::par.on(stream_), 
                 d_bucket_counts_.begin(), d_bucket_counts_.begin() + NUM_BUCKETS, 0);

    dim3 grid_count((input_count + block.x - 1) / block.x);
    GPUBucketSort::countBucketSizesKernel<<<grid_count, block, 0, stream_>>>(
        thrust::raw_pointer_cast(d_bucket_indices_.data()),
        thrust::raw_pointer_cast(d_bucket_counts_.data()),
        static_cast<int>(input_count),
        NUM_BUCKETS);

    // Step D: 前缀和扫描计算桶偏移量
    thrust::exclusive_scan(thrust::cuda::par.on(stream_),
                           d_bucket_counts_.begin(), 
                           d_bucket_counts_.begin() + NUM_BUCKETS,
                           d_bucket_offsets_.begin(),
                           0);

    // Step E: 全局重排（关键优化点，消除224ms CPU重排）
    thrust::fill(thrust::cuda::par.on(stream_),
                 d_bucket_positions_.begin(), d_bucket_positions_.begin() + NUM_BUCKETS, 0);

    dim3 grid_distribute((input_count + block.x - 1) / block.x);
    GPUBucketSort::distributeToBucketsKernel<<<grid_distribute, block, 0, stream_>>>(
        thrust::raw_pointer_cast(d_temp_points_.data()),
        thrust::raw_pointer_cast(d_voxel_keys_.data()),
        thrust::raw_pointer_cast(d_bucket_indices_.data()),
        thrust::raw_pointer_cast(d_bucket_offsets_.data()),
        thrust::raw_pointer_cast(d_temp_points_sort_.data()),
        thrust::raw_pointer_cast(d_temp_keys_sort_.data()),
        thrust::raw_pointer_cast(d_bucket_positions_.data()),
        static_cast<int>(input_count));

    // 交换缓冲区（使用swap避免拷贝）
    d_temp_points_.swap(d_temp_points_sort_);
    d_voxel_keys_.swap(d_temp_keys_sort_);

    // Step F: 桶内精排（使用基数排序）
    dim3 grid_radix(NUM_BUCKETS, 1);  // 每个桶一个block
    dim3 block_radix(32);  // 每个block 32个线程（一个warp）

    GPUBucketSort::radixSortWithinBucketsKernel<<<grid_radix, block_radix, 0, stream_>>>(
        thrust::raw_pointer_cast(d_temp_points_.data()),
        thrust::raw_pointer_cast(d_voxel_keys_.data()),
        thrust::raw_pointer_cast(d_temp_points_sort_.data()),
        thrust::raw_pointer_cast(d_temp_keys_sort_.data()),
        thrust::raw_pointer_cast(d_bucket_offsets_.data()),
        thrust::raw_pointer_cast(d_bucket_counts_.data()),
        NUM_BUCKETS);

    // 最终交换回排序后的数据
    d_temp_points_.swap(d_temp_points_sort_);
    d_voxel_keys_.swap(d_temp_keys_sort_);

    // 确保所有异步操作完成
    cudaStreamSynchronize(stream_);

    // 错误检查
    cudaError_t sort_error = cudaGetLastError();
    if (sort_error != cudaSuccess)
    {
        std::cerr << "[ERROR] GPU Bucket Sort failed: " << cudaGetErrorString(sort_error) << std::endl;
        return;
    }

    auto sort_end = std::chrono::high_resolution_clock::now();
    float sort_time = std::chrono::duration<float, std::milli>(sort_end - sort_start).count();

    // Step 5: 后续处理
    auto process_start = std::chrono::high_resolution_clock::now();
    // 🔧 修复：使用实际点云大小而不是原始输入大小
    size_t actual_count = d_temp_points_.size();
    processVoxelCentroids(actual_count);
    auto process_end = std::chrono::high_resolution_clock::now();
    float process_time = std::chrono::duration<float, std::milli>(process_end - process_start).count();

    auto total_end = std::chrono::high_resolution_clock::now();
    float total_time = std::chrono::duration<float, std::milli>(total_end - total_start).count();

    std::cout << "[VoxelFilter] Timing breakdown:" << std::endl;
    std::cout << "  Memory setup: " << memory_time << " ms" << std::endl;
    std::cout << "  Kernel compute: " << kernel_time << " ms" << std::endl;
    std::cout << "  Size check: " << check_time << " ms" << std::endl;
    std::cout << "  GPU Bucket Sort: " << sort_time << " ms" << std::endl;
    std::cout << "  Process centroids: " << process_time << " ms" << std::endl;
    std::cout << "  Total: " << total_time << " ms" << std::endl;
}

//  基数排序实现 - 专门优化64位整数keys
void radixSort(std::vector<size_t> &indices, const std::vector<uint64_t> &keys)
{
    const size_t n = indices.size();
    if (n <= 1)
        return;

    std::vector<size_t> temp_indices(n);
    const int RADIX_BITS = 8;                                  // 每次处理8位
    const int RADIX_SIZE = 1 << RADIX_BITS;                    // 256
    const int NUM_PASSES = (64 + RADIX_BITS - 1) / RADIX_BITS; // 8次遍历

    for (int pass = 0; pass < NUM_PASSES; ++pass)
    {
        // 计数数组
        std::vector<int> count(RADIX_SIZE, 0);
        int shift = pass * RADIX_BITS;

        // 统计每个桶的元素数量
        for (size_t i = 0; i < n; ++i)
        {
            int digit = (keys[indices[i]] >> shift) & (RADIX_SIZE - 1);
            count[digit]++;
        }

        // 转换为累积计数
        for (int i = 1; i < RADIX_SIZE; ++i)
        {
            count[i] += count[i - 1];
        }

        // 从后往前分配到临时数组
        for (int i = static_cast<int>(n) - 1; i >= 0; --i)
        {
            int digit = (keys[indices[i]] >> shift) & (RADIX_SIZE - 1);
            temp_indices[--count[digit]] = indices[i];
        }

        // 复制回原数组
        indices = temp_indices;
    }
}

bool GPUPreprocessor::cpuFallbackSort(size_t input_count)
{
    auto cpu_total_start = std::chrono::high_resolution_clock::now();
    std::cout << "[INFO] Using CPU radix sort fallback..." << std::endl;

    try
    {
        // Step 1: 下载数据到CPU
        auto download_start = std::chrono::high_resolution_clock::now();
        thrust::host_vector<GPUPoint3f> h_points = d_temp_points_;
        thrust::host_vector<uint64_t> h_keys = d_voxel_keys_;
        auto download_end = std::chrono::high_resolution_clock::now();
        float download_time = std::chrono::duration<float, std::milli>(download_end - download_start).count();

        // Step 2: 创建索引
        auto index_start = std::chrono::high_resolution_clock::now();
        std::vector<size_t> indices(input_count);
        std::iota(indices.begin(), indices.end(), 0);
        auto index_end = std::chrono::high_resolution_clock::now();
        float index_time = std::chrono::duration<float, std::milli>(index_end - index_start).count();

        // 调试：检查原始keys
        std::vector<uint64_t> std_keys(h_keys.begin(), h_keys.end());
        std::cout << "[DEBUG] First 10 voxel keys: ";
        for (size_t i = 0; i < std::min(size_t(10), input_count); ++i)
        {
            std::cout << std_keys[i] << " ";
        }
        std::cout << std::endl;

        // 检查是否所有keys都相同
        uint64_t first_key = std_keys[0];
        bool all_same = true;
        for (size_t i = 1; i < input_count; ++i)
        {
            if (std_keys[i] != first_key)
            {
                all_same = false;
                break;
            }
        }
        std::cout << "[DEBUG] All keys same? " << (all_same ? "YES" : "NO") << std::endl;

        // Step 3: CPU基数排序 (专门优化64位keys)
        auto sort_start = std::chrono::high_resolution_clock::now();

        if (all_same)
        {
            std::cout << "[WARNING] All voxel keys are identical - skipping sort" << std::endl;
        }
        else
        {
            radixSort(indices, std_keys);
        }

        auto sort_end = std::chrono::high_resolution_clock::now();
        float sort_time = std::chrono::duration<float, std::milli>(sort_end - sort_start).count();

        //  调试：检查排序后的前几个索引
        std::cout << "[DEBUG] First 10 sorted indices: ";
        for (size_t i = 0; i < std::min(size_t(10), input_count); ++i)
        {
            std::cout << indices[i] << " ";
        }
        std::cout << std::endl; // Step 4: 重新排列数据
        auto rearrange_start = std::chrono::high_resolution_clock::now();
        thrust::host_vector<GPUPoint3f> sorted_points(input_count);
        thrust::host_vector<uint64_t> sorted_keys(input_count);

        for (size_t i = 0; i < input_count; ++i)
        {
            sorted_points[i] = h_points[indices[i]];
            sorted_keys[i] = h_keys[indices[i]];
        }
        auto rearrange_end = std::chrono::high_resolution_clock::now();
        float rearrange_time = std::chrono::duration<float, std::milli>(rearrange_end - rearrange_start).count();

        // Step 5: 上传回GPU
        auto upload_start = std::chrono::high_resolution_clock::now();
        d_temp_points_ = sorted_points;
        d_voxel_keys_ = sorted_keys;
        auto upload_end = std::chrono::high_resolution_clock::now();
        float upload_time = std::chrono::duration<float, std::milli>(upload_end - upload_start).count();

        auto cpu_total_end = std::chrono::high_resolution_clock::now();
        float cpu_total_time = std::chrono::duration<float, std::milli>(cpu_total_end - cpu_total_start).count();

        std::cout << "[CPUSort] Detailed timing breakdown (Radix Sort):" << std::endl;
        std::cout << "  GPU->CPU download: " << download_time << " ms" << std::endl;
        std::cout << "  Index creation: " << index_time << " ms" << std::endl;
        std::cout << "  CPU radix sort: " << sort_time << " ms" << std::endl;
        std::cout << "  Data rearrange: " << rearrange_time << " ms" << std::endl;
        std::cout << "  CPU->GPU upload: " << upload_time << " ms" << std::endl;
        std::cout << "  CPU total: " << cpu_total_time << " ms" << std::endl;

        return true;
    }
    catch (const std::exception &e)
    {
        std::cerr << "[ERROR] CPU fallback sort failed: " << e.what() << std::endl;
        return false;
    }
}

// 将后续处理拆分为独立函数
void GPUPreprocessor::processVoxelCentroids(size_t input_count)
{
    // 确保输入数据一致性
    if (d_temp_points_.size() != input_count || d_voxel_keys_.size() != input_count)
    {
        std::cerr << "[ERROR] Size mismatch in processVoxelCentroids!" << std::endl;
        return;
    }

    // Step 1: 统计每个体素的原始点数 + 累加坐标
    thrust::device_vector<int> d_point_counts(input_count);
    thrust::device_vector<int> d_ones(input_count, 1);

    d_unique_keys_.resize(input_count);
    thrust::device_vector<GPUPoint3f> d_temp_centroids(input_count);

    // reduce_by_key：统计点数
    auto count_end = thrust::reduce_by_key(
        d_voxel_keys_.begin(), d_voxel_keys_.begin() + input_count,
        d_ones.begin(),
        d_unique_keys_.begin(),
        d_point_counts.begin());

    // reduce_by_key：累加坐标
    thrust::reduce_by_key(
        d_voxel_keys_.begin(), d_voxel_keys_.begin() + input_count,
        d_temp_points_.begin(),
        d_unique_keys_.begin(),
        d_temp_centroids.begin(),
        thrust::equal_to<uint64_t>(),
        [] __device__(const GPUPoint3f &a, const GPUPoint3f &b)
        {
            return GPUPoint3f{a.x + b.x, a.y + b.y, a.z + b.z, a.intensity + b.intensity};
        });

    size_t unique_count = count_end.second - d_point_counts.begin();

    if (unique_count == 0)
    {
        std::cerr << "[WARNING] No unique voxels found!" << std::endl;
        d_output_points_.clear();
        d_temp_points_.clear();
        return;
    }

    // Step 2: 计算质心（坐标 / 点数）
    thrust::transform(
        d_temp_centroids.begin(), d_temp_centroids.begin() + unique_count,
        d_point_counts.begin(),
        d_temp_centroids.begin(),
        [] __device__(const GPUPoint3f &sum_point, int count)
        {
            float inv = 1.0f / count;
            return GPUPoint3f{sum_point.x * inv, sum_point.y * inv,
                              sum_point.z * inv, sum_point.intensity * inv};
        });

    // Step 3: 体素密度过滤（voxel_min_points > 0 时启用）
    // d_voxel_min_points_ 通过 last_voxel_min_points_ 成员传入
    size_t output_count = unique_count;
    if (last_voxel_min_points_ > 0)
    {
        // 用 copy_if 保留 count >= threshold 的质心（保持 key 有序）
        thrust::device_vector<GPUPoint3f> d_filtered(unique_count);
        // 同时过滤出对应的序号（0..unique_count-1）
        thrust::device_vector<int> d_seq(unique_count);
        thrust::sequence(d_seq.begin(), d_seq.end(), 0);
        thrust::device_vector<int> d_filtered_ids(unique_count);

        auto end_pts = thrust::copy_if(
            d_temp_centroids.begin(), d_temp_centroids.begin() + unique_count,
            d_point_counts.begin(),
            d_filtered.begin(),
            [min_pts = last_voxel_min_points_] __device__(int cnt) {
                return cnt >= min_pts;
            });
        auto end_ids = thrust::copy_if(
            d_seq.begin(), d_seq.begin() + unique_count,
            d_point_counts.begin(),
            d_filtered_ids.begin(),
            [min_pts = last_voxel_min_points_] __device__(int cnt) {
                return cnt >= min_pts;
            });
        output_count = end_pts - d_filtered.begin();

        thrust::host_vector<GPUPoint3f> h_result(output_count);
        thrust::host_vector<int> h_ids(output_count);
        if (output_count > 0)
        {
            thrust::copy_n(d_filtered.begin(), output_count, h_result.begin());
            thrust::copy_n(d_filtered_ids.begin(), output_count, h_ids.begin());
        }
        d_output_points_ = h_result;
        h_output_voxel_ids_.assign(h_ids.begin(), h_ids.end());
    }
    else
    {
        thrust::host_vector<GPUPoint3f> h_result(unique_count);
        thrust::copy_n(d_temp_centroids.begin(), unique_count, h_result.begin());
        d_output_points_ = h_result;
        // 体素序号就是 0..unique_count-1（有序）
        h_output_voxel_ids_.resize(unique_count);
        for (size_t i = 0; i < unique_count; ++i) h_output_voxel_ids_[i] = (int)i;
    }

    d_temp_points_ = d_output_points_;

    ROS_INFO("[GPUPreprocessor] Voxel filter: %zu raw -> %zu voxels -> %zu after density filter (min_pts=%d)",
             input_count, unique_count, output_count, last_voxel_min_points_);
}

void GPUPreprocessor::cuda_launchOutlierRemoval(const PreprocessConfig &config)
{
    int point_count = getCurrentPointCount();
    if (point_count == 0)
    {
        std::cout << "[OutlierRemoval] No points to process" << std::endl;
        return;
    }

    std::cout << "[OutlierRemoval] Processing " << point_count << " points" << std::endl;
    std::cout << "[OutlierRemoval] Parameters: radius=" << config.radius_search
              << ", min_neighbors=" << config.min_radius_neighbors << std::endl;

    // 参数计算 - 针对体素下采样后的点云优化参数
    float grid_size = config.radius_search * 0.4f; // 减小网格大小，提高精度
    int hash_table_size = point_count * 6;         // 🔧 进一步增大哈希表，减少冲突

    std::cout << "[OutlierRemoval] Grid size: " << grid_size
              << ", hash table size: " << hash_table_size << std::endl;

    // 确保缓冲区大小 (复用现有缓冲区)
    d_voxel_keys_.resize(point_count);  // 复用作为point_hashes
    d_knn_indices_.resize(point_count); // 复用作为hash_entries
    // d_hash_table_.resize(hash_table_size);

    // 临时有效性掩码
    static thrust::device_vector<bool> d_valid_mask;
    d_valid_mask.resize(point_count);

    // 临时输出缓冲区
    static thrust::device_vector<GPUPoint3f> d_filtered_points;
    d_filtered_points.resize(point_count);

    // // 调用空间哈希离群点移除
    // int filtered_count = SpatialHashOutlier::launchSpatialHashOutlierRemoval(
    //     thrust::raw_pointer_cast(d_temp_points_.data()),    // 输入
    //     thrust::raw_pointer_cast(d_filtered_points.data()), // 输出
    //     thrust::raw_pointer_cast(d_valid_mask.data()),      // 掩码
    //     thrust::raw_pointer_cast(d_voxel_keys_.data()),     // 复用哈希
    //     thrust::raw_pointer_cast(d_knn_indices_.data()),    // 复用链表
    //     thrust::raw_pointer_cast(d_hash_table_.data()),     // 哈希表
    //     point_count,
    //     config.radius_search,
    //     config.min_radius_neighbors,
    //     grid_size,
    //     hash_table_size);

    // // 更新工作点云
    // d_temp_points_.resize(filtered_count);
    // thrust::copy(d_filtered_points.begin(),
    //              d_filtered_points.begin() + filtered_count,
    //              d_temp_points_.begin());

    // std::cout << "[OutlierRemoval] Result: " << point_count << " -> " << filtered_count
    //           << " points (removed " << (point_count - filtered_count) << " outliers)" << std::endl;
}

void GPUPreprocessor::cuda_launchGroundRemoval(float threshold)
{
    std::cout << "[GPUPreprocessor] Starting ground removal" << std::endl;

    size_t input_count = d_temp_points_.size();
    if (input_count == 0)
        return;

    dim3 block(256);
    dim3 grid((input_count + block.x - 1) / block.x);

    // 重用d_valid_flags_作为ground_flags
    GroundRemoval::ransacGroundDetectionKernel<<<grid, block, 0, stream_>>>(
        thrust::raw_pointer_cast(d_temp_points_.data()),
        thrust::raw_pointer_cast(d_valid_flags_.data()),
        input_count, threshold, 1000);
    cudaStreamSynchronize(stream_);

    // 直接过滤非地面点
    thrust::device_vector<GPUPoint3f> d_temp_result(input_count);

    auto new_end = thrust::copy_if(
        d_temp_points_.begin(), d_temp_points_.begin() + input_count,
        d_valid_flags_.begin(),
        d_temp_result.begin(),
        [] __device__(bool is_ground)
        { return !is_ground; });

    size_t output_count = new_end - d_temp_result.begin();

    // 安全地更新成员变量
    if (output_count > 0)
    {
        thrust::host_vector<GPUPoint3f> h_result(output_count);
        thrust::copy_n(d_temp_result.begin(), output_count, h_result.begin());
        d_output_points_ = h_result;
    }
    else
    {
        d_output_points_.clear();
    }

    d_temp_points_ = d_output_points_;

    std::cout << "[GPUPreprocessor] Ground removal: " << input_count << " -> " << output_count << " points" << std::endl;
}

void GPUPreprocessor::cuda_compactValidPoints()
{
    size_t input_count = d_temp_points_.size();
    if (input_count == 0)
        return;

    // 直接使用thrust::copy_if进行压缩
    thrust::device_vector<GPUPoint3f> d_temp_result(input_count);

    auto new_end = thrust::copy_if(
        d_temp_points_.begin(), d_temp_points_.begin() + input_count,
        d_valid_flags_.begin(),
        d_temp_result.begin(),
        thrust::identity<bool>());

    size_t output_count = new_end - d_temp_result.begin();

    // 安全地更新成员变量
    if (output_count > 0)
    {
        thrust::host_vector<GPUPoint3f> h_result(output_count);
        thrust::copy_n(d_temp_result.begin(), output_count, h_result.begin());
        d_output_points_ = h_result;
    }
    else
    {
        d_output_points_.clear();
    }

    d_temp_points_ = d_output_points_;
}

void GPUPreprocessor::cuda_prepareInputBuffer(size_t count)
{
    // 在.cu文件中，resize是安全的
    if (d_input_points_.size() < count)
    {
        d_input_points_.resize(count);
    }
}

void GPUPreprocessor::cuda_unpackROSMsg(
    const uint8_t* d_raw_data,
    GPUPoint3f* d_output_points,
    int point_step,
    int x_offset, int y_offset, int z_offset, int intensity_offset,
    uint8_t x_datatype, uint8_t y_datatype, uint8_t z_datatype, uint8_t intensity_datatype,
    size_t num_points
)
{
    if (num_points == 0 || d_raw_data == nullptr || d_output_points == nullptr)
        return;

    dim3 block(256);
    dim3 grid((num_points + block.x - 1) / block.x);
    
    unpackROSMsgKernel<<<grid, block, 0, stream_>>>(
        d_raw_data,
        d_output_points,
        point_step,
        x_offset, y_offset, z_offset, intensity_offset,
        x_datatype, y_datatype, z_datatype, intensity_datatype,
        num_points
    );
}

void GPUPreprocessor::cuda_uploadGPUPoints(const GPUPoint3f* h_pinned_points, size_t count)
{
    if (count == 0 || h_pinned_points == nullptr)
        return;

    if (stream_ == nullptr)
    {
        std::cerr << "[ERROR] CUDA stream not initialized" << std::endl;
        return;
    }

    auto start = std::chrono::high_resolution_clock::now();

    // 确保 d_input_points_ 有足够容量
    if (d_input_points_.size() < count)
    {
        d_input_points_.resize(count);
    }

    // 🔥 关键：使用异步上传和 pinned memory（DMA直接访问，避免驱动层拷贝）
    cudaError_t err = cudaMemcpyAsync(
        thrust::raw_pointer_cast(d_input_points_.data()), // 预分配的GPU空间
        h_pinned_points,                                  // CPU源（pinned memory）
        count * sizeof(GPUPoint3f),                        // 字节数
        cudaMemcpyHostToDevice,                            // 传输方向
        stream_                                            // 绑定到stream
    );
    if (err != cudaSuccess)
    {
        std::cerr << "[ERROR] cudaMemcpyAsync failed: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    // 🚀 GPU内部拷贝（超快，异步）
    if (d_temp_points_.size() < count)
    {
        d_temp_points_.resize(count);
    }
    err = cudaMemcpyAsync(
        thrust::raw_pointer_cast(d_temp_points_.data()),
        thrust::raw_pointer_cast(d_input_points_.data()),
        count * sizeof(GPUPoint3f),
        cudaMemcpyDeviceToDevice,
        stream_  // 绑定到stream
    );
    if (err != cudaSuccess)
    {
        std::cerr << "[ERROR] GPU internal copy failed: " << cudaGetErrorString(err) << std::endl;
        return;
    }

    // 🔧 关键修复：正确设置d_temp_points_的逻辑大小
    d_temp_points_.resize(count);

    // 移除同步：保持异步，由调用者决定何时同步
    // cudaStreamSynchronize(stream_);  // 仅在必要时同步

    auto end = std::chrono::high_resolution_clock::now();
    float upload_time = std::chrono::duration<float, std::milli>(end - start).count();

    std::cout << "[GPUPreprocessor] ⚡ ASYNC upload: " << count
              << " points in " << upload_time << " ms (pinned memory + async)" << std::endl;
}

void GPUPreprocessor::reserveMemory(size_t max_points)
{
    // 使用resize()而不是reserve()来预分配内存
    d_input_points_.resize(max_points);
    d_temp_points_.resize(max_points);
    d_output_points_.resize(max_points);
    d_voxel_keys_.resize(max_points);
    d_valid_flags_.resize(max_points);
    // d_radix_temp_points_.resize(max_points);
    // d_radix_temp_keys_.resize(max_points);

    std::cout << "[GPUPreprocessor] Pre-allocated memory for " << max_points << " points" << std::endl;
}

void GPUPreprocessor::clearMemory()
{
    d_input_points_.clear();
    d_temp_points_.clear();
    d_output_points_.clear();
    d_voxel_keys_.clear();
    d_voxel_boundaries_.clear();
    d_unique_keys_.clear();
    d_neighbor_counts_.clear();
    d_valid_flags_.clear();
    d_knn_indices_.clear();
    d_knn_distances_.clear();

    d_input_points_.shrink_to_fit();
    d_temp_points_.shrink_to_fit();
    d_output_points_.shrink_to_fit();
}

// 桶排序代替
namespace GPUBucketSort
{

    // Step 1: 分析key分布，确定桶的范围
    __global__ void analyzeKeyRangeKernel(
        const uint64_t *keys,
        int count,
        uint64_t *min_key,
        uint64_t *max_key)
    {

        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= count)
            return;

        // 使用block-level reduction找min/max
        __shared__ uint64_t smin[256], smax[256];

        smin[threadIdx.x] = (idx < count) ? keys[idx] : UINT64_MAX;
        smax[threadIdx.x] = (idx < count) ? keys[idx] : 0;

        __syncthreads();

        // Reduction in shared memory
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
        {
            if (threadIdx.x < stride)
            {
                smin[threadIdx.x] = min(smin[threadIdx.x], smin[threadIdx.x + stride]);
                smax[threadIdx.x] = max(smax[threadIdx.x], smax[threadIdx.x + stride]);
            }
            __syncthreads();
        }

        if (threadIdx.x == 0)
        {
            atomicMin((unsigned long long *)min_key, (unsigned long long)smin[0]);
            atomicMax((unsigned long long *)max_key, (unsigned long long)smax[0]);
        }
    }

    // Step 2: 计算每个点属于哪个桶
    __global__ void computeBucketIndicesKernel(
        const uint64_t *keys,
        int *bucket_indices,
        int count,
        uint64_t min_key,
        uint64_t key_range,
        int num_buckets)
    {

        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= count)
            return;

        uint64_t key = keys[idx];
        uint64_t normalized_key = key - min_key;

        // 避免除法，使用位运算（如果num_buckets是2的幂）
        int bucket_id = (int)((normalized_key * num_buckets) / (key_range + 1));
        bucket_id = min(bucket_id, num_buckets - 1); // 确保不越界

        bucket_indices[idx] = bucket_id;
    }

    // Step 3: 统计每个桶的大小
    __global__ void countBucketSizesKernel(
        const int *bucket_indices,
        int *bucket_counts,
        int count,
        int num_buckets)
    {

        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= count)
            return;

        int bucket_id = bucket_indices[idx];
        atomicAdd(&bucket_counts[bucket_id], 1);
    }

    // Step 4: 计算每个桶的起始位置（prefix sum）
    __global__ void computeBucketOffsetsKernel(
        const int *bucket_counts,
        int *bucket_offsets,
        int num_buckets)
    {

        // 简单的sequential prefix sum (可以优化为并行)
        if (blockIdx.x == 0 && threadIdx.x == 0)
        {
            bucket_offsets[0] = 0;
            for (int i = 1; i < num_buckets; i++)
            {
                bucket_offsets[i] = bucket_offsets[i - 1] + bucket_counts[i - 1];
            }
        }
    }

    // Step 5: 将数据分配到各个桶
    __global__ void distributeToBucketsKernel(
        const GPUPoint3f *input_points,
        const uint64_t *input_keys,
        const int *bucket_indices,
        const int *bucket_offsets,
        GPUPoint3f *output_points,
        uint64_t *output_keys,
        int *bucket_positions, // 每个桶当前位置的原子计数器
        int count)
    {

        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= count)
            return;

        int bucket_id = bucket_indices[idx];
        int pos = atomicAdd(&bucket_positions[bucket_id], 1);
        int output_idx = bucket_offsets[bucket_id] + pos;

        output_points[output_idx] = input_points[idx];
        output_keys[output_idx] = input_keys[idx];
    }

    // Step 6: 对每个桶内部排序（使用简单的并行插入排序）
    __global__ void sortWithinBucketsKernel(
        GPUPoint3f *points,
        uint64_t *keys,
        const int *bucket_offsets,
        const int *bucket_counts,
        int num_buckets)
    {

        int bucket_id = blockIdx.x * blockDim.x + threadIdx.x;
        if (bucket_id >= num_buckets)
            return;

        int start = bucket_offsets[bucket_id];
        int size = bucket_counts[bucket_id];

        if (size <= 1)
            return;

        // 单线程对每个桶进行插入排序
        for (int i = start + 1; i < start + size; i++)
        {
            uint64_t key = keys[i];
            GPUPoint3f point = points[i];
            int j = i - 1;

            // 标准插入排序
            while (j >= start && keys[j] > key)
            {
                keys[j + 1] = keys[j];
                points[j + 1] = points[j];
                j--;
            }
            keys[j + 1] = key;
            points[j + 1] = point;
        }
    }
    // 在GPUBucketSort namespace中添加：

    __global__ void radixSortWithinBucketsKernel(
        GPUPoint3f *points,
        uint64_t *keys,
        GPUPoint3f *temp_points, // 临时缓冲区
        uint64_t *temp_keys,     // 临时缓冲区
        const int *bucket_offsets,
        const int *bucket_counts,
        int num_buckets)
    {

        int bucket_id = blockIdx.x;
        if (bucket_id >= num_buckets)
            return;

        int start = bucket_offsets[bucket_id];
        int size = bucket_counts[bucket_id];

        if (size <= 1)
            return;

        // � 优化1: 使用warp内协作，每个桶32个线程
        int lane = threadIdx.x; // 0-31
        int warp_size = 32;

        // 优化2: 8位基数排序，但并行处理
        for (int pass = 0; pass < 8; pass++)
        {
            int shift = pass * 8;

            // 优化3: 使用shared memory减少全局内存访问
            __shared__ int shared_counts[256];

            // 初始化共享内存计数器（并行）
            for (int i = lane; i < 256; i += warp_size)
            {
                shared_counts[i] = 0;
            }
            __syncthreads();

            // Step 1: 并行统计字节值出现次数
            for (int i = lane; i < size; i += warp_size)
            {
                int digit = (keys[start + i] >> shift) & 0xFF;
                atomicAdd(&shared_counts[digit], 1);
            }
            __syncthreads();

            // Step 2: 并行前缀和计算
            // 简单的串行前缀和（由单线程完成，因为只有256个元素）
            if (lane == 0)
            {
                for (int i = 1; i < 256; i++)
                {
                    shared_counts[i] += shared_counts[i - 1];
                }
            }
            __syncthreads();

            // Step 3: 并行分配到临时数组
            //  优化4: 使用局部原子操作减少冲突
            for (int i = size - 1 - lane; i >= 0; i -= warp_size)
            {
                if (i >= 0)
                {
                    int digit = (keys[start + i] >> shift) & 0xFF;
                    int pos = atomicSub(&shared_counts[digit], 1) - 1;
                    temp_keys[start + pos] = keys[start + i];
                    temp_points[start + pos] = points[start + i];
                }
            }
            __syncthreads();

            // Step 4: 并行复制回原数组
            for (int i = lane; i < size; i += warp_size)
            {
                keys[start + i] = temp_keys[start + i];
                points[start + i] = temp_points[start + i];
            }
            __syncthreads();
        }
    }

} // namespace GPUBucketSort

// ========== ROS消息解包相关 ==========
// 数据类型常量定义（与sensor_msgs::PointField一致）
namespace {
    constexpr uint8_t POINT_FIELD_INT8 = 1;
    constexpr uint8_t POINT_FIELD_UINT8 = 2;
    constexpr uint8_t POINT_FIELD_INT16 = 3;
    constexpr uint8_t POINT_FIELD_UINT16 = 4;
    constexpr uint8_t POINT_FIELD_INT32 = 5;
    constexpr uint8_t POINT_FIELD_UINT32 = 6;
    constexpr uint8_t POINT_FIELD_FLOAT32 = 7;
    constexpr uint8_t POINT_FIELD_FLOAT64 = 8;
}

// 数据类型读取辅助函数
__device__ inline float readFloat(const uint8_t* ptr, uint8_t datatype)
{
    switch (datatype)
    {
        case POINT_FIELD_FLOAT32:
            return *reinterpret_cast<const float*>(ptr);
        case POINT_FIELD_FLOAT64:
            return static_cast<float>(*reinterpret_cast<const double*>(ptr));
        case POINT_FIELD_INT8:
            return static_cast<float>(*reinterpret_cast<const int8_t*>(ptr));
        case POINT_FIELD_UINT8:
            return static_cast<float>(*reinterpret_cast<const uint8_t*>(ptr));
        case POINT_FIELD_INT16:
            return static_cast<float>(*reinterpret_cast<const int16_t*>(ptr));
        case POINT_FIELD_UINT16:
            return static_cast<float>(*reinterpret_cast<const uint16_t*>(ptr));
        case POINT_FIELD_INT32:
            return static_cast<float>(*reinterpret_cast<const int32_t*>(ptr));
        case POINT_FIELD_UINT32:
            return static_cast<float>(*reinterpret_cast<const uint32_t*>(ptr));
        default:
            return 0.0f; // 不支持的类型，返回0
    }
}

// ROS消息解包内核
__global__ void unpackROSMsgKernel(
    const uint8_t* raw_data,
    GPUPoint3f* output_points,
    int point_step,
    int x_offset, int y_offset, int z_offset, int intensity_offset,
    uint8_t x_datatype, uint8_t y_datatype, uint8_t z_datatype, uint8_t intensity_datatype,
    size_t num_points
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points)
        return;

    const uint8_t* point_data = raw_data + idx * point_step;
    GPUPoint3f& out = output_points[idx];

    // 解析x, y, z（必须存在）
    out.x = readFloat(point_data + x_offset, x_datatype);
    out.y = readFloat(point_data + y_offset, y_datatype);
    out.z = readFloat(point_data + z_offset, z_datatype);

    // 解析intensity（如果存在）
    if (intensity_offset >= 0)
    {
        out.intensity = readFloat(point_data + intensity_offset, intensity_datatype);
    }
    else
    {
        out.intensity = 0.0f;
    }
}
