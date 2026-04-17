#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>

// 前向声明GPU点类型
struct GPUPoint3f;

// ========== 体素下采样相关 ==========
namespace VoxelFilter
{
    // 计算体素哈希key
    __global__ void computeVoxelKeysKernel(
        const GPUPoint3f *points,
        uint64_t *voxel_keys,
        float voxel_size,
        int point_count);

    // 计算体素质心
    __global__ void computeVoxelCentroidsKernel(
        const GPUPoint3f *sorted_points,
        const int *voxel_boundaries,
        const int *voxel_counts,
        GPUPoint3f *output_points,
        int unique_voxel_count);

    // Device函数
    __device__ inline uint64_t computeVoxelHash(float x, float y, float z, float voxel_size);
    __device__ inline void unpackVoxelHash(uint64_t hash, int &vx, int &vy, int &vz);
}

// ========== 离群点移除相关 ==========
namespace OutlierRemoval
{
    // 统计离群点检测
    __global__ void statisticalOutlierKernel(
        const GPUPoint3f *points,
        bool *valid_flags,
        int point_count,
        int k,
        float std_dev_multiplier);

    // 半径离群点检测
    __global__ void radiusOutlierKernel(
        const GPUPoint3f *points,
        bool *valid_flags,
        int point_count,
        float radius,
        int min_neighbors);

    // KNN查询kernel
    __global__ void findKNearestNeighborsKernel(
        const GPUPoint3f *points,
        int *neighbor_indices,
        float *neighbor_distances,
        int point_count,
        int k);

    // Device函数
    __device__ inline float computeDistance(const GPUPoint3f &p1, const GPUPoint3f &p2);
    __device__ inline void insertionSort(float *distances, int *indices, int k, float new_dist, int new_idx);
}

// ========== 地面移除相关 ==========
namespace GroundRemoval
{
    // RANSAC地面检测
    __global__ void ransacGroundDetectionKernel(
        const GPUPoint3f *points,
        bool *ground_flags,
        int point_count,
        float threshold,
        int max_iterations);

    // Device函数
    __device__ inline void fitPlaneRANSAC(
        const GPUPoint3f *points,
        int point_count,
        float plane_coeffs[4], // ax + by + cz + d = 0
        int *inlier_count,
        float threshold,
        int max_iterations);

    __device__ inline float pointToPlaneDistance(
        const GPUPoint3f &point,
        const float plane_coeffs[4]);
}

// ========== 工具函数 ==========
namespace Utils
{
    // 点云压缩 (移除无效点)
    __global__ void compactPointsKernel(
        const GPUPoint3f *input_points,
        const bool *valid_flags,
        GPUPoint3f *output_points,
        int *output_indices,
        int point_count);
}

// ========== GPU 桶排序相关 ==========
namespace GPUBucketSort
{
    // Step 1: 分析key分布，确定桶的范围
    __global__ void analyzeKeyRangeKernel(
        const uint64_t *keys,
        int count,
        uint64_t *min_key,
        uint64_t *max_key);

    // Step 2: 计算每个点属于哪个桶
    __global__ void computeBucketIndicesKernel(
        const uint64_t *keys,
        int *bucket_indices,
        int count,
        uint64_t min_key,
        uint64_t key_range,
        int num_buckets);

    // Step 3: 统计每个桶的大小
    __global__ void countBucketSizesKernel(
        const int *bucket_indices,
        int *bucket_counts,
        int count,
        int num_buckets);

    // Step 5: 将数据分配到各个桶
    __global__ void distributeToBucketsKernel(
        const GPUPoint3f *input_points,
        const uint64_t *input_keys,
        const int *bucket_indices,
        const int *bucket_offsets,
        GPUPoint3f *output_points,
        uint64_t *output_keys,
        int *bucket_positions,
        int count);

    // Step 6: 对每个桶内部排序（使用基数排序）
    __global__ void radixSortWithinBucketsKernel(
        GPUPoint3f *points,
        uint64_t *keys,
        GPUPoint3f *temp_points,
        uint64_t *temp_keys,
        const int *bucket_offsets,
        const int *bucket_counts,
        int num_buckets);
}
// ========== ROS消息解包相关 ==========
// ROS消息解包内核
__global__ void unpackROSMsgKernel(
    const uint8_t* raw_data,
    GPUPoint3f* output_points,
    int point_step,
    int x_offset, int y_offset, int z_offset, int intensity_offset,
    uint8_t x_datatype, uint8_t y_datatype, uint8_t z_datatype, uint8_t intensity_datatype,
    size_t num_points
);

// ========== CUDA错误检查宏 ==========
#define CUDA_CHECK(call)                                                                                \
    do                                                                                                  \
    {                                                                                                   \
        cudaError_t error = call;                                                                       \
        if (error != cudaSuccess)                                                                       \
        {                                                                                               \
            fprintf(stderr, "CUDA error at %s:%d - %s", __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(1);                                                                                    \
        }                                                                                               \
    } while (0)

#define KERNEL_CHECK()                                                                                         \
    do                                                                                                         \
    {                                                                                                          \
        cudaError_t error = cudaGetLastError();                                                                \
        if (error != cudaSuccess)                                                                              \
        {                                                                                                      \
            fprintf(stderr, "CUDA kernel error at %s:%d - %s", __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(1);                                                                                           \
        }                                                                                                      \
        cudaDeviceSynchronize();                                                                               \
    } while (0)
