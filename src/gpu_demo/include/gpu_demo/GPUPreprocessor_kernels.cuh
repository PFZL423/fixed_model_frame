#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>

// 前向声明GPU点类型
struct GPUPoint3f;
struct GPUPointNormal3f;

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

// ========== 法线估计相关 ==========

namespace SpatialHashNormals {
    
    // 空间哈希表构建
    __global__ void buildSpatialHashKernel(
        const GPUPoint3f* points,
        uint64_t* point_hashes,
        int* hash_table,
        int* hash_entries,
        int num_points,
        float grid_size,
        int hash_table_size);
    
    // 基于空间哈希的KNN搜索 + 法线计算一体化
    __global__ void spatialHashNormalsKernel(
        const GPUPoint3f* points,
        const uint64_t* point_hashes,
        const int* hash_table,
        const int* hash_entries,
        GPUPointNormal3f* points_with_normals,
        int num_points,
        float search_radius,
        int min_neighbors,
        float grid_size,
        int hash_table_size);
    
    // Device函数
    __device__ inline uint64_t computeSpatialHash(
        float x, float y, float z, float grid_size);
    
    __device__ inline void fastEigen3x3(
        float cov[6], float* normal, float* curvature);
    
    __device__ inline void searchHashGrid(
        const GPUPoint3f& query_point,
        const GPUPoint3f* all_points,
        const uint64_t* point_hashes,
        const int* hash_table,
        const int* hash_entries,
        int* neighbors,
        float* distances,
        int* neighbor_count,
        float search_radius,
        float grid_size,
        int hash_table_size,
        int max_neighbors);
}
namespace NormalEstimation
{
    // 法线估计主kernel
    __global__ void estimateNormalsKernel(
        const GPUPoint3f *points,
        GPUPointNormal3f *points_with_normals,
        int point_count,
        float radius,
        int k);

    // KNN搜索专用kernel
    __global__ void findNormalKNNKernel(
        const GPUPoint3f *points,
        int *knn_indices,
        float *knn_distances,
        int point_count,
        float radius,
        int k);

    // PCA计算kernel
    __global__ void computePCANormalsKernel(
        const GPUPoint3f *points,
        const int *knn_indices,
        GPUPointNormal3f *points_with_normals,
        int point_count,
        int k);

    // Device函数
    __device__ inline void computeCovarianceMatrix(
        const GPUPoint3f *neighbors,
        int neighbor_count,
        float cov_matrix[6] // 对称矩阵，只存储上三角
    );

    __device__ inline void eigenDecomposition3x3(
        const float cov_matrix[6],
        float eigenvalues[3],
        float eigenvectors[9]);

    __device__ inline void selectNormalDirection(
        const GPUPoint3f &query_point,
        const GPUPoint3f *neighbors,
        int neighbor_count,
        float normal[3]);
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

    // 点类型转换
    __global__ void convertToPointNormalKernel(
        const GPUPoint3f *input_points,
        GPUPointNormal3f *output_points,
        int point_count);

    // 拷贝法线到PointNormal结构
    __global__ void copyNormalsKernel(
        const GPUPoint3f *points,
        const float *normals, // [nx, ny, nz, nx, ny, nz, ...]
        GPUPointNormal3f *points_with_normals,
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
namespace SpatialHashOutlier {
    
    // 离群点移除kernel
    __global__ void spatialHashOutlierKernel(
        const GPUPoint3f* input_points,
        bool* is_valid,
        const uint64_t* point_hashes,
        const int* hash_table,
        const int* hash_entries,
        int num_points,
        float search_radius,
        int min_neighbors_threshold,
        float grid_size,
        int hash_table_size);
    
    // 离群点移除主函数
    int launchSpatialHashOutlierRemoval(
        const GPUPoint3f* d_input_points,
        GPUPoint3f* d_output_points,
        bool* d_valid_mask,
        uint64_t* d_point_hashes,
        int* d_hash_entries,
        int* d_hash_table,
        int point_count,
        float outlier_radius,
        int min_neighbors_threshold,
        float grid_size,
        int hash_table_size);
    
    // 公共哈希构建函数 (从法线估计中抽取)
    void buildCommonSpatialHash(
        const GPUPoint3f* points,
        uint64_t* point_hashes,
        int* hash_table,
        int* hash_entries,
        int point_count,
        float grid_size,
        int hash_table_size);
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
