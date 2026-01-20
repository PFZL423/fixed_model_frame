#pragma once
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "gpu_demo/QuadricDetect.h" // 包含 GPUPoint3f 和 GPUQuadricModel 的定义

// ========================================
// GPU内核函数声明 - 全GPU化RANSAC的核心
// ========================================

/**
 * @brief 初始化GPU随机数生成器状态
 * 为每个GPU线程分配独立的curand状态，确保并行采样的随机性
 * @param states [out] 随机数状态数组
 * @param seed 随机种子
 * @param n 需要初始化的状态数量
 */
__global__ void initCurandStates_Kernel(curandState *states, unsigned long seed, int n);

/**
 * @brief 批量采样和矩阵构建内核 - 核心创新
 * 每个GPU线程并行采样9个点并构建对应的9×10约束矩阵A
 * 相比point包的CPU串行采样，实现~100x加速
 * @param all_points 所有点云数据 (GPU)
 * @param remaining_indices 剩余点索引 (GPU)
 * @param num_remaining 剩余点数量
 * @param rand_states GPU随机数状态
 * @param batch_size 并行处理的模型数量 (通常1024)
 * @param batch_matrices [out] 输出的批量矩阵 (batch_size × 9 × 10)
 */
__global__ void sampleAndBuildMatrices_Kernel(
    const GPUPoint3f *all_points,
    const int *remaining_indices,
    int num_remaining,
    curandState *rand_states,
    int batch_size,
    float *batch_matrices);

/**
 * @brief 批量内点计数内核 - 2D并行验证
 * 使用2D Grid架构：blockIdx.y对应模型ID，blockIdx.x×threadIdx.x对应点ID
 * 每个block内使用shared memory reduce提高效率
 * @param all_points 所有点云数据 (GPU)
 * @param remaining_indices 剩余点索引 (GPU)
 * @param num_remaining 剩余点数量
 * @param batch_models 批量二次曲面模型 (GPU)
 * @param batch_size 模型数量
 * @param threshold 内点距离阈值
 * @param batch_inlier_counts [out] 每个模型的内点计数
 */
__global__ void countInliersBatch_Kernel(
    const GPUPoint3f *all_points,
    const int *remaining_indices,
    int num_remaining,
    const GPUQuadricModel *batch_models,
    int batch_size,
    float threshold,
    int *batch_inlier_counts);

/**
 * @brief 最优模型查找内核
 * 使用GPU并行reduce在batch中找出内点数最多的模型
 * @param batch_inlier_counts 每个模型的内点计数数组
 * @param batch_size 模型数量
 * @param best_index [out] 最优模型的索引
 * @param best_count [out] 最优模型的内点数
 */
__global__ void findBestModel_Kernel(
    const int *batch_inlier_counts,
    int batch_size,
    int *best_index,
    int *best_count);

/**
 * @brief 内点提取内核
 * 提取指定模型的所有内点索引，用于后续精炼
 * @param all_points 所有点云数据 (GPU)
 * @param remaining_indices 剩余点索引 (GPU)
 * @param num_remaining 剩余点数量
 * @param model 用于提取内点的二次曲面模型
 * @param threshold 内点距离阈值
 * @param inlier_indices [out] 提取的内点索引数组
 * @param inlier_count [out] 内点数量计数器
 */
__global__ void extractInliers_Kernel(
    const GPUPoint3f *all_points,
    const int *remaining_indices,
    int num_remaining,
    const GPUQuadricModel *model,
    float threshold,
    int *inlier_indices,
    int *inlier_count);

/**
 * @brief 移除内点内核 - GPU并行点移除
 * @param remaining_points 当前剩余点索引
 * @param remaining_count 剩余点数量
 * @param sorted_inliers 已排序的内点索引
 * @param inlier_count 内点数量
 * @param output_points [out] 输出的新剩余点索引
 * @param output_count [out] 输出的新剩余点数量
 */
__global__ void removePointsKernel(
    const int *remaining_points,
    int remaining_count,
    const int *sorted_inliers,
    int inlier_count,
    int *output_points,
    int *output_count);

// ========================================
// GPU设备函数 - 内联数学计算
// ========================================

/**
 * @brief 计算点到二次曲面的代数距离
 * 实现公式：|[x y z 1] * Q * [x y z 1]^T|
 * @param point 3D点坐标
 * @param model 4×4二次曲面矩阵Q (展开为16个float)
 * @return 点到曲面的代数距离的绝对值
 */
__device__ inline float evaluateQuadricDistance(
    const GPUPoint3f &point,
    const GPUQuadricModel &model);

// ========================================
// 🆕 反幂迭代相关内核函数声明
// ========================================

/**
 * @brief 计算A^T*A矩阵内核
 * 从9×10的A矩阵计算10×10的A^T*A对称矩阵
 * @param batch_matrices 输入：1024个9×10矩阵
 * @param batch_ATA_matrices [out] 输出：1024个10×10 A^T*A矩阵
 * @param batch_size 批量大小
 */
__global__ void computeATA_Kernel(
    const float *batch_matrices,
    float *batch_ATA_matrices,
    int batch_size);

/**
 * @brief 批量QR分解内核
 * 对1024个10×10对称矩阵并行进行QR分解
 * @param batch_ATA_matrices 输入：1024个10×10对称矩阵
 * @param batch_R_matrices [out] 输出：1024个10×10上三角矩阵
 * @param batch_size 批量大小
 */
__global__ void batchQR_Kernel(
    const float *batch_ATA_matrices,
    float *batch_R_matrices,
    int batch_size);

/**
 * @brief 批量反幂迭代内核
 * 对1024个10×10 R矩阵并行进行反幂迭代求最小特征向量
 * @param batch_R_matrices 输入：1024个10×10 R矩阵
 * @param batch_eigenvectors [out] 输出：1024个10维最小特征向量
 * @param rand_states 随机数状态
 * @param batch_size 批量大小
 */
__global__ void batchInversePowerIteration_Kernel(
    const float *batch_R_matrices,
    float *batch_eigenvectors,
    curandState *rand_states,
    int batch_size);

/**
 * @brief 提取二次曲面模型系数内核
 * 从10维特征向量构建GPUQuadricModel.coeffs[16]数组
 * @param batch_eigenvectors 输入：1024个10维特征向量
 * @param batch_models [out] 输出：1024个二次曲面模型
 * @param batch_size 批量大小
 */
__global__ void extractQuadricModels_Kernel(
    const float *batch_eigenvectors,
    GPUQuadricModel *batch_models,
    int batch_size);
