#pragma once
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "PlaneDetect/PlaneDetect.h" // 包含 GPUPoint3f 的定义

/**
 * @brief 初始化GPU随机数生成器状态
 * 为每个GPU线程分配独立的curand状态，确保并行采样的随机性
 * @param states [out] 随机数状态数组
 * @param seed 随机种子
 * @param n 需要初始化的状态数量
 */
__global__ void initCurandStates_Kernel(curandState *states, unsigned long seed, int n);

/**
 * @brief 批量采样和平面拟合内核 - 简化版采样
 * 每个GPU线程并行采样3个点并直接计算平面方程系数
 * 相比二次曲面的9点采样，大幅简化计算复杂度
 * @param all_points 所有点云数据 (GPU)
 * @param remaining_indices 剩余点索引 (GPU)
 * @param num_remaining 剩余点数量
 * @param rand_states GPU随机数状态
 * @param batch_size 并行处理的模型数量 (通常2048)
 * @param batch_models [out] 输出的批量平面模型
 */
__global__ void sampleAndFitPlanes_Kernel(
    const GPUPoint3f *all_points,
    const int *remaining_indices,
    int num_remaining,
    curandState *rand_states,
    int batch_size,
    GPUPlaneModel *batch_models);

/**
 * @brief 批量内点计数内核 - 2D并行验证
 * 使用2D Grid架构：blockIdx.y对应模型ID，blockIdx.x×threadIdx.x对应点ID
 * 每个block内使用shared memory reduce提高效率
 * @param all_points 所有点云数据 (GPU)
 * @param remaining_indices 剩余点索引 (GPU)
 * @param num_remaining 剩余点数量
 * @param batch_models 批量平面模型 (GPU)
 * @param batch_size 模型数量
 * @param threshold 内点距离阈值
 * @param batch_inlier_counts [out] 每个模型的内点计数
 */
__global__ void countInliersBatch_Kernel(
    const GPUPoint3f *all_points,
    const int *remaining_indices,
    int num_remaining,
    const GPUPlaneModel *batch_models,
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
 * 提取指定模型的所有内点索引，用于后续处理
 * @param all_points 所有点云数据 (GPU)
 * @param remaining_indices 剩余点索引 (GPU)
 * @param num_remaining 剩余点数量
 * @param model 用于提取内点的平面模型
 * @param threshold 内点距离阈值
 * @param inlier_indices [out] 提取的内点索引数组
 * @param inlier_count [out] 内点数量计数器
 */
__global__ void extractInliers_Kernel(
    const GPUPoint3f *all_points,
    const int *remaining_indices,
    int num_remaining,
    const GPUPlaneModel *model,
    float threshold,
    int *inlier_indices,
    int *inlier_count);

/**
 * @brief 移除内点内核 - GPU并行点移除
 * @param remaining_points 当前剩余点索引
 * @param remaining_count 剩余点数量
 * @param inlier_indices 内点索引（不需要排序）
 * @param inlier_count 内点数量
 * @param output_points [out] 输出的新剩余点索引
 * @param output_count [out] 输出的新剩余点数量
 */
__global__ void removePointsKernel(
    const int *remaining_points,
    int remaining_count,
    const int *inlier_indices,
    int inlier_count,
    int *output_points,
    int *output_count);

// ========================================
// GPU设备函数 - 内联数学计算
// ========================================

/**
 * @brief 从三个点计算平面方程系数
 * 使用叉积计算法向量，然后求解平面方程 ax + by + cz + d = 0
 * @param p1, p2, p3 三个不共线的3D点
 * @param model [out] 输出的平面模型系数
 * @return true表示成功计算，false表示三点共线
 */
__device__ inline bool computePlaneFromThreePoints(
    const GPUPoint3f &p1,
    const GPUPoint3f &p2,
    const GPUPoint3f &p3,
    GPUPlaneModel &model);

/**
 * @brief 计算点到平面的距离
 * 实现公式：|ax + by + cz + d| / sqrt(a² + b² + c²)
 * @param point 3D点坐标
 * @param model 平面方程系数 [a, b, c, d]
 * @return 点到平面的几何距离
 */
__device__ inline float evaluatePlaneDistance(
    const GPUPoint3f &point,
    const GPUPlaneModel &model);
