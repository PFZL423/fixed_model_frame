#include "gpu_demo/QuadricDetect.h"
#include "gpu_demo/QuadricDetect_kernels.cuh"
#include <cusolverDn.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/extrema.h>  // 必须包含，用于 max_element
#include <thrust/device_ptr.h>
#include <thrust/gather.h>   // 用于 gather 操作
#include <thrust/remove.h>   // 用于 remove_if
#include <thrust/distance.h> // 用于 distance
#include <thrust/copy.h>     // 用于 copy_if
#include <thrust/iterator/permutation_iterator.h> // 用于 make_permutation_iterator
#include <thrust/functional.h> // 用于 identity
#include <ctime>
#include <iostream>
#include <cmath>     // 添加这个头文件用于isfinite函数
#include <algorithm> // 添加这个头文件用于min函数
#include <stdexcept> // 用于 std::exception
#include <chrono>    // 用于纳秒级时间戳
#include <unistd.h>  // 用于 getpid()

// ========================================
// CUDA内核函数定义 (每个内核只定义一次!)
// ========================================

// 静态变量：保存初始化时的随机数状态地址
static void* g_init_rand_states_addr = nullptr;

__launch_bounds__(128)
__global__ void initCurandStates_Kernel(curandState *states, unsigned long long seed, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // 强制打印 GPU 端的索引：如果控制台没动静，说明内核罢工了
    if (idx < 10) {
        printf("Thread %d initializing\n", idx);
    }
    if (idx < n)
    {
        // 强制初始化：手动给 states[idx] 赋一个非零的占位值
        states[idx] = curandState(); // 使用默认构造函数初始化
        
        // 修正：使用 idx 作为 sequence 参数是 NVIDIA 官方推荐的确保 1024 个线程随机序列互不相关的标准做法
        curand_init(seed, idx, 0, &states[idx]);
    }
}

// 调试内核：验证随机数状态是否不同（不修改原始状态）
__global__ void debugRandomStates_Kernel(const curandState *rand_states, int n, unsigned int *output_rands)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        // 修复：使用 const 指针，创建临时副本，不修改原始状态
        curandState temp_state = rand_states[idx];
        // 生成3个随机数来验证状态是否不同
        unsigned int r1 = curand(&temp_state);
        unsigned int r2 = curand(&temp_state);
        unsigned int r3 = curand(&temp_state);
        // 存储到输出数组（每个状态3个随机数）
        output_rands[idx * 3 + 0] = r1;
        output_rands[idx * 3 + 1] = r2;
        output_rands[idx * 3 + 2] = r3;
        // 不恢复状态，因为使用的是临时副本
    }
}

// ========================================
// PCA局部坐标系构建辅助函数
// ========================================

/**
 * @brief 使用卡尔丹公式求解3×3对称矩阵的特征值
 * @param C 3×3对称矩阵（9个float，行主序存储）
 * @param eigenvalues [out] 3个特征值（按从小到大排序）
 */
__device__ void solveCubicEigenvalues(const float *C, float *eigenvalues)
{
    // 对于3×3对称矩阵C，特征多项式为 det(C - λI) = -λ³ + trace(C)λ² - ... + det(C) = 0
    // 转换为标准形式：λ³ + aλ² + bλ + c = 0
    // 其中 a = -trace(C), b = (trace(C)² - trace(C²))/2, c = -det(C)
    
    float trace_C = C[0] + C[4] + C[8]; // C[0,0] + C[1,1] + C[2,2]
    
    // 计算C²的迹
    float trace_C2 = C[0]*C[0] + C[1]*C[3] + C[2]*C[6] +  // 第一行
                     C[3]*C[1] + C[4]*C[4] + C[5]*C[7] +  // 第二行
                     C[6]*C[2] + C[7]*C[5] + C[8]*C[8];   // 第三行
    
    // 计算行列式 det(C)
    float det_C = C[0] * (C[4]*C[8] - C[5]*C[7]) -
                  C[1] * (C[3]*C[8] - C[5]*C[6]) +
                  C[2] * (C[3]*C[7] - C[4]*C[6]);
    
    // 特征多项式系数
    float a = -trace_C;
    float b = (trace_C * trace_C - trace_C2) * 0.5f;
    float c = -det_C;
    
    // 卡尔丹公式：将 λ³ + aλ² + bλ + c = 0 转换为 t³ + pt + q = 0
    // 其中 t = λ + a/3
    float p = b - a * a / 3.0f;
    float q = (2.0f * a * a * a) / 27.0f - (a * b) / 3.0f + c;
    
    // 判别式 Δ = (q/2)² + (p/3)³
    float delta = (q * q) / 4.0f + (p * p * p) / 27.0f;
    
    float sqrt_delta = sqrtf(fabsf(delta));
    
    if (delta >= 0.0f)
    {
        // 一个实根和两个共轭复根（退化情况，取实根）
        float u = cbrtf(-q / 2.0f + sqrt_delta);
        float v = cbrtf(-q / 2.0f - sqrt_delta);
        float t1 = u + v;
        
        // 转换回λ
        eigenvalues[0] = t1 - a / 3.0f;
        eigenvalues[1] = eigenvalues[0]; // 重复根
        eigenvalues[2] = eigenvalues[0];
    }
    else
    {
        // 三个不同的实根
        float rho = sqrtf(-p * p * p / 27.0f);
        float theta = acosf(-q / (2.0f * rho));
        
        float t1 = 2.0f * cbrtf(rho) * cosf(theta / 3.0f);
        float t2 = 2.0f * cbrtf(rho) * cosf((theta + 2.0f * 3.14159265359f) / 3.0f);
        float t3 = 2.0f * cbrtf(rho) * cosf((theta + 4.0f * 3.14159265359f) / 3.0f);
        
        // 转换回λ并排序
        eigenvalues[0] = t1 - a / 3.0f;
        eigenvalues[1] = t2 - a / 3.0f;
        eigenvalues[2] = t3 - a / 3.0f;
        
        // 简单排序（冒泡排序）
        if (eigenvalues[0] > eigenvalues[1]) { float tmp = eigenvalues[0]; eigenvalues[0] = eigenvalues[1]; eigenvalues[1] = tmp; }
        if (eigenvalues[1] > eigenvalues[2]) { float tmp = eigenvalues[1]; eigenvalues[1] = eigenvalues[2]; eigenvalues[2] = tmp; }
        if (eigenvalues[0] > eigenvalues[1]) { float tmp = eigenvalues[0]; eigenvalues[0] = eigenvalues[1]; eigenvalues[1] = tmp; }
    }
}

/**
 * @brief 计算最小特征值对应的特征向量（通过列向量叉乘）
 * @param C 3×3对称矩阵（9个float，行主序存储）
 * @param lambda_min 最小特征值
 * @param eigenvector [out] 归一化的特征向量（3个float）
 */
__device__ void computeEigenvector(const float *C, float lambda_min, float *eigenvector)
{
    // 构造 (C - λ_min*I)
    float C_minus_lambda[9];
    for (int i = 0; i < 9; ++i)
    {
        C_minus_lambda[i] = C[i];
    }
    C_minus_lambda[0] -= lambda_min; // C[0,0] - λ
    C_minus_lambda[4] -= lambda_min; // C[1,1] - λ
    C_minus_lambda[8] -= lambda_min; // C[2,2] - λ
    
    // 方法：使用列向量叉乘
    // 取C - λI的前两列，叉乘得到特征向量
    float col0[3] = {C_minus_lambda[0], C_minus_lambda[3], C_minus_lambda[6]}; // 第0列
    float col1[3] = {C_minus_lambda[1], C_minus_lambda[4], C_minus_lambda[7]}; // 第1列
    
    // 叉乘：eigenvector = col0 × col1
    eigenvector[0] = col0[1] * col1[2] - col0[2] * col1[1];
    eigenvector[1] = col0[2] * col1[0] - col0[0] * col1[2];
    eigenvector[2] = col0[0] * col1[1] - col0[1] * col1[0];
    
    // 归一化
    float norm = sqrtf(eigenvector[0]*eigenvector[0] + eigenvector[1]*eigenvector[1] + eigenvector[2]*eigenvector[2]);
    
    if (norm < 1e-6f)
    {
        // 如果叉乘结果为零向量（退化情况），使用默认方向
        eigenvector[0] = 0.0f;
        eigenvector[1] = 0.0f;
        eigenvector[2] = 1.0f;
    }
    else
    {
        float inv_norm = 1.0f / norm;
        eigenvector[0] *= inv_norm;
        eigenvector[1] *= inv_norm;
        eigenvector[2] *= inv_norm;
    }
}

// ========================================
// 6×6方程组求解与Q矩阵变换辅助函数
// ========================================

/**
 * @brief 使用高斯消元法求解6×6线性方程组 M * x = b
 * @param M 6×6设计矩阵（36个float，行主序存储）
 * @param b 6×1目标向量（6个float）
 * @param x [out] 6×1解向量（6个float，存储a,b,c,d,e,f）
 * @return true表示求解成功，false表示矩阵奇异
 */
__device__ bool solve6x6GaussElimination(const float *M, const float *b, float *x)
{
    // 创建增广矩阵 [M | b]，大小为6×7
    float aug[6][7];
    for (int i = 0; i < 6; ++i)
    {
        for (int j = 0; j < 6; ++j)
        {
            aug[i][j] = M[i*6 + j];
        }
        aug[i][6] = b[i];
    }
    
    // 前向消元（高斯消元）
    for (int col = 0; col < 6; ++col)
    {
        // 找主元（列主元）
        int max_row = col;
        float max_val = fabsf(aug[col][col]);
        for (int row = col + 1; row < 6; ++row)
        {
            if (fabsf(aug[row][col]) > max_val)
            {
                max_val = fabsf(aug[row][col]);
                max_row = row;
            }
        }
        
        // 交换行
        if (max_row != col)
        {
            for (int j = col; j < 7; ++j)
            {
                float tmp = aug[col][j];
                aug[col][j] = aug[max_row][j];
                aug[max_row][j] = tmp;
            }
        }
        
        // 检查主元是否为零（奇异矩阵）
        if (fabsf(aug[col][col]) < 1e-6f)
        {
            return false; // 矩阵奇异，求解失败
        }
        
        // 消元
        float pivot = aug[col][col];
        for (int row = col + 1; row < 6; ++row)
        {
            float factor = aug[row][col] / pivot;
            for (int j = col; j < 7; ++j)
            {
                aug[row][j] -= factor * aug[col][j];
            }
        }
    }
    
    // 回代求解
    for (int i = 5; i >= 0; --i)
    {
        x[i] = aug[i][6];
        for (int j = i + 1; j < 6; ++j)
        {
            x[i] -= aug[i][j] * x[j];
        }
        x[i] /= aug[i][i];
    }
    
    return true;
}

/**
 * @brief 构造4×4齐次变换矩阵T = [[R, p], [0, 0, 0, 1]]
 * @param X X轴向量（3个float）
 * @param Y Y轴向量（3个float）
 * @param Z Z轴向量（3个float）
 * @param p 3×1平移向量（质心）
 * @param T [out] 4×4齐次变换矩阵（行主序存储）
 */
__device__ void constructHomogeneousTransform(
    const float *X, const float *Y, const float *Z,
    const GPUPoint3f &p,
    float *T)
{
    // T = [[R, p], [0, 0, 0, 1]]
    // 行主序存储：T[i*4+j] = T(i,j)
    // R = [X | Y | Z]（列主序），所以R的行是X、Y、Z的分量
    
    // 第0行：[X[0], Y[0], Z[0], p.x]
    T[0] = X[0]; T[1] = Y[0]; T[2] = Z[0]; T[3] = p.x;
    // 第1行：[X[1], Y[1], Z[1], p.y]
    T[4] = X[1]; T[5] = Y[1]; T[6] = Z[1]; T[7] = p.y;
    // 第2行：[X[2], Y[2], Z[2], p.z]
    T[8] = X[2]; T[9] = Y[2]; T[10] = Z[2]; T[11] = p.z;
    // 第3行：[0, 0, 0, 1]
    T[12] = 0.0f; T[13] = 0.0f; T[14] = 0.0f; T[15] = 1.0f;
}

/**
 * @brief 计算4×4齐次变换矩阵的逆：T^-1
 * 对于 T = [[R, p], [0, 1]]，有 T^-1 = [[R^T, -R^T*p], [0, 1]]
 * @param T 4×4齐次变换矩阵（行主序）
 * @param T_inv [out] 4×4逆矩阵（行主序）
 */
__device__ void invertHomogeneousTransform(const float *T, float *T_inv)
{
    // 提取R和p
    // T的前3行前3列是R（行主序），第4列是p
    float R[9]; // 3×3旋转矩阵，行主序
    float p[3];
    R[0] = T[0]; R[1] = T[1]; R[2] = T[2];   // 第0行
    R[3] = T[4]; R[4] = T[5]; R[5] = T[6];   // 第1行
    R[6] = T[8]; R[7] = T[9]; R[8] = T[10];  // 第2行
    p[0] = T[3]; p[1] = T[7]; p[2] = T[11];
    
    // R^T（转置）
    float RT[9];
    RT[0] = R[0]; RT[1] = R[3]; RT[2] = R[6]; // 第0行 = R的第0列
    RT[3] = R[1]; RT[4] = R[4]; RT[5] = R[7]; // 第1行 = R的第1列
    RT[6] = R[2]; RT[7] = R[5]; RT[8] = R[8]; // 第2行 = R的第2列
    
    // -R^T * p
    float neg_RT_p[3];
    neg_RT_p[0] = -(RT[0]*p[0] + RT[1]*p[1] + RT[2]*p[2]);
    neg_RT_p[1] = -(RT[3]*p[0] + RT[4]*p[1] + RT[5]*p[2]);
    neg_RT_p[2] = -(RT[6]*p[0] + RT[7]*p[1] + RT[8]*p[2]);
    
    // T^-1 = [[R^T, -R^T*p], [0, 1]]
    T_inv[0] = RT[0]; T_inv[1] = RT[1]; T_inv[2] = RT[2]; T_inv[3] = neg_RT_p[0];
    T_inv[4] = RT[3]; T_inv[5] = RT[4]; T_inv[6] = RT[5]; T_inv[7] = neg_RT_p[1];
    T_inv[8] = RT[6]; T_inv[9] = RT[7]; T_inv[10] = RT[8]; T_inv[11] = neg_RT_p[2];
    T_inv[12] = 0.0f; T_inv[13] = 0.0f; T_inv[14] = 0.0f; T_inv[15] = 1.0f;
}

/**
 * @brief 计算 Qglobal = (T^-1)^T * Qlocal * T^-1
 * @param T_inv 4×4逆变换矩阵（行主序）
 * @param Qlocal 4×4局部Q矩阵（行主序）
 * @param Qglobal [out] 4×4全局Q矩阵（行主序）
 */
__device__ void transformQuadricMatrix(const float *T_inv, const float *Qlocal, float *Qglobal)
{
    // 计算中间结果：Qtemp = Qlocal * T^-1
    float Qtemp[16];
    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            float sum = 0.0f;
            for (int k = 0; k < 4; ++k)
            {
                sum += Qlocal[i*4 + k] * T_inv[k*4 + j];
            }
            Qtemp[i*4 + j] = sum;
        }
    }
    
    // 计算 Qglobal = (T^-1)^T * Qtemp
    // (T^-1)^T的行是T^-1的列
    float T_inv_T[16]; // (T^-1)^T，行主序
    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            T_inv_T[i*4 + j] = T_inv[j*4 + i]; // 转置
        }
    }
    
    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            float sum = 0.0f;
            for (int k = 0; k < 4; ++k)
            {
                sum += T_inv_T[i*4 + k] * Qtemp[k*4 + j];
            }
            Qglobal[i*4 + j] = sum;
        }
    }
}

__global__ void sampleAndBuildMatrices_Kernel(
    const GPUPoint3f *all_points,
    const int *remaining_indices,
    int num_remaining,
    curandState *rand_states,
    int batch_size,
    float *batch_matrices,
    GPUQuadricModel *batch_models,
    float *batch_explicit_coeffs,  // 🆕 输出：显式系数 [batch_size × 6]
    float *batch_transforms)       // 🆕 输出：变换矩阵 [batch_size × 12] (3x4)
{
    int model_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (model_id >= batch_size)
        return;

    curandState local_state = rand_states[model_id];

    // ========================================
    // 锚点驱动的局部采样
    // ========================================

    // 第一步：选取全局锚点（种子点）
    int seed_pos = curand(&local_state) % num_remaining;
    int seed_idx = remaining_indices[seed_pos];
    int sample_indices[6];
    sample_indices[0] = seed_idx;  // 第一个点是锚点

    // 第二步：定义局部搜索窗口
    int range = (int)(0.01f * num_remaining);
    int low = max(0, seed_pos - range);
    int high = min(num_remaining - 1, seed_pos + range);
    int window_size = high - low + 1;

    // 第三步：窗口内伙伴采样（5个点）
    int partner_count = 0;
    int max_attempts = 100;  // 防止无限循环
    int attempts = 0;

    while (partner_count < 5 && attempts < max_attempts)
    {
        attempts++;
        // 在窗口内随机选择一个位置
        int candidate_pos = low + (curand(&local_state) % window_size);
        
        // 强制约束：跳过Z轴方向过度密集的邻近点
        if (abs(candidate_pos - seed_pos) > 3)
        {
            int candidate_idx = remaining_indices[candidate_pos];
            
            // 检查是否重复（简单检查）
            bool is_duplicate = false;
            for (int j = 0; j <= partner_count; ++j)
            {
                if (sample_indices[j] == candidate_idx)
                {
                    is_duplicate = true;
                    break;
                }
            }
            
            if (!is_duplicate)
            {
                partner_count++;
                sample_indices[partner_count] = candidate_idx;
            }
        }
    }

    // 如果未能采样到5个伙伴点，使用全局随机填充
    if (partner_count < 5)
    {
        for (int i = partner_count + 1; i < 6; ++i)
        {
            sample_indices[i] = remaining_indices[curand(&local_state) % num_remaining];
        }
    }

    // ========================================
    // 几何跨度校验
    // ========================================

    // 获取6个点的3D坐标
    GPUPoint3f sampled_points[6];
    for (int i = 0; i < 6; ++i)
    {
        sampled_points[i] = all_points[sample_indices[i]];
    }

    // 计算包围盒
    float min_x = sampled_points[0].x, max_x = sampled_points[0].x;
    float min_y = sampled_points[0].y, max_y = sampled_points[0].y;
    float min_z = sampled_points[0].z, max_z = sampled_points[0].z;

    for (int i = 1; i < 6; ++i)
    {
        min_x = fminf(min_x, sampled_points[i].x);
        max_x = fmaxf(max_x, sampled_points[i].x);
        min_y = fminf(min_y, sampled_points[i].y);
        max_y = fmaxf(max_y, sampled_points[i].y);
        min_z = fminf(min_z, sampled_points[i].z);
        max_z = fmaxf(max_z, sampled_points[i].z);
    }

    float dx = max_x - min_x;
    float dy = max_y - min_y;
    float dz = max_z - min_z;

    // 退化判定：XY平面上几乎共线或过于聚集
    bool is_degenerate = (dx < 0.2f) && (dy < 0.2f);

    // 调试输出：打印前3个模型的采样信息
    if (model_id < 3)
    {
        printf("模型 %d 锚点采样: seed_pos=%d, seed_idx=%d, window=[%d,%d], ", 
               model_id, seed_pos, seed_idx, low, high);
        printf("包围盒长度: dx=%.3f, dy=%.3f, dz=%.3f, ", dx, dy, dz);
        printf("退化=%s\n", is_degenerate ? "是" : "否");
        if (!is_degenerate)
        {
            printf("  采样点: ");
            for (int i = 0; i < 6; ++i)
            {
                printf("点%d=(%.3f,%.3f,%.3f) ", i, 
                       sampled_points[i].x, sampled_points[i].y, sampled_points[i].z);
            }
            printf("\n");
        }
    }

    if (is_degenerate)
    {
        // 设置无效标志：将模型的所有系数设为0，coeffs[15]设为-1.0f表示无效
        GPUQuadricModel *model = &batch_models[model_id];
        for (int i = 0; i < 16; ++i)
        {
            model->coeffs[i] = 0.0f;
        }
        model->coeffs[15] = -1.0f;  // 使用负数作为无效标志
        
        // 恢复随机数状态
        rand_states[model_id] = local_state;
        return;  // 跳过后续PCA和6×6求解
    }

    // ========================================
    // PCA局部坐标系构建
    // ========================================
    
    // 2.1 计算质心p
    GPUPoint3f centroid = {0.0f, 0.0f, 0.0f};
    for (int i = 0; i < 6; ++i)
    {
        GPUPoint3f pt = all_points[sample_indices[i]];
        // 检查无效点
        if (!isfinite(pt.x) || !isfinite(pt.y) || !isfinite(pt.z) ||
            isnan(pt.x) || isnan(pt.y) || isnan(pt.z) ||
            isinf(pt.x) || isinf(pt.y) || isinf(pt.z))
        {
            continue; // 跳过无效点
        }
        centroid.x += pt.x;
        centroid.y += pt.y;
        centroid.z += pt.z;
    }
    centroid.x /= 6.0f;
    centroid.y /= 6.0f;
    centroid.z /= 6.0f;
    
    // 2.2 计算3×3协方差矩阵C
    float C[9] = {0.0f}; // 3×3矩阵，按行主序存储
    int valid_count = 0;
    for (int i = 0; i < 6; ++i)
    {
        GPUPoint3f pt = all_points[sample_indices[i]];
        if (!isfinite(pt.x) || !isfinite(pt.y) || !isfinite(pt.z) ||
            isnan(pt.x) || isnan(pt.y) || isnan(pt.z) ||
            isinf(pt.x) || isinf(pt.y) || isinf(pt.z))
        {
            continue; // 跳过无效点
        }
        valid_count++;
        float dx = pt.x - centroid.x;
        float dy = pt.y - centroid.y;
        float dz = pt.z - centroid.z;
        // C = sum((p - centroid) * (p - centroid)^T)
        C[0] += dx * dx; // C[0,0]
        C[1] += dx * dy; // C[0,1]
        C[2] += dx * dz; // C[0,2]
        C[3] += dy * dx; // C[1,0]
        C[4] += dy * dy; // C[1,1]
        C[5] += dy * dz; // C[1,2]
        C[6] += dz * dx; // C[2,0]
        C[7] += dz * dy; // C[2,1]
        C[8] += dz * dz; // C[2,2]
    }
    
    // 除以(n-1) = 5（如果valid_count < 2，使用valid_count-1）
    float inv_n_minus_1 = (valid_count > 1) ? (1.0f / (valid_count - 1.0f)) : 1.0f;
    for (int i = 0; i < 9; ++i)
    {
        C[i] *= inv_n_minus_1;
    }
    
    // 2.3 使用卡尔丹解析解求特征值
    float eigenvalues[3];
    solveCubicEigenvalues(C, eigenvalues);
    float lambda_min = eigenvalues[0]; // 最小特征值
    
    // 2.4 提取最小特征值对应的特征向量n
    float n[3]; // 归一化的特征向量（作为Z轴）
    computeEigenvector(C, lambda_min, n);
    
    // 2.5 构造正交矩阵R
    // Z轴 = n（最小特征值对应的特征向量）
    float Z[3] = {n[0], n[1], n[2]};
    
    // X轴：选择与Z轴垂直的向量
    float X[3];
    if (fabsf(Z[0]) < 0.9f)
    {
        // 使用[1,0,0]投影到垂直于Z的平面
        float dot = Z[0];
        X[0] = 1.0f - dot * Z[0];
        X[1] = -dot * Z[1];
        X[2] = -dot * Z[2];
    }
    else
    {
        // 使用[0,1,0]投影
        float dot = Z[1];
        X[0] = -dot * Z[0];
        X[1] = 1.0f - dot * Z[1];
        X[2] = -dot * Z[2];
    }
    // 归一化X
    float norm_x = sqrtf(X[0]*X[0] + X[1]*X[1] + X[2]*X[2]);
    if (norm_x > 1e-6f)
    {
        float inv_norm_x = 1.0f / norm_x;
        X[0] *= inv_norm_x;
        X[1] *= inv_norm_x;
        X[2] *= inv_norm_x;
    }
    else
    {
        // 退化情况：使用默认X轴
        X[0] = 1.0f;
        X[1] = 0.0f;
        X[2] = 0.0f;
    }
    
    // Y轴 = Z × X（叉乘）
    float Y[3];
    Y[0] = Z[1] * X[2] - Z[2] * X[1];
    Y[1] = Z[2] * X[0] - Z[0] * X[2];
    Y[2] = Z[0] * X[1] - Z[1] * X[0];
    // 归一化Y
    float norm_y = sqrtf(Y[0]*Y[0] + Y[1]*Y[1] + Y[2]*Y[2]);
    if (norm_y > 1e-6f)
    {
        float inv_norm_y = 1.0f / norm_y;
        Y[0] *= inv_norm_y;
        Y[1] *= inv_norm_y;
        Y[2] *= inv_norm_y;
    }
    
    // 旋转矩阵R = [X | Y | Z]（3×3，列主序）
    // 注意：R和p作为局部变量，不写入全局内存
    
    // ========================================
    // 第二阶段：坐标系对齐与6×6方程组求解
    // ========================================
    
    // 1. 坐标系对齐：将6个全局点映射到局部坐标系 Plocal = R^T(Pglobal - p)
    GPUPoint3f local_points[6];
    for (int i = 0; i < 6; ++i)
    {
        GPUPoint3f pt_global = all_points[sample_indices[i]];
        // P - p
        float dx = pt_global.x - centroid.x;
        float dy = pt_global.y - centroid.y;
        float dz = pt_global.z - centroid.z;
        // R^T * (P - p)，其中R = [X | Y | Z]（列主序）
        // R^T的行是X、Y、Z（转置后）
        local_points[i].x = X[0]*dx + X[1]*dy + X[2]*dz;  // X^T * (P-p)
        local_points[i].y = Y[0]*dx + Y[1]*dy + Y[2]*dz;  // Y^T * (P-p)
        local_points[i].z = Z[0]*dx + Z[1]*dy + Z[2]*dz;  // Z^T * (P-p)
    }
    
    // 2. 构建6×6设计矩阵M：每一行是 [xi², xiyi, yi², xi, yi, 1]
    float M[36]; // 6×6矩阵，行主序
    float z_vec[6]; // 目标向量：6个点的局部z坐标
    
    for (int i = 0; i < 6; ++i)
    {
        float x = local_points[i].x;
        float y = local_points[i].y;
        float z = local_points[i].z;
        
        // 第i行
        M[i*6 + 0] = x * x;      // xi²
        M[i*6 + 1] = x * y;      // xiyi
        M[i*6 + 2] = y * y;      // yi²
        M[i*6 + 3] = x;          // xi
        M[i*6 + 4] = y;          // yi
        M[i*6 + 5] = 1.0f;       // 1
        
        z_vec[i] = z;            // 目标值
    }
    
    // 3. 求解6×6方程组：M * [a, b, c, d, e, f]^T = z_vec
    float coeffs_local[6]; // [a, b, c, d, e, f]
    bool solve_success = solve6x6GaussElimination(M, z_vec, coeffs_local);
    
    // 4. 构造Qlocal并转换到Qglobal
    GPUQuadricModel *model = &batch_models[model_id];
    
    if (!solve_success)
    {
        // 求解失败，使用占位Q矩阵（z=0平面）
        // z=0平面对应的10维系数：[0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
        // 映射到4×4对称矩阵Q（行主序存储到coeffs[16]）
        model->coeffs[0] = 0.0f;   // Q(0,0) = A = 0
        model->coeffs[1] = 0.0f;   // Q(0,1) = D = 0
        model->coeffs[2] = 0.0f;   // Q(0,2) = E = 0
        model->coeffs[3] = 0.0f;   // Q(0,3) = G = 0
        model->coeffs[4] = 0.0f;   // Q(1,0) = D = 0
        model->coeffs[5] = 0.0f;   // Q(1,1) = B = 0
        model->coeffs[6] = 0.0f;   // Q(1,2) = F = 0
        model->coeffs[7] = 0.0f;   // Q(1,3) = H = 0
        model->coeffs[8] = 0.0f;   // Q(2,0) = E = 0
        model->coeffs[9] = 0.0f;   // Q(2,1) = F = 0
        model->coeffs[10] = 0.0f;  // Q(2,2) = 0 【修正：z²项系数为0】
        model->coeffs[11] = 0.5f;  // Q(2,3) = 0.5 【修正：z项系数为1】
        model->coeffs[12] = 0.0f;  // Q(3,0) = G = 0
        model->coeffs[13] = 0.0f;  // Q(3,1) = H = 0
        model->coeffs[14] = 0.0f;  // Q(3,2) = I = 0
        model->coeffs[15] = 0.0f;  // Q(3,3) = J = 0
        
        // 🆕 求解失败时，填充零值（占位Q矩阵）
        for (int i = 0; i < 6; ++i) {
            batch_explicit_coeffs[model_id * 6 + i] = 0.0f;
        }
        for (int i = 0; i < 12; ++i) {
            batch_transforms[model_id * 12 + i] = 0.0f;
        }
    }
    else
    {
        // 构造Qlocal矩阵（4×4，行主序存储）
        // Qlocal = [[-a, -b/2, 0, -d/2],
        //           [-b/2, -c, 0, -e/2],
        //           [0, 0, 0.5, 0],
        //           [-d/2, -e/2, 0, -f]]
        float Qlocal[16];
        float a = coeffs_local[0];
        float b = coeffs_local[1];
        float c = coeffs_local[2];
        float d = coeffs_local[3];
        float e = coeffs_local[4];
        float f = coeffs_local[5];
        
        Qlocal[0] = -a;        // Q(0,0)
        Qlocal[1] = -b * 0.5f; // Q(0,1)
        Qlocal[2] = 0.0f;      // Q(0,2)
        Qlocal[3] = -d * 0.5f; // Q(0,3)
        Qlocal[4] = -b * 0.5f; // Q(1,0) = Q(0,1)
        Qlocal[5] = -c;        // Q(1,1)
        Qlocal[6] = 0.0f;      // Q(1,2)
        Qlocal[7] = -e * 0.5f; // Q(1,3)
        Qlocal[8] = 0.0f;      // Q(2,0)
        Qlocal[9] = 0.0f;      // Q(2,1)
        Qlocal[10] = 0.0f;     // Q(2,2) = 0 【修正：z²项系数为0】
        Qlocal[11] = 0.5f;     // Q(2,3) = 0.5 【修正：z项系数为1】
        Qlocal[12] = -d * 0.5f; // Q(3,0) = Q(0,3)
        Qlocal[13] = -e * 0.5f; // Q(3,1) = Q(1,3)
        Qlocal[14] = 0.5f;      // Q(3,2) = 0.5 【修正：z项的一半，与Q(2,3)对称】
        Qlocal[15] = -f;        // Q(3,3)
        
        // 构造齐次变换矩阵T
        float T[16];
        constructHomogeneousTransform(X, Y, Z, centroid, T);
        
        // 计算T^-1
        float T_inv[16];
        invertHomogeneousTransform(T, T_inv);
        
        // 计算Qglobal = (T^-1)^T * Qlocal * T^-1
        float Qglobal[16];
        transformQuadricMatrix(T_inv, Qlocal, Qglobal);
        
        // 写入到GPUQuadricModel（行主序）
        for (int i = 0; i < 16; ++i)
        {
            model->coeffs[i] = Qglobal[i];
        }
        
        // 🆕 保存显式系数到GPU缓冲区
        for (int i = 0; i < 6; ++i) {
            batch_explicit_coeffs[model_id * 6 + i] = coeffs_local[i];
        }
        
        // 🆕 保存变换矩阵T的前3行（3x4，即[R|p]）到GPU缓冲区
        // T的前3行是 [R | p]，行主序存储
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 4; ++j) {
                batch_transforms[model_id * 12 + i * 4 + j] = T[i * 4 + j];
            }
        }
    }

    rand_states[model_id] = local_state;
}

__global__ void countInliersBatch_Kernel(
    const GPUPoint3f *all_points,
    const int *remaining_indices,
    int num_remaining,
    const uint8_t *valid_mask,
    const GPUQuadricModel *batch_models,
    int batch_size,
    float threshold,
    int stride,
    int *batch_inlier_counts)
{
    int model_id = blockIdx.y;
    if (model_id >= batch_size)
        return;

    // Shared Memory 优化：缓存当前 Block 对应的二次曲面模型参数
    __shared__ GPUQuadricModel shared_model;
    
    // 协作式加载：threadIdx.x == 0 的线程负责从 Global Memory 加载模型
    if (threadIdx.x == 0)
    {
        shared_model = batch_models[model_id];
    }
    __syncthreads();  // 确保所有线程等待模型加载完成

    // 优化：每个线程直接处理一个采样点，不再使用循环
    int sampled_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int local_count = 0;

    // 为了避免走样问题，使用基于model_id和sampled_idx的伪随机偏移
    int random_offset = (model_id * 17 + sampled_idx) % stride;
    
    // 直接计算原始点索引（基于采样索引）
    int i = sampled_idx * stride + random_offset;
    
    // 边界检查：确保不越界
    if (i < num_remaining)
    {
        int global_idx = remaining_indices[i];
        if (valid_mask[global_idx] == 0) return;  // 跳过已移除的点
        
        GPUPoint3f point = all_points[global_idx];
        // 使用 shared memory 中的模型，而不是从 Global Memory 读取
        float dist = evaluateQuadricDistance(point, shared_model);

        if (dist < threshold)
        {
            local_count = 1;  // 单个点，直接设为1
        }
    }

    // Block内reduce求和
    __shared__ int shared_counts[256];
    shared_counts[threadIdx.x] = local_count;
    __syncthreads();

    for (int reduce_stride = blockDim.x / 2; reduce_stride > 0; reduce_stride >>= 1)
    {
        if (threadIdx.x < reduce_stride)
        {
            shared_counts[threadIdx.x] += shared_counts[threadIdx.x + reduce_stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        atomicAdd(&batch_inlier_counts[model_id], shared_counts[0]);
    }
}

// 精选阶段内点计数内核 - 对Top-K模型全量计数
__global__ void fineCountInliers_Kernel(
    const GPUPoint3f *all_points,
    const int *remaining_indices,
    int num_remaining,
    const uint8_t *valid_mask,
    const GPUQuadricModel *candidate_models,
    const int *candidate_indices,
    int k,
    float threshold,
    int *fine_inlier_counts)
{
    int candidate_id = blockIdx.y;
    if (candidate_id >= k)
        return;

    // Shared Memory 优化：缓存当前 Block 对应的候选模型参数
    __shared__ GPUQuadricModel shared_model;
    
    // 协作式加载：threadIdx.x == 0 的线程负责从 Global Memory 加载模型
    if (threadIdx.x == 0)
    {
        shared_model = candidate_models[candidate_id];
    }
    __syncthreads();  // 确保所有线程等待模型加载完成

    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int local_count = 0;

    // 全量计数（stride=1），对每个候选模型执行100%点云验证
    for (int i = thread_id; i < num_remaining; i += blockDim.x * gridDim.x)
    {
        int global_idx = remaining_indices[i];
        if (valid_mask[global_idx] == 0) continue;  // 跳过已移除的点
        
        GPUPoint3f point = all_points[global_idx];
        // 使用 shared memory 中的模型，而不是从 Global Memory 读取
        float dist = evaluateQuadricDistance(point, shared_model);

        if (dist < threshold)
        {
            local_count++;
        }
    }

    // Block内reduce求和
    __shared__ int shared_counts[256];
    shared_counts[threadIdx.x] = local_count;
    __syncthreads();

    for (int reduce_stride = blockDim.x / 2; reduce_stride > 0; reduce_stride >>= 1)
    {
        if (threadIdx.x < reduce_stride)
        {
            shared_counts[threadIdx.x] += shared_counts[threadIdx.x + reduce_stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        atomicAdd(&fine_inlier_counts[candidate_id], shared_counts[0]);
    }
}

__device__ __forceinline__ float evaluateQuadricDistance(
    const GPUPoint3f &point,
    const GPUQuadricModel &model)
{
    float x = point.x, y = point.y, z = point.z;

    // 手动展开4x4矩阵乘法：[x y z 1] * Q * [x y z 1]^T
    // Q矩阵按行主序存储：coeffs[i*4+j] = Q(i,j)
    // 确保所有16个系数都参与计算，与数学定义完全一致
    float result = 
        x * x * model.coeffs[0] + x * y * model.coeffs[1] + x * z * model.coeffs[2] + x * model.coeffs[3] +
        y * x * model.coeffs[4] + y * y * model.coeffs[5] + y * z * model.coeffs[6] + y * model.coeffs[7] +
        z * x * model.coeffs[8] + z * y * model.coeffs[9] + z * z * model.coeffs[10] + z * model.coeffs[11] +
        x * model.coeffs[12] + y * model.coeffs[13] + z * model.coeffs[14] + model.coeffs[15];

    // 只保留最终结果检查作为兜底
    if (!isfinite(result) || isnan(result) || isinf(result))
    {
        return 1e10f; // 返回一个很大的距离，表示计算失败
    }

    return fabsf(result);
}


__global__ void extractInliers_Kernel(
    const GPUPoint3f *all_points,
    const int *remaining_indices,
    int num_remaining,
    const GPUQuadricModel *model,
    float threshold,
    int *inlier_indices,
    int *inlier_count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_remaining)
        return;

    //  修复开始：添加更多安全检查
    // 检查输入参数有效性
    if (all_points == nullptr || remaining_indices == nullptr ||
        model == nullptr || inlier_indices == nullptr || inlier_count == nullptr)
    {
        return; // 静默返回，避免在GPU上打印错误
    }

    // 检查索引边界
    int global_point_index = remaining_indices[idx];
    if (global_point_index < 0)
    {
        return; // 无效的点索引
    }

        //  关键修复：确保我们不访问超出all_points数组边界的内存
    // 注意：我们无法在GPU内核中直接获取all_points的大小，所以需要依赖调用方确保索引有效

    GPUPoint3f point = all_points[global_point_index];

    //  验证点的有效性
    if (!isfinite(point.x) || !isfinite(point.y) || !isfinite(point.z) ||
        isnan(point.x) || isnan(point.y) || isnan(point.z) ||
        isinf(point.x) || isinf(point.y) || isinf(point.z))
    {
        return; // 跳过无效点
    }

    float dist = evaluateQuadricDistance(point, *model);

    //  验证距离计算结果的有效性
    if (!isfinite(dist) || isnan(dist) || isinf(dist))
    {
        return; // 跳过无效距离计算结果
    }
    //  修复结束

    if (dist < threshold)
    {
        //  修复开始：添加边界检查防止数组越界
        int write_pos = atomicAdd(inlier_count, 1);

        //  关键安全检查：确保不会越界访问
        // 理论上 d_temp_inlier_indices_ 大小等于 d_remaining_indices_.size()
        // 所以 write_pos 应该永远 < num_remaining，但为了安全还是检查
        if (write_pos < num_remaining)
        {
            inlier_indices[write_pos] = global_point_index;
        }
        else
        {
            //  如果发生越界，至少不会崩溃，但会丢失这个内点
            // 在实际应用中这种情况不应该发生
            atomicAdd(inlier_count, -1); // 回滚计数器
        }
        //  修复结束
    }
} // ========================================
// 成员函数实现 (每个函数只定义一次!)
// ========================================

// 初始化序列内核实现
__global__ void initSequenceKernel(int *indices, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        indices[idx] = idx;
    }
}

// Bitonic Sort内核实现 - 高性能Top-K选择（降序排序）
__global__ void bitonicSort1024Kernel(
    int *inlier_counts,
    int *model_indices,
    int n)
{
    // Shared Memory：存储计数和索引（需要512线程，每个线程处理2个元素）
    __shared__ int s_counts[1024];
    __shared__ int s_indices[1024];
    
    int tid = threadIdx.x;
    
    // 加载数据到shared memory（512线程，每个线程加载2个元素）
    if (tid < 512)
    {
        s_counts[tid * 2] = inlier_counts[tid * 2];
        s_indices[tid * 2] = model_indices[tid * 2];
        if (tid * 2 + 1 < n)
        {
            s_counts[tid * 2 + 1] = inlier_counts[tid * 2 + 1];
            s_indices[tid * 2 + 1] = model_indices[tid * 2 + 1];
        }
    }
    __syncthreads();
    
    // Bitonic Sort（降序排序，最大值在索引0）
    // 外层循环：构建bitonic序列的大小（2, 4, 8, ..., 1024）
    for (int k = 2; k <= n; k <<= 1)
    {
        // 内层循环：比较和交换的步长（k/2, k/4, ..., 1）
        for (int j = k >> 1; j > 0; j >>= 1)
        {
            // 每个线程处理一个元素对
            int i = tid;
            int ixj = i ^ j;  // 要比较的另一个索引
            
            if (i < n && ixj < n && i < ixj)
            {
                // 对于降序排序（最大值在索引0）：
                // 无论元素在哪个段，都应该让较大的值上移（较小的值下移）
                // 统一使用 s_counts[i] < s_counts[ixj] 作为交换条件
                if (s_counts[i] < s_counts[ixj])
                {
                    // swap
                    int tmp_c = s_counts[i];
                    int tmp_i = s_indices[i];
                    s_counts[i] = s_counts[ixj];
                    s_indices[i] = s_indices[ixj];
                    s_counts[ixj] = tmp_c;
                    s_indices[ixj] = tmp_i;
                }
            }
            __syncthreads();
        }
    }
    
    // 写回结果
    if (tid < 512)
    {
        inlier_counts[tid * 2] = s_counts[tid * 2];
        model_indices[tid * 2] = s_indices[tid * 2];
        if (tid * 2 + 1 < n)
        {
            inlier_counts[tid * 2 + 1] = s_counts[tid * 2 + 1];
            model_indices[tid * 2 + 1] = s_indices[tid * 2 + 1];
        }
    }
}

// 辅助函数：获取点云指针（支持外部内存）
GPUPoint3f* QuadricDetect::getPointsPtr() const
{
    if (is_external_memory_)
    {
        return d_external_points_;
    }
    else
    {
        return const_cast<GPUPoint3f*>(thrust::raw_pointer_cast(d_all_points_.data()));
    }
}

void QuadricDetect::initializeGPUMemory(int batch_size)
{
    // 分配GPU内存
    d_batch_matrices_.resize(batch_size * 9 * 10);
    d_batch_models_.resize(batch_size);
    d_batch_inlier_counts_.resize(batch_size);
    d_rand_states_.resize(batch_size * 10);
    d_batch_explicit_coeffs_.resize(batch_size * 6);  // 🆕 显式系数 [batch_size × 6]
    d_batch_transforms_.resize(batch_size * 12);     // 🆕 变换矩阵 [batch_size × 12] (3x4)

    // 初始化结果存储
    d_best_model_index_.resize(1);
    d_best_model_count_.resize(1);
    
    // 修复：初始化为0，确保有有效的初始值
    thrust::fill(thrust::cuda::par.on(stream_), d_best_model_index_.begin(), d_best_model_index_.end(), 0);
    thrust::fill(thrust::cuda::par.on(stream_), d_best_model_count_.begin(), d_best_model_count_.end(), 0);

    //  添加反幂迭代相关
    d_batch_ATA_matrices_.resize(batch_size * 10 * 10);
    d_batch_R_matrices_.resize(batch_size * 10 * 10);
    d_batch_eigenvectors_.resize(batch_size * 10);
    
    // 预分配掩码缓冲区（使用原始总点数，确保掩码大小固定）
    if (original_total_count_ > 0)
    {
        if (d_valid_mask_ == nullptr || original_total_count_ > max_points_capacity_)
        {
            // 释放旧缓冲区（如果存在）
            if (d_valid_mask_ != nullptr)
            {
                cudaFree(d_valid_mask_);
                d_valid_mask_ = nullptr;
            }
            
            // 分配新的掩码缓冲区（使用原始总点数）
            cudaError_t err = cudaMalloc((void**)&d_valid_mask_, original_total_count_ * sizeof(uint8_t));
            if (err != cudaSuccess)
            {
                std::cerr << "[initializeGPUMemory] 错误：无法分配掩码缓冲区: " 
                          << cudaGetErrorString(err) << std::endl;
                d_valid_mask_ = nullptr;
                max_points_capacity_ = 0;
            }
            else
            {
                max_points_capacity_ = original_total_count_;
                if (params_.verbosity > 1)
                {
                    std::cout << "[initializeGPUMemory] 分配掩码缓冲区: " << original_total_count_ 
                              << " 点 (" << (original_total_count_ * sizeof(uint8_t) / 1024.0 / 1024.0) 
                              << " MB)" << std::endl;
                }
            }
        }
    }
    
    // 预分配两阶段RANSAC竞速相关内存（避免循环内resize）
    const int max_k = 128;  // 足够处理大部分k值（默认20）
    if (d_indices_full_.size() < static_cast<size_t>(batch_size))
    {
        d_indices_full_.resize(batch_size);
    }
    if (d_top_k_indices_.size() < max_k)
    {
        d_top_k_indices_.resize(max_k);
    }
    if (d_fine_inlier_counts_.size() < max_k)
    {
        d_fine_inlier_counts_.resize(max_k);
    }
    if (d_candidate_models_.size() < max_k)
    {
        d_candidate_models_.resize(max_k);
    }
}

void QuadricDetect::initializeRemainingIndices(size_t count)
{
    if (count > 0) {
        d_remaining_indices_.resize(count);
        // 使用 kernel 初始化序列（因为 thrust::sequence 不支持流绑定）
        dim3 block(256);
        dim3 grid((count + block.x - 1) / block.x);
        initSequenceKernel<<<grid, block, 0, stream_>>>(
            thrust::raw_pointer_cast(d_remaining_indices_.data()),
            static_cast<int>(count));
    }
    else
    {
        d_remaining_indices_.clear();
    }
}

void QuadricDetect::uploadPointsToGPU(const std::vector<GPUPoint3f> &h_points)
{
    //  关键修复：强制清空旧数据，防止多帧复用时的内存污染
    d_all_points_.clear();
    d_remaining_indices_.clear();

    // 重新上传新数据
    d_all_points_ = h_points;
    d_remaining_indices_.resize(h_points.size());
    thrust::sequence(d_remaining_indices_.begin(), d_remaining_indices_.end(), 0);
}

void QuadricDetect::launchInitCurandStates(int batch_size)
{
    dim3 block(128);  // 修改：从 256 改为 128，匹配 __launch_bounds__
    dim3 grid((batch_size * 10 + block.x - 1) / block.x);

    // 加入 clock() 增加随机性
    unsigned long long base_seed = (unsigned long long)time(nullptr) ^ (unsigned long long)clock();

    // 检查显存地址：打印初始化时的地址
    g_init_rand_states_addr = thrust::raw_pointer_cast(d_rand_states_.data());
    std::cout << "[launchInitCurandStates] 初始化时 d_rand_states_ 地址: " << g_init_rand_states_addr << std::endl;

    initCurandStates_Kernel<<<grid, block, 0, stream_>>>( 
        thrust::raw_pointer_cast(d_rand_states_.data()),
        base_seed,
        batch_size * 10);
    
    // 检查内核是否启动成功：如果 err 不是 cudaSuccess，那就破案了
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "!!! FATAL: initCurandStates_Kernel failed: " 
                  << cudaGetErrorString(err) << std::endl;
    } else {
        std::cout << "[launchInitCurandStates] 内核启动成功" << std::endl;
    }
    
    // 强制同步并检查
    cudaError_t sync_err = cudaStreamSynchronize(stream_);
    if (sync_err != cudaSuccess) {
        std::cerr << "!!! FATAL: initCurandStates_Kernel sync failed: " 
                  << cudaGetErrorString(sync_err) << std::endl;
    } else {
        std::cout << "[launchInitCurandStates] 内核同步成功" << std::endl;
    }
}

void QuadricDetect::launchSampleAndBuildMatrices(int batch_size)
{
    // 核心修复：定义 total_points（支持零拷贝模式）
    size_t total_points = is_external_memory_ ? d_remaining_indices_.size() : d_all_points_.size();
    
    // 获取初始化时的地址用于比较（从 launchInitCurandStates 中保存的静态变量）
    extern void* g_init_rand_states_addr;  // 声明外部静态变量

    if (params_.verbosity > 0)
    {
        std::cout << "[launchSampleAndBuildMatrices] 开始生成批量矩阵，batch_size=" << batch_size << std::endl;
        std::cout << "  - 剩余点数: " << d_remaining_indices_.size() << std::endl;
        std::cout << "  - 总点数: " << total_points << std::endl;
    }

    //  验证输入数据
    if (d_remaining_indices_.size() < 9)
    {
        std::cerr << "[launchSampleAndBuildMatrices]  错误：剩余点数不足9个，无法生成矩阵！" << std::endl;
        return;
    }

    if (total_points == 0)
    {
        std::cerr << "[launchSampleAndBuildMatrices]  错误：点云数据为空！" << std::endl;
        return;
    }

            //  新增：验证点云数据的有效性
    if (params_.verbosity > 1)
    {
        std::cout << "[launchSampleAndBuildMatrices]  验证输入点云数据有效性..." << std::endl;

        // 检查前几个点的数据
        thrust::host_vector<GPUPoint3f> h_sample_points(std::min(10, static_cast<int>(total_points)));
        cudaMemcpy(h_sample_points.data(),
                   getPointsPtr(),
                   h_sample_points.size() * sizeof(GPUPoint3f),
                   cudaMemcpyDeviceToHost);

        // 输出前3个点的xyz坐标
        std::cout << "    前3个点的坐标:" << std::endl;
        size_t num_points_to_show = std::min(static_cast<size_t>(3), h_sample_points.size());
        for (size_t i = 0; i < num_points_to_show; ++i)
        {
            const GPUPoint3f &pt = h_sample_points[i];
            std::cout << "      点[" << i << "]: (" << pt.x << ", " << pt.y << ", " << pt.z << ")" << std::endl;
        }

        int invalid_points = 0;
        for (size_t i = 0; i < h_sample_points.size(); ++i)
        {
            const GPUPoint3f &pt = h_sample_points[i];
            if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z) ||
                std::isnan(pt.x) || std::isnan(pt.y) || std::isnan(pt.z) ||
                std::isinf(pt.x) || std::isinf(pt.y) || std::isinf(pt.z))
            {
                invalid_points++;
                std::cout << "    发现无效点[" << i << "]: ("
                          << pt.x << ", " << pt.y << ", " << pt.z << ")" << std::endl;
            }
        }

        if (invalid_points > 0)
        {
            std::cout << "    警告：输入点云包含 " << invalid_points << " 个无效点！" << std::endl;
            std::cout << "    这可能导致SVD计算失败，建议预处理点云数据" << std::endl;
        }
        else
        {
            std::cout << "    ✓ 输入点云数据有效" << std::endl;
        }
    }

    dim3 block(256);
    dim3 grid((batch_size + block.x - 1) / block.x);

    // 检查显存地址：打印采样时的地址
    void* sample_addr = thrust::raw_pointer_cast(d_rand_states_.data());
    std::cout << "[launchSampleAndBuildMatrices] 采样时 d_rand_states_ 地址: " << sample_addr << std::endl;
    
    // 从 launchInitCurandStates 获取初始化时的地址（通过静态变量）
    // 注意：这里需要确保 launchInitCurandStates 已经调用过
    // 如果地址不同，说明 Thrust 在中间偷偷搬了家
    if (g_init_rand_states_addr != nullptr && sample_addr != g_init_rand_states_addr) {
        std::cerr << "!!! WARNING: d_rand_states_ 地址已改变！初始化时: " << g_init_rand_states_addr 
                  << ", 采样时: " << sample_addr << std::endl;
    }

    //  先清零矩阵数据，确保没有垃圾数据
    thrust::fill(thrust::cuda::par.on(stream_), d_batch_matrices_.begin(), d_batch_matrices_.end(), 0.0f);

    sampleAndBuildMatrices_Kernel<<<grid, block, 0, stream_>>>(
        getPointsPtr(),
        thrust::raw_pointer_cast(d_remaining_indices_.data()),
        static_cast<int>(d_remaining_indices_.size()),
        thrust::raw_pointer_cast(d_rand_states_.data()),
        batch_size,
        thrust::raw_pointer_cast(d_batch_matrices_.data()),
        thrust::raw_pointer_cast(d_batch_models_.data()),
        thrust::raw_pointer_cast(d_batch_explicit_coeffs_.data()),  // 🆕 显式系数缓冲区
        thrust::raw_pointer_cast(d_batch_transforms_.data()));       // 🆕 变换矩阵缓冲区

    cudaError_t kernel_error = cudaGetLastError();
    if (kernel_error != cudaSuccess)
    {
        std::cerr << "[launchSampleAndBuildMatrices]  内核启动错误: " << cudaGetErrorString(kernel_error) << std::endl;
        return;
    }

    cudaStreamSynchronize(stream_);

    cudaError_t sync_error = cudaGetLastError();
    if (sync_error != cudaSuccess)
    {
        std::cerr << "[launchSampleAndBuildMatrices]  内核执行错误: " << cudaGetErrorString(sync_error) << std::endl;
        return;
    }

    // 调试信息：验证随机数状态是否不同
    if (params_.verbosity > 1)
    {
        std::cout << "[launchSampleAndBuildMatrices] 验证随机数状态..." << std::endl;
        
        // 创建临时设备内存存储随机数输出
        thrust::device_vector<unsigned int> d_rand_outputs(std::min(3, batch_size) * 3);
        dim3 debug_block(256);
        dim3 debug_grid((std::min(3, batch_size) + debug_block.x - 1) / debug_block.x);
        
        debugRandomStates_Kernel<<<debug_grid, debug_block, 0, stream_>>>(
            thrust::raw_pointer_cast(d_rand_states_.data()),
            std::min(3, batch_size),
            thrust::raw_pointer_cast(d_rand_outputs.data()));
        cudaStreamSynchronize(stream_);
        cudaStreamSynchronize(stream_);
        
        // 读取并打印
        thrust::host_vector<unsigned int> h_rand_outputs = d_rand_outputs;
        std::cout << "  前3个模型的随机数状态验证（每个状态生成3个随机数）:" << std::endl;
        for (int model_id = 0; model_id < 3 && model_id < batch_size; ++model_id)
        {
            std::cout << "    模型 " << model_id << " (rand_states[" << model_id << "]): "
                      << h_rand_outputs[model_id * 3 + 0] << ", "
                      << h_rand_outputs[model_id * 3 + 1] << ", "
                      << h_rand_outputs[model_id * 3 + 2] << std::endl;
        }
    }
    
    // 调试信息：采样点已在内核中通过printf输出（仅前3个模型）
    if (params_.verbosity > 1)
    {
        std::cout << "[launchSampleAndBuildMatrices] 采样点调试信息已在内核中输出（仅前3个模型）" << std::endl;
    }

    //  验证生成的模型数据
    if (params_.verbosity > 1)
    {
        std::cout << "[launchSampleAndBuildMatrices] 验证生成的模型..." << std::endl;

        // 检查前3个模型的系数
        thrust::host_vector<GPUQuadricModel> h_models(3);
        cudaMemcpy(h_models.data(),
                   thrust::raw_pointer_cast(d_batch_models_.data()),
                   3 * sizeof(GPUQuadricModel),
                   cudaMemcpyDeviceToHost);

        bool all_zero = true;
        for (int model_id = 0; model_id < 3 && model_id < batch_size; ++model_id)
        {
            for (int i = 0; i < 16; ++i)
            {
                if (fabsf(h_models[model_id].coeffs[i]) > 1e-6f)
                {
                    all_zero = false;
                    break;
                }
            }
            if (!all_zero) break;
        }

        if (all_zero)
        {
            std::cerr << "[launchSampleAndBuildMatrices]  生成的模型全为零！检查内核实现" << std::endl;

            //  检查输入点云数据
            thrust::host_vector<GPUPoint3f> h_points_sample(std::min(10, static_cast<int>(total_points)));
            cudaMemcpy(h_points_sample.data(),
                       getPointsPtr(),
                       h_points_sample.size() * sizeof(GPUPoint3f),
                       cudaMemcpyDeviceToHost);

            std::cout << "  - 前几个点云数据样本:" << std::endl;
            for (size_t i = 0; i < h_points_sample.size(); ++i)
            {
                std::cout << "    点" << i << ": (" << h_points_sample[i].x
                          << ", " << h_points_sample[i].y
                          << ", " << h_points_sample[i].z << ")" << std::endl;
            }

            //  检查剩余索引
            thrust::host_vector<int> h_indices_sample(std::min(10, (int)d_remaining_indices_.size()));
            cudaMemcpy(h_indices_sample.data(),
                       thrust::raw_pointer_cast(d_remaining_indices_.data()),
                       h_indices_sample.size() * sizeof(int),
                       cudaMemcpyDeviceToHost);

            std::cout << "  - 前几个剩余索引:" << std::endl;
            for (size_t i = 0; i < h_indices_sample.size(); ++i)
            {
                std::cout << "    索引" << i << ": " << h_indices_sample[i] << std::endl;
            }
        }
        else
        {
            std::cout << "[launchSampleAndBuildMatrices] ✓ 模型验证通过，前3个模型的系数非零" << std::endl;
        }
    }

    if (params_.verbosity > 0)
    {
        std::cout << "[launchSampleAndBuildMatrices] 矩阵生成完成" << std::endl;
    }
}

void QuadricDetect::launchCountInliersBatch(int batch_size, int stride)
{
    // 优化：Grid基于采样点数计算，而不是全量点数
    dim3 block(256);
    int num_remaining = static_cast<int>(d_remaining_indices_.size());
    int sampled_points = (num_remaining + stride - 1) / stride;  // 向上取整
    dim3 grid_x((sampled_points + block.x - 1) / block.x);
    dim3 grid(grid_x.x, batch_size); // 2D grid: (sampled_points, models)

    // 先清零计数器
    thrust::fill(thrust::cuda::par.on(stream_), d_batch_inlier_counts_.begin(), d_batch_inlier_counts_.end(), 0);

    countInliersBatch_Kernel<<<grid, block, 0, stream_>>>(
        getPointsPtr(),
        thrust::raw_pointer_cast(d_remaining_indices_.data()),
        num_remaining,
        d_valid_mask_,  // 新增：传递掩码
        thrust::raw_pointer_cast(d_batch_models_.data()),
        batch_size,
        static_cast<float>(params_.quadric_distance_threshold),
        stride,  // 新增：传递采样步长
        thrust::raw_pointer_cast(d_batch_inlier_counts_.data()));
    
    // 检查 CUDA 错误
    cudaError_t kernel_error = cudaGetLastError();
    if (kernel_error != cudaSuccess)
    {
        std::cerr << "[launchCountInliersBatch] 内核启动错误: " << cudaGetErrorString(kernel_error) << std::endl;
    }
    
    // 移除同步：让后续操作异步执行（参考平面检测）
    // cudaStreamSynchronize(stream_);  // 注释掉，让Top-K选择异步执行
}

void QuadricDetect::launchSelectTopKModels(int k)
{
    // 确保预分配的内存足够大
    int batch_size = static_cast<int>(d_batch_inlier_counts_.size());
    
    if (d_indices_full_.size() < static_cast<size_t>(batch_size))
    {
        std::cerr << "[launchSelectTopKModels] 错误：d_indices_full_ 预分配内存不足，需要 " << batch_size 
                  << " 但只有 " << d_indices_full_.size() << std::endl;
        return;
    }
    if (d_top_k_indices_.size() < static_cast<size_t>(k))
    {
        std::cerr << "[launchSelectTopKModels] 错误：d_top_k_indices_ 预分配内存不足，需要 " << k 
                  << " 但只有 " << d_top_k_indices_.size() << std::endl;
        return;
    }
    
    // 使用预分配的 d_indices_full_ 创建索引序列 [0, 1, 2, ..., batch_size-1]
    dim3 block(256);
    dim3 grid((batch_size + block.x - 1) / block.x);
    initSequenceKernel<<<grid, block, 0, stream_>>>(thrust::raw_pointer_cast(d_indices_full_.data()), batch_size);

    // 优化：使用Bitonic Sort替代thrust::sort_by_key（仅当batch_size==1024时）
    if (batch_size == 1024)
    {
        // 使用Bitonic Sort（单Block，512线程）
        dim3 sort_block(512);
        bitonicSort1024Kernel<<<1, sort_block, 0, stream_>>>(
            thrust::raw_pointer_cast(d_batch_inlier_counts_.data()),
            thrust::raw_pointer_cast(d_indices_full_.data()),
            batch_size);
    }
    else
    {
        // 降级方案：使用thrust::sort_by_key（对于非1024的batch_size）
        thrust::sort_by_key(
            thrust::cuda::par.on(stream_),
            d_batch_inlier_counts_.begin(), 
            d_batch_inlier_counts_.begin() + batch_size,
            d_indices_full_.begin(),
            thrust::greater<int>()
        );
    }

    // 提取前k个索引（排序后索引0是最大值）
    thrust::copy_n(
        thrust::cuda::par.on(stream_),
        d_indices_full_.begin(), 
        k, 
        d_top_k_indices_.begin()
    );
}

void QuadricDetect::launchFineCountInliersBatch(int k)
{
    // 确保预分配的内存足够大
    if (d_fine_inlier_counts_.size() < static_cast<size_t>(k) ||
        d_candidate_models_.size() < static_cast<size_t>(k))
    {
        std::cerr << "[launchFineCountInliersBatch] 错误：预分配内存不足" << std::endl;
        return;
    }

    // 从 d_batch_models_ 中提取候选模型
    // 使用 thrust::gather 高效提取
    thrust::gather(
        thrust::cuda::par.on(stream_),
        d_top_k_indices_.begin(),
        d_top_k_indices_.begin() + k,
        d_batch_models_.begin(),
        d_candidate_models_.begin()
    );

    // 清零精选计数数组
    thrust::fill_n(thrust::cuda::par.on(stream_), d_fine_inlier_counts_.begin(), k, 0);

    // 启动精选kernel
    dim3 block(256);
    dim3 grid_x((d_remaining_indices_.size() + block.x - 1) / block.x);
    dim3 grid(grid_x.x, k);  // Y维度对应k个候选模型

    fineCountInliers_Kernel<<<grid, block, 0, stream_>>>(
        getPointsPtr(),
        thrust::raw_pointer_cast(d_remaining_indices_.data()),
        static_cast<int>(d_remaining_indices_.size()),
        d_valid_mask_,
        thrust::raw_pointer_cast(d_candidate_models_.data()),
        thrust::raw_pointer_cast(d_top_k_indices_.data()),
        k,
        static_cast<float>(params_.quadric_distance_threshold),
        thrust::raw_pointer_cast(d_fine_inlier_counts_.data()));
    
    cudaStreamSynchronize(stream_);
}

void QuadricDetect::launchFindBestModel(int batch_size)
{
    // 直接获取排序后的结果（排序后索引0是最优模型）
    // 注意：此函数假设d_batch_inlier_counts_和d_indices_full_已经通过launchSelectTopKModels排序
    int best_idx = 0;
    int best_count = 0;
    
    // 直接从Device内存异步拷贝索引0的值
    cudaMemcpyAsync(
        &best_idx,
        thrust::raw_pointer_cast(d_indices_full_.data()),
        sizeof(int),
        cudaMemcpyDeviceToHost,
        stream_
    );
    
    cudaMemcpyAsync(
        &best_count,
        thrust::raw_pointer_cast(d_batch_inlier_counts_.data()),
        sizeof(int),
        cudaMemcpyDeviceToHost,
        stream_
    );
    
    // 同步等待拷贝完成
    cudaStreamSynchronize(stream_);
    
    // 写回Device内存（保持兼容性）
    cudaMemcpyAsync(
        thrust::raw_pointer_cast(d_best_model_index_.data()),
        &best_idx,
        sizeof(int),
        cudaMemcpyHostToDevice,
        stream_
    );
    
    cudaMemcpyAsync(
        thrust::raw_pointer_cast(d_best_model_count_.data()),
        &best_count,
        sizeof(int),
        cudaMemcpyHostToDevice,
        stream_
    );
    
    // 验证日志
    if (params_.verbosity > 0)
    {
        std::cout << "[Final Check] Best model count: " << best_count 
                  << ", Index: " << best_idx << std::endl;
    }
}

// 替换你 QuadricDetect.cu 文件中的占位符实现：
void QuadricDetect::launchExtractInliers(const GPUQuadricModel *model)
{
    if (params_.verbosity > 0)
    {
        std::cout << "[launchExtractInliers] 开始提取内点索引" << std::endl;
    }

    // 修复开始：添加详细的输入验证
    // std::cout << "debug1" << std::endl;

    // 验证输入参数
    if (model == nullptr)
    {
        std::cerr << "[launchExtractInliers]  错误：model指针为空！" << std::endl;
        current_inlier_count_ = 0;
        return;
    }

    if (d_remaining_indices_.size() == 0)
    {
        std::cerr << "[launchExtractInliers] 错误：没有剩余点可处理！" << std::endl;
        current_inlier_count_ = 0;
        return;
    }

    // 检查点云数据是否有效
    GPUPoint3f* points_ptr = getPointsPtr();
    if (points_ptr == nullptr)
    {
        std::cerr << "[launchExtractInliers] 错误：点云数据为空！" << std::endl;
        current_inlier_count_ = 0;
        return;
    }

    std::cout << "  - 剩余点数: " << d_remaining_indices_.size() << std::endl;
    std::cout << "  - 使用外部内存: " << (is_external_memory_ ? "是" : "否") << std::endl;
    std::cout << "  - 距离阈值: " << params_.quadric_distance_threshold << std::endl;

    //  关键修复：将model从CPU拷贝到GPU专用内存
    thrust::device_vector<GPUQuadricModel> d_model_safe(1);
    d_model_safe[0] = *model; // 安全拷贝
    // std::cout << "debug1.5 - 模型已安全拷贝到GPU" << std::endl;
        //  修复结束

    // 分配临时GPU内存存储内点索引
    d_temp_inlier_indices_.resize(d_remaining_indices_.size());
    // std::cout << "debug2" << std::endl;
    thrust::device_vector<int> d_inlier_count(1, 0);
    // std::cout << "debug3" << std::endl;

    // 配置CUDA网格
    dim3 block(256);
    dim3 grid((d_remaining_indices_.size() + block.x - 1) / block.x);
    // std::cout << "debug3.5 - Grid配置: " << grid.x << " blocks, " << block.x << " threads" << std::endl;

    //  修复：使用安全的GPU内存而不是CPU指针
    extractInliers_Kernel<<<grid, block, 0, stream_>>>(
        getPointsPtr(),
        thrust::raw_pointer_cast(d_remaining_indices_.data()),
        static_cast<int>(d_remaining_indices_.size()),
        thrust::raw_pointer_cast(d_model_safe.data()), //  使用GPU内存
        static_cast<float>(params_.quadric_distance_threshold),
        thrust::raw_pointer_cast(d_temp_inlier_indices_.data()),
        thrust::raw_pointer_cast(d_inlier_count.data()));
    // std::cout << "debug4" << std::endl;

    cudaStreamSynchronize(stream_);
    // std::cout << "debug5" << std::endl;

    //  修复开始：使用更安全的内存访问方法替代thrust::copy
    // 检查内核执行是否有错误
    cudaError_t kernel_error = cudaGetLastError();
    if (kernel_error != cudaSuccess)
    {
        std::cerr << "[launchExtractInliers] 内核执行错误: " << cudaGetErrorString(kernel_error) << std::endl;
        current_inlier_count_ = 0;
        return;
    }

    // 获取内点数量并调整大小
    // 原始代码 - 可能导致非法内存访问：
    // thrust::host_vector<int> h_count = d_inlier_count;
    // thrust::host_vector<int> h_count(1);
    // thrust::copy(d_inlier_count.begin(), d_inlier_count.end(), h_count.begin());

    //  新方案：使用原生cudaMemcpy，更安全可控
    int h_count_temp = 0;
    cudaError_t copy_error = cudaMemcpy(&h_count_temp,
                                        thrust::raw_pointer_cast(d_inlier_count.data()),
                                        sizeof(int),
                                        cudaMemcpyDeviceToHost);

    if (copy_error != cudaSuccess)
    {
        std::cerr << "[launchExtractInliers]  内存拷贝错误: " << cudaGetErrorString(copy_error) << std::endl;
        current_inlier_count_ = 0;
        return;
    }

    current_inlier_count_ = h_count_temp;
            //  修复结束

    // std::cout << "debug6" << std::endl;

    // 原始代码已移除 - 会导致编译错误：
    // current_inlier_count_ = h_count[0];

    // std::cout << "debug7" << std::endl;

    d_temp_inlier_indices_.resize(current_inlier_count_);
    // std::cout << "debug8" << std::endl;

    if (params_.verbosity > 0)
    {
        std::cout << "[launchExtractInliers] 找到 " << current_inlier_count_ << " 个内点" << std::endl;
    }
}

void QuadricDetect::getBestModelResults(thrust::host_vector<int> &h_best_index, thrust::host_vector<int> &h_best_count)
{
    // 从device拷贝到host
    h_best_index = d_best_model_index_;
    h_best_count = d_best_model_count_;
}

// 标记掩码内核 - 极速逻辑移除
__global__ void markMaskKernel(
    const int *inlier_indices,
    int inlier_count,
    uint8_t *valid_mask,
    int mask_size)  // 添加掩码大小参数，防止越界
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < inlier_count) {
        int point_idx = inlier_indices[idx];
        // 添加边界检查，防止越界访问
        if (point_idx >= 0 && point_idx < mask_size) {
            valid_mask[point_idx] = 0;  // 标记为已移除
        }
    }
}

// 掩码压缩实现 - 双缓冲压实方案（避免 remove_if 原地操作失效）
void QuadricDetect::launchRemovePointsKernel()
{
    // 边界检查
    if (current_inlier_count_ <= 0 || current_inlier_count_ > static_cast<int>(d_temp_inlier_indices_.size()))
    {
        if (params_.verbosity > 0)
        {
            std::cout << "[launchRemovePointsKernel] 跳过：内点数量无效" << std::endl;
        }
        return;
    }
    
    if (d_valid_mask_ == nullptr)
    {
        std::cerr << "[launchRemovePointsKernel] 错误：掩码缓冲区未初始化" << std::endl;
        return;
    }
    
    // Step 0: 确保上一轮的掩码重置已完成（关键！）
    cudaStreamSynchronize(stream_);
    
    // Step 1: 标记内点为0
    dim3 block(256);
    dim3 grid((current_inlier_count_ + block.x - 1) / block.x);
    
    // 添加边界检查：确保掩码大小有效
    if (original_total_count_ == 0)
    {
        std::cerr << "[launchRemovePointsKernel] 错误：original_total_count_ 为0，无法标记掩码" << std::endl;
        return;
    }
    
    // 调试：打印一些内点索引，检查是否在范围内
    if (params_.verbosity > 1 && current_inlier_count_ > 0)
    {
        thrust::host_vector<int> h_check_indices(std::min(3, current_inlier_count_));
        thrust::copy_n(d_temp_inlier_indices_.begin(), h_check_indices.size(), h_check_indices.begin());
        std::cout << "[launchRemovePointsKernel] 准备标记 " << current_inlier_count_ 
                  << " 个内点，掩码大小=" << original_total_count_ << std::endl;
        std::cout << "  前3个内点索引: ";
        for (size_t i = 0; i < h_check_indices.size(); ++i)
        {
            std::cout << h_check_indices[i];
            if (h_check_indices[i] >= static_cast<int>(original_total_count_))
            {
                std::cout << "(越界！)";
            }
            std::cout << " ";
        }
        std::cout << std::endl;
    }
    
    markMaskKernel<<<grid, block, 0, stream_>>>(
        thrust::raw_pointer_cast(d_temp_inlier_indices_.data()),
        current_inlier_count_,
        d_valid_mask_,
        static_cast<int>(original_total_count_));  // 传入掩码大小
    
    // Step 1.5: 关键同步点 - 确保标记完成后再进行压实
    cudaStreamSynchronize(stream_);
    
    // 检查内核执行错误
    cudaError_t mark_err = cudaGetLastError();
    if (mark_err != cudaSuccess)
    {
        std::cerr << "[launchRemovePointsKernel] markMaskKernel 错误: " 
                  << cudaGetErrorString(mark_err) << std::endl;
        return;
    }
    
    // 验证标记是否成功（调试用）
    if (params_.verbosity > 1 && current_inlier_count_ > 0)
    {
        // 检查前几个内点索引的掩码值
        thrust::host_vector<int> h_sample_inliers(std::min(5, current_inlier_count_));
        thrust::copy_n(d_temp_inlier_indices_.begin(), h_sample_inliers.size(), h_sample_inliers.begin());
        
        thrust::host_vector<uint8_t> h_sample_mask(h_sample_inliers.size());
        for (size_t i = 0; i < h_sample_inliers.size(); ++i)
        {
            int idx = h_sample_inliers[i];
            if (idx >= 0 && idx < static_cast<int>(original_total_count_))
            {
                cudaMemcpy(&h_sample_mask[i], &d_valid_mask_[idx], sizeof(uint8_t), cudaMemcpyDeviceToHost);
            }
            else
            {
                h_sample_mask[i] = 255;  // 标记为越界
            }
        }
        
        std::cout << "[launchRemovePointsKernel] 标记验证（前" << h_sample_inliers.size() << "个内点，掩码大小=" << original_total_count_ << "）:" << std::endl;
        for (size_t i = 0; i < h_sample_inliers.size(); ++i)
        {
            std::cout << "  索引 " << h_sample_inliers[i] << " 的掩码值: " << static_cast<int>(h_sample_mask[i]);
            if (h_sample_inliers[i] >= static_cast<int>(original_total_count_))
            {
                std::cout << " (越界！)";
            }
            std::cout << std::endl;
        }
    }
    
    // Step 2: 双缓冲压实 - 创建临时桶
    thrust::device_vector<int> d_next_remaining(d_remaining_indices_.size());
    
    // Step 3: 使用 Stencil 版本的 copy_if 进行压实
    // Input: d_remaining_indices_
    // Stencil: 使用 valid_mask 作为判定标准（通过 permutation_iterator 按索引取掩码）
    // Output: 写入 d_next_remaining
    // Predicate: 只有掩码为 1（有效点）才拷贝
    thrust::device_ptr<uint8_t> mask_ptr = thrust::device_pointer_cast(d_valid_mask_);
    
    auto new_end = thrust::copy_if(
        thrust::cuda::par.on(stream_),
        d_remaining_indices_.begin(), 
        d_remaining_indices_.end(),
        thrust::make_permutation_iterator(mask_ptr, d_remaining_indices_.begin()), // 关键：按索引取掩码
        d_next_remaining.begin(),
        [] __device__ (uint8_t mask_val) { return mask_val == 1; }); // 只有掩码为 1 的才留下
    
    // Step 4: 关键同步点 - 确保压实完成
    cudaStreamSynchronize(stream_);
    
    // Step 5: 错误检查
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        std::cerr << "[launchRemovePointsKernel] CUDA 错误: " 
                  << cudaGetErrorString(err) << std::endl;
        return;
    }
    
    // Step 6: 计算新大小并交换
    size_t new_size = thrust::distance(d_next_remaining.begin(), new_end);
    d_remaining_indices_.swap(d_next_remaining);
    d_remaining_indices_.resize(new_size);
    
    // Step 7: 重置掩码为1（为下一轮准备）
    // 使用原始总点数确保全量覆盖
    if (original_total_count_ > 0)
    {
        cudaMemsetAsync(d_valid_mask_, 1, original_total_count_ * sizeof(uint8_t), stream_);
    }
    
    if (params_.verbosity > 0)
    {
        std::cout << "[launchRemovePointsKernel] 压缩完成，剩余点数: " << new_size << std::endl;
    }
}

// 新增函数实现--反幂迭代的核心实现
// 添加到QuadricDetect.cu

// 1. 计算A^T*A矩阵
__global__ void computeATA_Kernel(
    const float *batch_matrices, // 输入：1024个6×10矩阵（填充为9×10）
    float *batch_ATA_matrices,   // 输出：1024个10×10 A^T*A矩阵
    int batch_size)
{
    int batch_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_id >= batch_size)
        return;

    const float *A = &batch_matrices[batch_id * 90];  // 9×10矩阵（内存布局）
    float *ATA = &batch_ATA_matrices[batch_id * 100]; // 10×10矩阵

    // 计算A^T * A（只使用前6行有效数据）
    for (int i = 0; i < 10; ++i)
    {
        for (int j = i; j < 10; ++j)
        { // 只计算上三角，利用对称性
            float sum = 0.0f;
            for (int k = 0; k < 6; ++k)  // 修改：只计算前6行
            {
                sum += A[i * 9 + k] * A[j * 9 + k]; // A^T[i][k] * A[j][k]
            }
            ATA[i * 10 + j] = sum;
            ATA[j * 10 + i] = sum; // 对称矩阵
        }
    }
}

__global__ void batchQR_Kernel(
    const float *batch_ATA_matrices,
    float *batch_R_matrices,
    int batch_size)
{
    int batch_id = blockIdx.x;
    if (batch_id >= batch_size)
        return;

    __shared__ float A[10][10];
    __shared__ float R[10][10];

    //  1. 先初始化R矩阵为零
    for (int i = threadIdx.x; i < 100; i += blockDim.x)
    {
        ((float *)R)[i] = 0.0f;
    }
    __syncthreads();

    // 2. 加载A^T*A到共享内存
    const float *ATA = &batch_ATA_matrices[batch_id * 100];
    for (int i = threadIdx.x; i < 100; i += blockDim.x)
    {
        ((float *)A)[i] = ATA[i];
    }
    __syncthreads();

    // 3. 执行Gram-Schmidt QR分解
    for (int k = 0; k < 10; ++k)
    {
        if (threadIdx.x == 0)
        {
            // 计算第k列的模长
            float norm_sq = 0.0f;
            for (int i = k; i < 10; ++i)
            {
                norm_sq += A[i][k] * A[i][k];
            }
            float norm = sqrtf(norm_sq);

            // 数值稳定性检查
            if (norm < 1e-12f)
            {
                for (int i = k; i < 10; ++i)
                {
                    A[i][k] = (i == k) ? 1.0f : 0.0f;
                }
                norm = 1.0f;
            }

            // 归一化第k列
            for (int i = k; i < 10; ++i)
            {
                A[i][k] /= norm;
            }

            //  设置R[k][k] (对角线元素)
            R[k][k] = norm;

            // 正交化后续列
            for (int j = k + 1; j < 10; ++j)
            {
                // 计算投影系数
                float proj_coeff = 0.0f;
                for (int i = k; i < 10; ++i)
                {
                    proj_coeff += A[i][k] * A[i][j];
                }

                //  设置R[k][j] (上三角元素)
                R[k][j] = proj_coeff;

                // 从a_j中减去投影
                for (int i = k; i < 10; ++i)
                {
                    A[i][j] -= proj_coeff * A[i][k];
                }
            }
        }
        __syncthreads();
    }

    //  4. 输出R矩阵 (不要再清零了!)
    float *R_out = &batch_R_matrices[batch_id * 100];
    for (int i = threadIdx.x; i < 100; i += blockDim.x)
    {
        R_out[i] = ((float *)R)[i];
    }
}

// 3. 反幂迭代内核
__global__ void batchInversePowerIteration_Kernel(
    const float *batch_R_matrices, // 输入：1024个10×10 R矩阵
    float *batch_eigenvectors,     // 输出：1024个10维最小特征向量
    curandState *rand_states,      // 随机数状态
    int batch_size)
{
    int batch_id = blockIdx.x;
    if (batch_id >= batch_size)
        return;

    __shared__ float R[10][10]; // R矩阵
    __shared__ float x[10];     // 当前向量
    __shared__ float y[10];     // 临时向量

    // 加载R矩阵
    const float *R_in = &batch_R_matrices[batch_id * 100];
    for (int i = threadIdx.x; i < 100; i += blockDim.x)
    {
        ((float *)R)[i] = R_in[i];
    }

    // 初始化随机向量
    if (threadIdx.x < 10)
    {
        curandState local_state = rand_states[batch_id * 10 + threadIdx.x];
        x[threadIdx.x] = curand_uniform(&local_state);
        rand_states[batch_id * 10 + threadIdx.x] = local_state;
    }
    __syncthreads();

    // 反幂迭代：8次迭代
    for (int iter = 0; iter < 8; ++iter)
    {
        // 解 R * y = x (回代法)
        if (threadIdx.x == 0)
        {
            for (int i = 9; i >= 0; --i)
            {
                float sum = x[i];
                for (int j = i + 1; j < 10; ++j)
                {
                    sum -= R[i][j] * y[j];
                }
                y[i] = (fabsf(R[i][i]) > 1e-12f) ? sum / R[i][i] : 0.0f;
            }
        }
        __syncthreads();

        // 归一化 y -> x
        if (threadIdx.x == 0)
        {
            float norm = 0.0f;
            for (int i = 0; i < 10; ++i)
            {
                norm += y[i] * y[i];
            }
            norm = sqrtf(norm);
            if (norm > 1e-12f)
            {
                for (int i = 0; i < 10; ++i)
                {
                    x[i] = y[i] / norm;
                }
            }
        }
        __syncthreads();
    }

    // 输出最终特征向量
    float *output = &batch_eigenvectors[batch_id * 10];
    if (threadIdx.x < 10)
    {
        output[threadIdx.x] = x[threadIdx.x];
    }
}

// 4. 提取二次曲面模型内核
__global__ void extractQuadricModels_Kernel(
    const float *batch_eigenvectors, // 输入：1024个10维特征向量
    GPUQuadricModel *batch_models,   // 输出：1024个二次曲面模型
    int batch_size)
{
    int batch_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_id >= batch_size)
        return;

    const float *eigenvec = &batch_eigenvectors[batch_id * 10];
    GPUQuadricModel *model = &batch_models[batch_id];

    // 初始化coeffs数组
    for (int i = 0; i < 16; ++i)
    {
        model->coeffs[i] = 0.0f;
    }

    // L2归一化
    float norm_sq = 0.0f;
    for (int i = 0; i < 10; ++i)
    {
        norm_sq += eigenvec[i] * eigenvec[i];
    }
    float norm_factor = (norm_sq > 1e-12f) ? 1.0f / sqrtf(norm_sq) : 1.0f;

    // 10维向量→16维coeffs的映射 (4x4对称矩阵按行主序存储)
    // 二次曲面方程: Ax² + By² + Cz² + 2Dxy + 2Exz + 2Fyz + 2Gx + 2Hy + 2Iz + J = 0
    // 对应特征向量: [A, B, C, D, E, F, G, H, I, J]

    float A = eigenvec[0] * norm_factor; // x²系数
    float B = eigenvec[1] * norm_factor; // y²系数
    float C = eigenvec[2] * norm_factor; // z²系数
    float D = eigenvec[3] * norm_factor; // xy系数
    float E = eigenvec[4] * norm_factor; // xz系数
    float F = eigenvec[5] * norm_factor; // yz系数
    float G = eigenvec[6] * norm_factor; // x系数
    float H = eigenvec[7] * norm_factor; // y系数
    float I = eigenvec[8] * norm_factor; // z系数
    float J = eigenvec[9] * norm_factor; // 常数项

    // 4×4对称矩阵Q的映射 (按行主序存储到coeffs[16])
    // Q = [[A,   D,   E,   G],
    //      [D,   B,   F,   H],
    //      [E,   F,   C,   I],
    //      [G,   H,   I,   J]]

    model->coeffs[0] = A;  // Q(0,0)
    model->coeffs[1] = D;  // Q(0,1)
    model->coeffs[2] = E;  // Q(0,2)
    model->coeffs[3] = G;  // Q(0,3)
    model->coeffs[4] = D;  // Q(1,0) = Q(0,1)
    model->coeffs[5] = B;  // Q(1,1)
    model->coeffs[6] = F;  // Q(1,2)
    model->coeffs[7] = H;  // Q(1,3)
    model->coeffs[8] = E;  // Q(2,0) = Q(0,2)
    model->coeffs[9] = F;  // Q(2,1) = Q(1,2)
    model->coeffs[10] = C; // Q(2,2)
    model->coeffs[11] = I; // Q(2,3)
    model->coeffs[12] = G; // Q(3,0) = Q(0,3)
    model->coeffs[13] = H; // Q(3,1) = Q(1,3)
    model->coeffs[14] = I; // Q(3,2) = Q(2,3)
    model->coeffs[15] = J; // Q(3,3)
}

// 包装函数
//  添加到QuadricDetect.cu

void QuadricDetect::launchComputeATA(int batch_size)
{
    dim3 block(256);
    dim3 grid((batch_size + block.x - 1) / block.x);

    computeATA_Kernel<<<grid, block, 0, stream_>>>(
        thrust::raw_pointer_cast(d_batch_matrices_.data()),
        thrust::raw_pointer_cast(d_batch_ATA_matrices_.data()),
        batch_size);
    cudaStreamSynchronize(stream_);
}

void QuadricDetect::launchBatchQR(int batch_size)
{
    dim3 block(256);
    dim3 grid(batch_size); // 每个block处理一个矩阵

    batchQR_Kernel<<<grid, block, 0, stream_>>>(
        thrust::raw_pointer_cast(d_batch_ATA_matrices_.data()),
        thrust::raw_pointer_cast(d_batch_R_matrices_.data()),
        batch_size);
    cudaStreamSynchronize(stream_);
}

void QuadricDetect::launchBatchInversePower(int batch_size)
{
    dim3 block(256);
    dim3 grid(batch_size); // 每个block处理一个矩阵

    batchInversePowerIteration_Kernel<<<grid, block, 0, stream_>>>(
        thrust::raw_pointer_cast(d_batch_R_matrices_.data()),
        thrust::raw_pointer_cast(d_batch_eigenvectors_.data()),
        thrust::raw_pointer_cast(d_rand_states_.data()),
        batch_size);
    cudaStreamSynchronize(stream_);
}

void QuadricDetect::launchExtractQuadricModels(int batch_size)
{
    dim3 block(256);
    dim3 grid((batch_size + block.x - 1) / block.x);

    extractQuadricModels_Kernel<<<grid, block, 0, stream_>>>(
        thrust::raw_pointer_cast(d_batch_eigenvectors_.data()),
        thrust::raw_pointer_cast(d_batch_models_.data()),
        batch_size);
    cudaStreamSynchronize(stream_);
}

// 重载实现
void QuadricDetect::uploadPointsToGPU(const thrust::device_vector<GPUPoint3f> &h_points)
{
    //  关键修复：强制清空旧数据，防止多帧复用时的内存污染
    d_all_points_.clear();
    d_remaining_indices_.clear();

    // 重新上传新数据
    d_all_points_ = h_points;
    d_remaining_indices_.resize(h_points.size());
    thrust::sequence(d_remaining_indices_.begin(), d_remaining_indices_.end(), 0);
}

// GPU 辅助函数：将内点聚集到紧凑缓冲区
void QuadricDetect::gatherInliersToCompact() const
{
    if (d_temp_inlier_indices_.empty() || current_inlier_count_ == 0)
    {
        return;
    }
    
    // 在 GPU 内部使用 gather 聚集内点到连续缓冲区
    d_compact_inliers_.resize(current_inlier_count_);
    
    if (is_external_memory_ && d_external_points_ != nullptr)
    {
        // 外部内存模式：使用 thrust::gather 从外部指针聚集内点
        thrust::device_ptr<GPUPoint3f> external_ptr = thrust::device_pointer_cast(d_external_points_);
        thrust::gather(
            thrust::cuda::par.on(stream_),
            d_temp_inlier_indices_.begin(),
            d_temp_inlier_indices_.begin() + current_inlier_count_,
            external_ptr,
            d_compact_inliers_.begin()
        );
    }
    else
    {
        // 内部内存模式：使用 thrust::gather 从 d_all_points_ 聚集内点
        thrust::gather(
            thrust::cuda::par.on(stream_),
            d_temp_inlier_indices_.begin(),
            d_temp_inlier_indices_.begin() + current_inlier_count_,
            d_all_points_.begin(),
            d_compact_inliers_.begin()
        );
    }
    
    cudaStreamSynchronize(stream_);
}

// GPU 辅助函数：将剩余点聚集到紧凑缓冲区
void QuadricDetect::gatherRemainingToCompact() const
{
    if (d_remaining_indices_.empty())
    {
        return;
    }
    
    size_t remaining_count = d_remaining_indices_.size();
    
    // 在 GPU 内部使用 gather 聚集剩余点到连续缓冲区
    d_compact_inliers_.resize(remaining_count);
    
    if (is_external_memory_ && d_external_points_ != nullptr)
    {
        // 外部内存模式：使用 thrust::gather 从外部指针聚集剩余点
        thrust::device_ptr<GPUPoint3f> external_ptr = thrust::device_pointer_cast(d_external_points_);
        thrust::gather(
            thrust::cuda::par.on(stream_),
            d_remaining_indices_.begin(),
            d_remaining_indices_.end(),
            external_ptr,
            d_compact_inliers_.begin()
        );
    }
    else
    {
        // 内部内存模式：使用 thrust::gather 从 d_all_points_ 聚集剩余点
        thrust::gather(
            thrust::cuda::par.on(stream_),
            d_remaining_indices_.begin(),
            d_remaining_indices_.end(),
            d_all_points_.begin(),
            d_compact_inliers_.begin()
        );
    }
    
    cudaStreamSynchronize(stream_);
}
