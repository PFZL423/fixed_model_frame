#include "gpu_demo/QuadricDetect.h"
#include "gpu_demo/QuadricDetect_kernels.cuh"
#include <cusolverDn.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <ctime>
#include <iostream>
#include <cmath>     // 添加这个头文件用于isfinite函数
#include <algorithm> // 添加这个头文件用于min函数
#include <stdexcept> // 用于 std::exception

// ========================================
// CUDA内核函数定义 (每个内核只定义一次!)
// ========================================

__global__ void initCurandStates_Kernel(curandState *states, unsigned long seed, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

__global__ void sampleAndBuildMatrices_Kernel(
    const GPUPoint3f *all_points,
    const int *remaining_indices,
    int num_remaining,
    curandState *rand_states,
    int batch_size,
    float *batch_matrices)
{
    int model_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (model_id >= batch_size)
        return;

    curandState local_state = rand_states[model_id];

    // 采样9个点
    int sample_indices[9];
    for (int i = 0; i < 9; ++i)
    {
        sample_indices[i] = remaining_indices[curand(&local_state) % num_remaining];
    }

    // 构造9x10的A矩阵 (🔧 修复：按列主序存储，符合cuSolver要求)
    float *A = &batch_matrices[model_id * 90]; // 9*10

    for (int i = 0; i < 9; ++i)
    {
        GPUPoint3f pt = all_points[sample_indices[i]];
        float x = pt.x, y = pt.y, z = pt.z;

        // 🔧 关键修复：检查并处理无效的点云数据
        if (!isfinite(x) || !isfinite(y) || !isfinite(z) ||
            isnan(x) || isnan(y) || isnan(z) ||
            isinf(x) || isinf(y) || isinf(z))
        {
            // 🚨 发现无效点，用默认值替换
            x = 0.0f;
            y = 0.0f;
            z = 0.0f;
        }

        // 🎯 关键修复：列主序存储 A[col * m + row]
        A[0 * 9 + i] = x * x; // x² (第0列)
        A[1 * 9 + i] = y * y; // y² (第1列)
        A[2 * 9 + i] = z * z; // z² (第2列)
        A[3 * 9 + i] = x * y; // xy (第3列)
        A[4 * 9 + i] = x * z; // xz (第4列)
        A[5 * 9 + i] = y * z; // yz (第5列)
        A[6 * 9 + i] = x;     // x  (第6列)
        A[7 * 9 + i] = y;     // y  (第7列)
        A[8 * 9 + i] = z;     // z  (第8列)
        A[9 * 9 + i] = 1.0f;  // 常数项 (第9列)

        // 🔧 二次验证：确保生成的值都是有效的
        for (int col = 0; col < 10; ++col)
        {
            float val = A[col * 9 + i];
            if (!isfinite(val) || isnan(val) || isinf(val))
            {
                A[col * 9 + i] = (col == 9) ? 1.0f : 0.0f; // 常数项设为1，其他设为0
            }
        }
    }

    rand_states[model_id] = local_state;
}

__global__ void countInliersBatch_Kernel(
    const GPUPoint3f *all_points,
    const int *remaining_indices,
    int num_remaining,
    const GPUQuadricModel *batch_models,
    int batch_size,
    float threshold,
    int *batch_inlier_counts)
{
    int model_id = blockIdx.y; // 使用2D grid，y维度对应模型
    if (model_id >= batch_size)
        return;

    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int local_count = 0;

    // 每个线程处理多个点
    for (int i = thread_id; i < num_remaining; i += blockDim.x * gridDim.x)
    {
        GPUPoint3f point = all_points[remaining_indices[i]];
        float dist = evaluateQuadricDistance(point, batch_models[model_id]);

        if (dist < threshold)
        {
            local_count++;
        }
    }

    // Block内reduce求和
    __shared__ int shared_counts[256];
    shared_counts[threadIdx.x] = local_count;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (threadIdx.x < stride)
        {
            shared_counts[threadIdx.x] += shared_counts[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        atomicAdd(&batch_inlier_counts[model_id], shared_counts[0]);
    }
}

__device__ inline float evaluateQuadricDistance(
    const GPUPoint3f &point,
    const GPUQuadricModel &model)
{
    float x = point.x, y = point.y, z = point.z;

    // 🔧 修复开始：添加输入验证
    // 验证输入点的有效性
    if (!isfinite(x) || !isfinite(y) || !isfinite(z) ||
        isnan(x) || isnan(y) || isnan(z) ||
        isinf(x) || isinf(y) || isinf(z))
    {
        return 1e10f; // 返回一个很大的距离，表示无效点
    }

    // 验证模型系数的有效性
    bool model_valid = true;
    for (int i = 0; i < 16; ++i)
    {
        if (!isfinite(model.coeffs[i]) || isnan(model.coeffs[i]) || isinf(model.coeffs[i]))
        {
            model_valid = false;
            break;
        }
    }

    if (!model_valid)
    {
        return 1e10f; // 返回一个很大的距离，表示无效模型
    }
    // 🔧 修复结束

    // 手写二次型计算: [x y z 1] * Q * [x y z 1]^T
    float result = 0.0f;
    float coords[4] = {x, y, z, 1.0f};

    // 🔧 修复：使用更安全的矩阵乘法，避免潜在的内存访问问题
    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            int idx = i * 4 + j;      // 确保索引在有效范围内
            if (idx >= 0 && idx < 16) // 🔧 添加边界检查
            {
                float coeff = model.coeffs[idx];
                // 🔧 验证每次乘法的结果
                float term = coords[i] * coeff * coords[j];
                if (isfinite(term) && !isnan(term) && !isinf(term))
                {
                    result += term;
                }
            }
        }
    }

    // 🔧 修复：验证最终结果的有效性
    if (!isfinite(result) || isnan(result) || isinf(result))
    {
        return 1e10f; // 返回一个很大的距离，表示计算失败
    }

    return fabsf(result);
}

__global__ void findBestModel_Kernel(
    const int *batch_inlier_counts,
    int batch_size,
    int *best_index,
    int *best_count)
{
    int thread_id = threadIdx.x;
    int local_best_idx = -1;
    int local_best_count = 0;

    // 每个线程处理多个模型
    for (int i = thread_id; i < batch_size; i += blockDim.x)
    {
        if (batch_inlier_counts[i] > local_best_count)
        {
            local_best_count = batch_inlier_counts[i];
            local_best_idx = i;
        }
    }

    // Block内reduce找最大值
    __shared__ int shared_counts[256];
    __shared__ int shared_indices[256];

    shared_counts[thread_id] = local_best_count;
    shared_indices[thread_id] = local_best_idx;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (thread_id < stride)
        {
            if (shared_counts[thread_id + stride] > shared_counts[thread_id])
            {
                shared_counts[thread_id] = shared_counts[thread_id + stride];
                shared_indices[thread_id] = shared_indices[thread_id + stride];
            }
        }
        __syncthreads();
    }

    if (thread_id == 0)
    {
        *best_count = shared_counts[0];
        *best_index = shared_indices[0];
    }
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

    // 🔧 修复开始：添加更多安全检查
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

    // 🔧 关键修复：确保我们不访问超出all_points数组边界的内存
    // 注意：我们无法在GPU内核中直接获取all_points的大小，所以需要依赖调用方确保索引有效

    GPUPoint3f point = all_points[global_point_index];

    // 🔧 验证点的有效性
    if (!isfinite(point.x) || !isfinite(point.y) || !isfinite(point.z) ||
        isnan(point.x) || isnan(point.y) || isnan(point.z) ||
        isinf(point.x) || isinf(point.y) || isinf(point.z))
    {
        return; // 跳过无效点
    }

    float dist = evaluateQuadricDistance(point, *model);

    // 🔧 验证距离计算结果的有效性
    if (!isfinite(dist) || isnan(dist) || isinf(dist))
    {
        return; // 跳过无效距离计算结果
    }
    // 🔧 修复结束

    if (dist < threshold)
    {
        // 🔧 修复开始：添加边界检查防止数组越界
        int write_pos = atomicAdd(inlier_count, 1);

        // 🔧 关键安全检查：确保不会越界访问
        // 理论上 d_temp_inlier_indices_ 大小等于 d_remaining_indices_.size()
        // 所以 write_pos 应该永远 < num_remaining，但为了安全还是检查
        if (write_pos < num_remaining)
        {
            inlier_indices[write_pos] = global_point_index;
        }
        else
        {
            // 🚨 如果发生越界，至少不会崩溃，但会丢失这个内点
            // 在实际应用中这种情况不应该发生
            atomicAdd(inlier_count, -1); // 回滚计数器
        }
        // 🔧 修复结束
    }
} // ========================================
// 成员函数实现 (每个函数只定义一次!)
// ========================================

void QuadricDetect::initializeGPUMemory(int batch_size)
{
    // 分配GPU内存
    d_batch_matrices_.resize(batch_size * 9 * 10);
    d_batch_models_.resize(batch_size);
    d_batch_inlier_counts_.resize(batch_size);
    d_rand_states_.resize(batch_size);

    // 初始化结果存储
    d_best_model_index_.resize(1);
    d_best_model_count_.resize(1);

    // 🆕 添加反幂迭代相关
    d_batch_ATA_matrices_.resize(batch_size * 10 * 10);
    d_batch_R_matrices_.resize(batch_size * 10 * 10);
    d_batch_eigenvectors_.resize(batch_size * 10);
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
    dim3 block(256);
    dim3 grid((batch_size + block.x - 1) / block.x);

    initCurandStates_Kernel<<<grid, block, 0, stream_>>>(
        thrust::raw_pointer_cast(d_rand_states_.data()),
        time(nullptr),
        batch_size);
    cudaStreamSynchronize(stream_);
}

void QuadricDetect::launchSampleAndBuildMatrices(int batch_size)
{
    if (params_.verbosity > 0)
    {
        std::cout << "[launchSampleAndBuildMatrices] 开始生成批量矩阵，batch_size=" << batch_size << std::endl;
        std::cout << "  - 剩余点数: " << d_remaining_indices_.size() << std::endl;
        std::cout << "  - 总点数: " << d_all_points_.size() << std::endl;
    }

    // 🔍 验证输入数据
    if (d_remaining_indices_.size() < 9)
    {
        std::cerr << "[launchSampleAndBuildMatrices] 🚨 错误：剩余点数不足9个，无法生成矩阵！" << std::endl;
        return;
    }

    if (d_all_points_.size() == 0)
    {
        std::cerr << "[launchSampleAndBuildMatrices] 🚨 错误：点云数据为空！" << std::endl;
        return;
    }

    // 🔧 新增：验证点云数据的有效性
    if (params_.verbosity > 1)
    {
        std::cout << "[launchSampleAndBuildMatrices] 🔍 验证输入点云数据有效性..." << std::endl;

        // 检查前几个点的数据
        thrust::host_vector<GPUPoint3f> h_sample_points(std::min(10, (int)d_all_points_.size()));
        cudaMemcpy(h_sample_points.data(),
                   thrust::raw_pointer_cast(d_all_points_.data()),
                   h_sample_points.size() * sizeof(GPUPoint3f),
                   cudaMemcpyDeviceToHost);

        int invalid_points = 0;
        for (size_t i = 0; i < h_sample_points.size(); ++i)
        {
            const GPUPoint3f &pt = h_sample_points[i];
            if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z) ||
                std::isnan(pt.x) || std::isnan(pt.y) || std::isnan(pt.z) ||
                std::isinf(pt.x) || std::isinf(pt.y) || std::isinf(pt.z))
            {
                invalid_points++;
                std::cout << "    🚨 发现无效点[" << i << "]: ("
                          << pt.x << ", " << pt.y << ", " << pt.z << ")" << std::endl;
            }
        }

        if (invalid_points > 0)
        {
            std::cout << "    🚨 警告：输入点云包含 " << invalid_points << " 个无效点！" << std::endl;
            std::cout << "    这可能导致SVD计算失败，建议预处理点云数据" << std::endl;
        }
        else
        {
            std::cout << "    ✓ 输入点云数据有效" << std::endl;
        }
    }

    dim3 block(256);
    dim3 grid((batch_size + block.x - 1) / block.x);

    // 🔍 先清零矩阵数据，确保没有垃圾数据
    thrust::fill(d_batch_matrices_.begin(), d_batch_matrices_.end(), 0.0f);

    sampleAndBuildMatrices_Kernel<<<grid, block, 0, stream_>>>(
        thrust::raw_pointer_cast(d_all_points_.data()),
        thrust::raw_pointer_cast(d_remaining_indices_.data()),
        static_cast<int>(d_remaining_indices_.size()),
        thrust::raw_pointer_cast(d_rand_states_.data()),
        batch_size,
        thrust::raw_pointer_cast(d_batch_matrices_.data()));

    cudaError_t kernel_error = cudaGetLastError();
    if (kernel_error != cudaSuccess)
    {
        std::cerr << "[launchSampleAndBuildMatrices] 🚨 内核启动错误: " << cudaGetErrorString(kernel_error) << std::endl;
        return;
    }

    cudaStreamSynchronize(stream_);

    cudaError_t sync_error = cudaGetLastError();
    if (sync_error != cudaSuccess)
    {
        std::cerr << "[launchSampleAndBuildMatrices] 🚨 内核执行错误: " << cudaGetErrorString(sync_error) << std::endl;
        return;
    }

    // 🔍 验证生成的矩阵数据
    if (params_.verbosity > 1)
    {
        std::cout << "[launchSampleAndBuildMatrices] 验证生成的矩阵..." << std::endl;

        // 检查第一个矩阵
        thrust::host_vector<float> h_first_matrix(9 * 10);
        cudaMemcpy(h_first_matrix.data(),
                   thrust::raw_pointer_cast(d_batch_matrices_.data()),
                   9 * 10 * sizeof(float),
                   cudaMemcpyDeviceToHost);

        bool all_zero = true;
        for (int i = 0; i < 9 * 10; ++i)
        {
            if (h_first_matrix[i] != 0.0f)
            {
                all_zero = false;
                break;
            }
        }

        if (all_zero)
        {
            std::cerr << "[launchSampleAndBuildMatrices] 🚨 生成的矩阵全为零！检查内核实现" << std::endl;

            // 🔍 检查输入点云数据
            thrust::host_vector<GPUPoint3f> h_points_sample(std::min(10, (int)d_all_points_.size()));
            cudaMemcpy(h_points_sample.data(),
                       thrust::raw_pointer_cast(d_all_points_.data()),
                       h_points_sample.size() * sizeof(GPUPoint3f),
                       cudaMemcpyDeviceToHost);

            std::cout << "  - 前几个点云数据样本:" << std::endl;
            for (size_t i = 0; i < h_points_sample.size(); ++i)
            {
                std::cout << "    点" << i << ": (" << h_points_sample[i].x
                          << ", " << h_points_sample[i].y
                          << ", " << h_points_sample[i].z << ")" << std::endl;
            }

            // 🔍 检查剩余索引
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
            std::cout << "[launchSampleAndBuildMatrices] ✓ 矩阵生成成功，包含非零数据" << std::endl;
        }
    }

    if (params_.verbosity > 0)
    {
        std::cout << "[launchSampleAndBuildMatrices] 矩阵生成完成" << std::endl;
    }
}

void QuadricDetect::launchCountInliersBatch(int batch_size)
{
    // 修复: 使用2D grid匹配内核实现
    dim3 block(256);
    dim3 grid_x((d_remaining_indices_.size() + block.x - 1) / block.x);
    dim3 grid(grid_x.x, batch_size); // 2D grid: (points, models)

    // 先清零计数器
    thrust::fill(d_batch_inlier_counts_.begin(), d_batch_inlier_counts_.end(), 0);

    countInliersBatch_Kernel<<<grid, block, 0, stream_>>>(
        thrust::raw_pointer_cast(d_all_points_.data()),
        thrust::raw_pointer_cast(d_remaining_indices_.data()),
        static_cast<int>(d_remaining_indices_.size()),
        thrust::raw_pointer_cast(d_batch_models_.data()),
        batch_size,
        static_cast<float>(params_.quadric_distance_threshold),
        thrust::raw_pointer_cast(d_batch_inlier_counts_.data()));
    cudaStreamSynchronize(stream_);
}

void QuadricDetect::launchFindBestModel(int batch_size)
{
    findBestModel_Kernel<<<1, 256, 0, stream_>>>(
        thrust::raw_pointer_cast(d_batch_inlier_counts_.data()),
        batch_size,
        thrust::raw_pointer_cast(d_best_model_index_.data()),
        thrust::raw_pointer_cast(d_best_model_count_.data()));
    cudaStreamSynchronize(stream_);
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

    if (d_all_points_.size() == 0)
    {
        std::cerr << "[launchExtractInliers] 错误：点云数据为空！" << std::endl;
        current_inlier_count_ = 0;
        return;
    }

    std::cout << "  - 剩余点数: " << d_remaining_indices_.size() << std::endl;
    std::cout << "  - 总点数: " << d_all_points_.size() << std::endl;
    std::cout << "  - 距离阈值: " << params_.quadric_distance_threshold << std::endl;

    // 🔧 关键修复：将model从CPU拷贝到GPU专用内存
    thrust::device_vector<GPUQuadricModel> d_model_safe(1);
    d_model_safe[0] = *model; // 安全拷贝
    // std::cout << "debug1.5 - 模型已安全拷贝到GPU" << std::endl;
    // 🔧 修复结束

    // 分配临时GPU内存存储内点索引
    d_temp_inlier_indices_.resize(d_remaining_indices_.size());
    // std::cout << "debug2" << std::endl;
    thrust::device_vector<int> d_inlier_count(1, 0);
    // std::cout << "debug3" << std::endl;

    // 配置CUDA网格
    dim3 block(256);
    dim3 grid((d_remaining_indices_.size() + block.x - 1) / block.x);
    // std::cout << "debug3.5 - Grid配置: " << grid.x << " blocks, " << block.x << " threads" << std::endl;

    // 🔧 修复：使用安全的GPU内存而不是CPU指针
    extractInliers_Kernel<<<grid, block, 0, stream_>>>(
        thrust::raw_pointer_cast(d_all_points_.data()),
        thrust::raw_pointer_cast(d_remaining_indices_.data()),
        static_cast<int>(d_remaining_indices_.size()),
        thrust::raw_pointer_cast(d_model_safe.data()), // 🔧 使用GPU内存
        static_cast<float>(params_.quadric_distance_threshold),
        thrust::raw_pointer_cast(d_temp_inlier_indices_.data()),
        thrust::raw_pointer_cast(d_inlier_count.data()));
    // std::cout << "debug4" << std::endl;

    cudaStreamSynchronize(stream_);
    // std::cout << "debug5" << std::endl;

    // 🔧 修复开始：使用更安全的内存访问方法替代thrust::copy
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

    // 🔧 新方案：使用原生cudaMemcpy，更安全可控
    int h_count_temp = 0;
    cudaError_t copy_error = cudaMemcpy(&h_count_temp,
                                        thrust::raw_pointer_cast(d_inlier_count.data()),
                                        sizeof(int),
                                        cudaMemcpyDeviceToHost);

    if (copy_error != cudaSuccess)
    {
        std::cerr << "[launchExtractInliers] 🚨 内存拷贝错误: " << cudaGetErrorString(copy_error) << std::endl;
        current_inlier_count_ = 0;
        return;
    }

    current_inlier_count_ = h_count_temp;
    // 🔧 修复结束

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

// remove的GPU函数实现
// 在 QuadricDetect.cu 中添加内核
__global__ void removePointsKernel(
    const int *remaining_points,
    int remaining_count,
    const int *sorted_inliers, // 已排序的内点索引
    int inlier_count,
    int *output_points,
    int *output_count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= remaining_count)
        return;

    int point_id = remaining_points[idx];

    // GPU上二分查找
    bool is_inlier = false;
    int left = 0, right = inlier_count - 1;
    while (left <= right)
    {
        int mid = (left + right) / 2;
        if (sorted_inliers[mid] == point_id)
        {
            is_inlier = true;
            break;
        }
        if (sorted_inliers[mid] < point_id)
            left = mid + 1;
        else
            right = mid - 1;
    }

    // 如果不是内点，就保留
    if (!is_inlier)
    {
        int write_pos = atomicAdd(output_count, 1);
        output_points[write_pos] = point_id;
    }
}

// 包装函数
void QuadricDetect::launchRemovePointsKernel()
{
    // 修复：添加边界检查，防止 radix_sort 错误
    if (current_inlier_count_ <= 0)
    {
        if (params_.verbosity > 0)
        {
            std::cout << "[launchRemovePointsKernel] 跳过：内点数量为0" << std::endl;
        }
        return;
    }
    
    if (current_inlier_count_ > static_cast<int>(d_temp_inlier_indices_.size()))
    {
        std::cerr << "[launchRemovePointsKernel] 错误：内点数量 (" << current_inlier_count_
                  << ") 超出数组大小 (" << d_temp_inlier_indices_.size() << ")" << std::endl;
        return;
    }
    
    if (d_temp_inlier_indices_.empty())
    {
        std::cerr << "[launchRemovePointsKernel] 错误：内点索引数组为空" << std::endl;
        return;
    }
    
    // 确保之前的 CUDA 操作已完成
    cudaError_t sync_error = cudaStreamSynchronize(stream_);
    if (sync_error != cudaSuccess)
    {
        std::cerr << "[launchRemovePointsKernel] CUDA流同步错误: " 
                  << cudaGetErrorString(sync_error) << std::endl;
        return;
    }
    
    // 1. 对内点索引排序（纯GPU操作）
    try {
        thrust::sort(d_temp_inlier_indices_.begin(),
                     d_temp_inlier_indices_.begin() + current_inlier_count_);
        
        // 检查排序操作是否有错误
        cudaError_t sort_error = cudaGetLastError();
        if (sort_error != cudaSuccess)
        {
            std::cerr << "[launchRemovePointsKernel] 排序错误: " 
                      << cudaGetErrorString(sort_error) << std::endl;
            return;
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "[launchRemovePointsKernel] 排序异常: " << e.what() << std::endl;
        return;
    }

    // 2. 分配输出空间
    thrust::device_vector<int> d_new_remaining(d_remaining_indices_.size());
    thrust::device_vector<int> d_output_count(1, 0);

    // 3. 启动内核
    dim3 block(256);
    dim3 grid((d_remaining_indices_.size() + block.x - 1) / block.x);

    removePointsKernel<<<grid, block, 0, stream_>>>(
        thrust::raw_pointer_cast(d_remaining_indices_.data()),
        static_cast<int>(d_remaining_indices_.size()),
        thrust::raw_pointer_cast(d_temp_inlier_indices_.data()),
        current_inlier_count_,
        thrust::raw_pointer_cast(d_new_remaining.data()),
        thrust::raw_pointer_cast(d_output_count.data()));

    cudaStreamSynchronize(stream_);

    // 4. 获取实际输出大小并调整
    thrust::host_vector<int> h_count = d_output_count;
    int new_size = h_count[0]; //  这里有一次小传输，但unavoidable

    d_new_remaining.resize(new_size);
    d_remaining_indices_ = std::move(d_new_remaining);
}

// 新增函数实现--反幂迭代的核心实现
// 添加到QuadricDetect.cu

// 1. 计算A^T*A矩阵
__global__ void computeATA_Kernel(
    const float *batch_matrices, // 输入：1024个9×10矩阵
    float *batch_ATA_matrices,   // 输出：1024个10×10 A^T*A矩阵
    int batch_size)
{
    int batch_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (batch_id >= batch_size)
        return;

    const float *A = &batch_matrices[batch_id * 90];  // 9×10矩阵
    float *ATA = &batch_ATA_matrices[batch_id * 100]; // 10×10矩阵

    // 计算A^T * A
    for (int i = 0; i < 10; ++i)
    {
        for (int j = i; j < 10; ++j)
        { // 只计算上三角，利用对称性
            float sum = 0.0f;
            for (int k = 0; k < 9; ++k)
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
// 🆕 添加到QuadricDetect.cu

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
