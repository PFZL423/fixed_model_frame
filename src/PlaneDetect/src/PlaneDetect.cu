#include "PlaneDetect/PlaneDetect.h"
#include "PlaneDetect/PlaneDetect.cuh"
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/copy.h>
#include <thrust/distance.h>
#include <thrust/count.h>
#include <ctime>
#include <iostream>
#include <cmath>
#include <algorithm>

// ========================================
// CUDA内核函数定义 - 平面检测简化版
// ========================================

__global__ void initCurandStates_Kernel(curandState *states, unsigned long seed, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

// 平面检测：采样3点并直接拟合平面
__global__ void sampleAndFitPlanes_Kernel(
    const GPUPoint3f *all_points,
    const int *remaining_indices,
    int num_remaining,
    const uint8_t *valid_mask,  // 新增：有效性掩码
    curandState *rand_states,
    int batch_size,
    GPUPlaneModel *batch_models)
{
    int model_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (model_id >= batch_size)
        return;

    curandState local_state = rand_states[model_id];

    // 采样3个点（相比二次曲面的9个点大大简化）
    // 增加掩码检查：只采样有效点
    int sample_indices[3];
    for (int i = 0; i < 3; ++i)
    {
        int candidate_idx;
        int attempts = 0;
        do {
            candidate_idx = remaining_indices[curand(&local_state) % num_remaining];
            attempts++;
        } while (valid_mask[candidate_idx] == 0 && attempts < 100);
        
        if (valid_mask[candidate_idx] == 0) {
            // 如果100次尝试都失败，使用默认平面
            batch_models[model_id].coeffs[0] = 0.0f;
            batch_models[model_id].coeffs[1] = 0.0f;
            batch_models[model_id].coeffs[2] = 1.0f;
            batch_models[model_id].coeffs[3] = 0.0f;
            rand_states[model_id] = local_state;
            return;
        }
        sample_indices[i] = candidate_idx;
    }

    // 获取3个点
    GPUPoint3f p1 = all_points[sample_indices[0]];
    GPUPoint3f p2 = all_points[sample_indices[1]];
    GPUPoint3f p3 = all_points[sample_indices[2]];

    // 验证点的有效性
    if (!isfinite(p1.x) || !isfinite(p1.y) || !isfinite(p1.z) ||
        !isfinite(p2.x) || !isfinite(p2.y) || !isfinite(p2.z) ||
        !isfinite(p3.x) || !isfinite(p3.y) || !isfinite(p3.z))
    {
        // 设置默认平面 z = 0
        batch_models[model_id].coeffs[0] = 0.0f; // A
        batch_models[model_id].coeffs[1] = 0.0f; // B
        batch_models[model_id].coeffs[2] = 1.0f; // C
        batch_models[model_id].coeffs[3] = 0.0f; // D
        rand_states[model_id] = local_state;
        return;
    }

    // 计算两个向量
    float v1x = p2.x - p1.x, v1y = p2.y - p1.y, v1z = p2.z - p1.z;
    float v2x = p3.x - p1.x, v2y = p3.y - p1.y, v2z = p3.z - p1.z;

    // 计算法向量（叉积）
    float nx = v1y * v2z - v1z * v2y;
    float ny = v1z * v2x - v1x * v2z;
    float nz = v1x * v2y - v1y * v2x;

    // 归一化法向量
    float norm = sqrtf(nx * nx + ny * ny + nz * nz);
    if (norm < 1e-8f)
    {
        // 三点共线，设置默认平面
        batch_models[model_id].coeffs[0] = 0.0f;
        batch_models[model_id].coeffs[1] = 0.0f;
        batch_models[model_id].coeffs[2] = 1.0f;
        batch_models[model_id].coeffs[3] = 0.0f;
    }
    else
    {
        nx /= norm;
        ny /= norm;
        nz /= norm;

        // 计算平面方程 Ax + By + Cz + D = 0
        float d = -(nx * p1.x + ny * p1.y + nz * p1.z);

        batch_models[model_id].coeffs[0] = nx; // A
        batch_models[model_id].coeffs[1] = ny; // B
        batch_models[model_id].coeffs[2] = nz; // C
        batch_models[model_id].coeffs[3] = d;  // D
    }

    rand_states[model_id] = local_state;
}

// 计算点到平面的距离
__device__ inline float evaluatePlaneDistance(
    const GPUPoint3f &point,
    const GPUPlaneModel &model)
{
    // 平面方程: Ax + By + Cz + D = 0
    // 点到平面距离: |Ax + By + Cz + D| / sqrt(A² + B² + C²)

    float x = point.x, y = point.y, z = point.z;

    // 验证输入
    if (!isfinite(x) || !isfinite(y) || !isfinite(z))
    {
        return 1e10f;
    }

    float A = model.coeffs[0], B = model.coeffs[1], C = model.coeffs[2], D = model.coeffs[3];

    if (!isfinite(A) || !isfinite(B) || !isfinite(C) || !isfinite(D))
    {
        return 1e10f;
    }

    float numerator = fabsf(A * x + B * y + C * z + D);
    float denominator = sqrtf(A * A + B * B + C * C);

    if (denominator < 1e-8f)
    {
        return 1e10f;
    }

    return numerator / denominator;
}

// 批量计算内点数
__global__ void countInliersBatch_Kernel(
    const GPUPoint3f *all_points,
    const int *remaining_indices,
    int num_remaining,
    const uint8_t *valid_mask,  // 新增：有效性掩码
    const GPUPlaneModel *batch_models,
    int batch_size,
    float threshold,
    int *batch_inlier_counts)
{
    int model_id = blockIdx.y;
    if (model_id >= batch_size)
        return;

    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int local_count = 0;

    for (int i = thread_id; i < num_remaining; i += blockDim.x * gridDim.x)
    {
        int global_idx = remaining_indices[i];
        if (valid_mask[global_idx] == 0) continue;  // 跳过已移除的点
        
        GPUPoint3f point = all_points[global_idx];
        float dist = evaluatePlaneDistance(point, batch_models[model_id]);

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

// 找最优模型
__global__ void findBestModel_Kernel(
    const int *batch_inlier_counts,
    int batch_size,
    int *best_index,
    int *best_count)
{
    int thread_id = threadIdx.x;
    int local_best_idx = -1;
    int local_best_count = 0;

    for (int i = thread_id; i < batch_size; i += blockDim.x)
    {
        if (batch_inlier_counts[i] > local_best_count)
        {
            local_best_count = batch_inlier_counts[i];
            local_best_idx = i;
        }
    }

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

// 提取内点索引
__global__ void extractInliers_Kernel(
    const GPUPoint3f *all_points,
    const int *remaining_indices,
    int num_remaining,
    const uint8_t *valid_mask,  // 新增：有效性掩码
    const GPUPlaneModel *model,
    float threshold,
    int *inlier_indices,
    int *inlier_count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_remaining)
        return;

    int global_point_index = remaining_indices[idx];
    if (global_point_index < 0 || valid_mask[global_point_index] == 0)
        return;

    GPUPoint3f point = all_points[global_point_index];
    float dist = evaluatePlaneDistance(point, *model);

    if (dist < threshold)
    {
        int write_pos = atomicAdd(inlier_count, 1);
        if (write_pos < num_remaining)
        {
            inlier_indices[write_pos] = global_point_index;
        }
        else
        {
            atomicAdd(inlier_count, -1);
        }
    }
}

// 移除内点（保留用于兼容性，但将被markMaskKernel替代）
__global__ void removePointsKernel(
    const int *remaining_points,
    int remaining_count,
    const int *inlier_indices, // 不再需要排序
    int inlier_count,
    int *output_points,
    int *output_count)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= remaining_count)
        return;

    int point_id = remaining_points[idx];

    //  改用线性查找，避免排序依赖
    bool is_inlier = false;
    for (int i = 0; i < inlier_count; ++i)
    {
        if (inlier_indices[i] == point_id)
        {
            is_inlier = true;
            break;
        }
    }

    // 如果不是内点，就保留
    if (!is_inlier)
    {
        int write_pos = atomicAdd(output_count, 1);
        output_points[write_pos] = point_id;
    }
}

// 标记掩码内核 - 极速逻辑移除
__global__ void markMaskKernel(
    const int *inlier_indices,
    int inlier_count,
    uint8_t *valid_mask)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < inlier_count) {
        int point_idx = inlier_indices[idx];
        if (point_idx >= 0) {
            valid_mask[point_idx] = 0;  // 标记为已移除
        }
    }
}

// ========================================
// 成员函数实现 - PlaneDetect简化版
// ========================================

template <typename PointT>
void PlaneDetect<PointT>::initializeGPUMemory(int batch_size)
{
    d_batch_models_.resize(batch_size);
    d_batch_inlier_counts_.resize(batch_size);
    d_rand_states_.resize(batch_size);
    d_best_model_index_.resize(1);
    d_best_model_count_.resize(1);
}

template <typename PointT>
void PlaneDetect<PointT>::uploadPointsToGPU(const GPUPoint3f* h_points, size_t point_count)
{
    // 安全检查
    if (point_count > max_points_capacity_)
    {
        std::cerr << "[uploadPointsToGPU] 错误：点云数量 (" << point_count 
                  << ") 超过预分配容量 (" << max_points_capacity_ << ")" << std::endl;
        return;
    }
    
    if (d_points_buffer_ == nullptr || stream_ == nullptr)
    {
        std::cerr << "[uploadPointsToGPU] 错误：GPU显存缓冲区或CUDA流未初始化" << std::endl;
        return;
    }
    
    // 异步拷贝到预分配的GPU缓冲区（使用流隔离，不阻塞其他操作）
    cudaError_t err = cudaMemcpyAsync(d_points_buffer_, h_points, 
                                      point_count * sizeof(GPUPoint3f), 
                                      cudaMemcpyHostToDevice, stream_);
    if (err != cudaSuccess)
    {
        std::cerr << "[uploadPointsToGPU] 错误：GPU异步拷贝失败: " 
                  << cudaGetErrorString(err) << std::endl;
        return;
    }
    
    // 设置当前点数量
    current_point_count_ = point_count;
    
    // 更新remaining_indices（在CPU端完成，不阻塞GPU流）
    d_remaining_indices_.resize(point_count);
    thrust::sequence(d_remaining_indices_.begin(), d_remaining_indices_.end(), 0);
}

template <typename PointT>
void PlaneDetect<PointT>::uploadPointsToGPU(const thrust::device_vector<GPUPoint3f> &h_points)
{
    // 使用预分配的缓冲区，避免每帧malloc/free
    size_t point_count = h_points.size();
    
    // 安全检查
    if (point_count > max_points_capacity_)
    {
        std::cerr << "[uploadPointsToGPU] 错误：点云数量 (" << point_count 
                  << ") 超过预分配容量 (" << max_points_capacity_ << ")" << std::endl;
        return;
    }
    
    if (d_points_buffer_ == nullptr || stream_ == nullptr)
    {
        std::cerr << "[uploadPointsToGPU] 错误：GPU显存缓冲区或CUDA流未初始化" << std::endl;
        return;
    }
    
    // 异步拷贝到预分配的GPU缓冲区（device-to-device拷贝，使用流隔离）
    cudaError_t err = cudaMemcpyAsync(d_points_buffer_, 
                                      thrust::raw_pointer_cast(h_points.data()), 
                                      point_count * sizeof(GPUPoint3f), 
                                      cudaMemcpyDeviceToDevice, stream_);
    if (err != cudaSuccess)
    {
        std::cerr << "[uploadPointsToGPU] 错误：GPU异步拷贝失败: " 
                  << cudaGetErrorString(err) << std::endl;
        return;
    }
    
    // 设置当前点数量
    current_point_count_ = point_count;
    
    // 更新remaining_indices（在CPU端完成，不阻塞GPU流）
    d_remaining_indices_.resize(point_count);
    thrust::sequence(d_remaining_indices_.begin(), d_remaining_indices_.end(), 0);
}

template <typename PointT>
void PlaneDetect<PointT>::launchInitCurandStates(int batch_size)
{
    dim3 block(256);
    dim3 grid((batch_size + block.x - 1) / block.x);

    initCurandStates_Kernel<<<grid, block>>>(
        thrust::raw_pointer_cast(d_rand_states_.data()),
        time(nullptr),
        batch_size);
    cudaDeviceSynchronize();
}

template <typename PointT>
void PlaneDetect<PointT>::launchSampleAndFitPlanes(int batch_size)
{
    if (d_remaining_indices_.size() < 3)
    {
        std::cerr << "[launchSampleAndFitPlanes] 错误：剩余点数不足3个！" << std::endl;
        return;
    }

    dim3 block(256);
    dim3 grid((batch_size + block.x - 1) / block.x);

    sampleAndFitPlanes_Kernel<<<grid, block>>>(
        d_points_buffer_,
        thrust::raw_pointer_cast(d_remaining_indices_.data()),
        static_cast<int>(d_remaining_indices_.size()),
        d_valid_mask_,  // 新增：传入掩码
        thrust::raw_pointer_cast(d_rand_states_.data()),
        batch_size,
        thrust::raw_pointer_cast(d_batch_models_.data()));
    cudaDeviceSynchronize();
}
template <typename PointT>
void PlaneDetect<PointT>::launchCountInliersBatch(int batch_size)
{
    dim3 block(256);
    dim3 grid_x((d_remaining_indices_.size() + block.x - 1) / block.x);
    dim3 grid(grid_x.x, batch_size);

    thrust::fill(d_batch_inlier_counts_.begin(), d_batch_inlier_counts_.end(), 0);

    countInliersBatch_Kernel<<<grid, block>>>(
        d_points_buffer_,
        thrust::raw_pointer_cast(d_remaining_indices_.data()),
        static_cast<int>(d_remaining_indices_.size()),
        d_valid_mask_,  // 新增：传入掩码
        thrust::raw_pointer_cast(d_batch_models_.data()),
        batch_size,
        static_cast<float>(params_.plane_distance_threshold),
        thrust::raw_pointer_cast(d_batch_inlier_counts_.data()));
    cudaDeviceSynchronize();
}

template <typename PointT>
void PlaneDetect<PointT>::launchFindBestModel(int batch_size)
{
    findBestModel_Kernel<<<1, 256>>>(
        thrust::raw_pointer_cast(d_batch_inlier_counts_.data()),
        batch_size,
        thrust::raw_pointer_cast(d_best_model_index_.data()),
        thrust::raw_pointer_cast(d_best_model_count_.data()));
    cudaDeviceSynchronize();
}

template <typename PointT>
void PlaneDetect<PointT>::launchExtractInliers(const GPUPlaneModel *model)
{
    if (d_remaining_indices_.size() == 0)
    {
        current_inlier_count_ = 0;
        return;
    }

    // 安全拷贝模型到GPU
    thrust::device_vector<GPUPlaneModel> d_model_safe(1);
    d_model_safe[0] = *model;

    d_temp_inlier_indices_.resize(d_remaining_indices_.size());
    thrust::device_vector<int> d_inlier_count(1, 0);

    dim3 block(256);
    dim3 grid((d_remaining_indices_.size() + block.x - 1) / block.x);

    extractInliers_Kernel<<<grid, block>>>(
        d_points_buffer_,
        thrust::raw_pointer_cast(d_remaining_indices_.data()),
        static_cast<int>(d_remaining_indices_.size()),
        d_valid_mask_,  // 新增：传入掩码
        thrust::raw_pointer_cast(d_model_safe.data()),
        static_cast<float>(params_.plane_distance_threshold),
        thrust::raw_pointer_cast(d_temp_inlier_indices_.data()),
        thrust::raw_pointer_cast(d_inlier_count.data()));
    cudaDeviceSynchronize();

    // 安全获取内点数量
    int h_count_temp = 0;
    cudaMemcpy(&h_count_temp,
               thrust::raw_pointer_cast(d_inlier_count.data()),
               sizeof(int),
               cudaMemcpyDeviceToHost);

    // 验证内点数量的有效性
    if (h_count_temp < 0 || h_count_temp > static_cast<int>(d_temp_inlier_indices_.size()))
    {
        std::cerr << "[launchExtractInliers] 警告：检测到无效内点数量 " << h_count_temp
                  << "，设置为0" << std::endl;
        current_inlier_count_ = 0;
        d_temp_inlier_indices_.resize(0);
    }
    else
    {
        current_inlier_count_ = h_count_temp;
        d_temp_inlier_indices_.resize(current_inlier_count_);
    }
}

template <typename PointT>
void PlaneDetect<PointT>::getBestModelResults(thrust::host_vector<int> &h_best_index, thrust::host_vector<int> &h_best_count)
{
    h_best_index = d_best_model_index_;
    h_best_count = d_best_model_count_;
}

template <typename PointT>
void PlaneDetect<PointT>::launchRemovePointsKernel()
{
    // 检查内点数量的有效性
    if (current_inlier_count_ <= 0 || current_inlier_count_ > static_cast<int>(d_temp_inlier_indices_.size()))
    {
        return;  // 静默返回，避免日志开销
    }

    if (d_valid_mask_ == nullptr)
    {
        std::cerr << "[launchRemovePointsKernel] 错误：掩码缓冲区未初始化" << std::endl;
        return;
    }

    // 极速标记：只调用标记kernel，无物理搬运
    dim3 block(256);
    dim3 grid((current_inlier_count_ + block.x - 1) / block.x);
    
    markMaskKernel<<<grid, block, 0, 0>>>(
        thrust::raw_pointer_cast(d_temp_inlier_indices_.data()),
        current_inlier_count_,
        d_valid_mask_);
    
    // 严禁cudaDeviceSynchronize - 保持异步
}

// 设备端lambda函数包装器（用于thrust::copy_if）
struct IsValidPoint {
    __device__ bool operator()(uint8_t mask) const {
        return mask == 1;
    }
};

template <typename PointT>
int PlaneDetect<PointT>::downloadCompactPoints(GPUPoint3f* h_out_buffer) const
{
    if (d_points_buffer_ == nullptr || d_valid_mask_ == nullptr || stream_ == nullptr || current_point_count_ == 0)
    {
        return 0;
    }

    // 流同步：确保所有异步操作完成后再下载
    cudaStreamSynchronize(stream_);

    size_t point_count = current_point_count_;
    
    // 使用thrust::copy_if根据掩码提取有效点
    thrust::device_vector<GPUPoint3f> temp_points(point_count);
    
    // 创建device指针包装器
    thrust::device_ptr<GPUPoint3f> d_points_ptr = thrust::device_pointer_cast(d_points_buffer_);
    thrust::device_ptr<uint8_t> d_mask_ptr = thrust::device_pointer_cast(d_valid_mask_);
    
    // 使用copy_if提取有效点（掩码为1的点）- 使用functor替代lambda
    auto end_it = thrust::copy_if(
        d_points_ptr,
        d_points_ptr + point_count,
        d_mask_ptr,
        temp_points.begin(),
        IsValidPoint());
    
    // 计算实际有效点数量
    int valid_count = thrust::distance(temp_points.begin(), end_it);
    
    if (valid_count > 0)
    {
        // 直接下载到CPU缓冲区（同步拷贝，因为需要立即使用数据）
        cudaError_t err = cudaMemcpy(
            h_out_buffer,
            thrust::raw_pointer_cast(temp_points.data()),
            valid_count * sizeof(GPUPoint3f),
            cudaMemcpyDeviceToHost);
        
        if (err != cudaSuccess)
        {
            std::cerr << "[downloadCompactPoints] 错误：GPU拷贝失败: " 
                      << cudaGetErrorString(err) << std::endl;
            return 0;
        }
    }
    
    return valid_count;
}

template <typename PointT>
int PlaneDetect<PointT>::countValidPoints() const
{
    if (d_valid_mask_ == nullptr || current_point_count_ == 0)
    {
        return 0;
    }
    
    size_t point_count = current_point_count_;
    thrust::device_ptr<uint8_t> d_mask_ptr = thrust::device_pointer_cast(d_valid_mask_);
    
    // 使用thrust::count统计掩码为1的点数
    return static_cast<int>(thrust::count(d_mask_ptr, d_mask_ptr + point_count, 1));
}

// 显式模板实例化
template class PlaneDetect<pcl::PointXYZ>;
template class PlaneDetect<pcl::PointXYZI>;  // 主要使用格式
// 可选：保留其他格式
// template class PlaneDetect<pcl::PointXYZRGB>;
// template class PlaneDetect<pcl::PointXYZRGBA>;