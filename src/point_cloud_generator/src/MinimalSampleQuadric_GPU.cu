
// ==========================================================================
// 文件名: MinimalSampleQuadric_GPU.cu
// 职责: 实现 MinimalSampleQuadric_GPU 类的所有功能, 包括GPU加速的RANSAC和模型精炼
// ==========================================================================

// --------------------------------------------------------------------------
// 第零部分: 包含所有必需的头文件
// --------------------------------------------------------------------------
#include "point_cloud_generator/MinimalSampleQuadric_GPU.h" // 包含我们自己的公共头文件

// --- CUDA 和相关库 ---
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cusolverDn.h> // 用于在GPU上进行SVD分解

// --- Thrust库 (CUDA的STL, 极大简化GPU内存管理和并行算法) ---
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>
#include <thrust/gather.h>
#include <thrust/sort.h>
#include <thrust/remove.h>        // for thrust::remove_if
#include <thrust/binary_search.h> // for thrust::binary_search
#include <thrust/functional.h>
#include <thrust/sequence.h>
#include <thrust/count.h>
// --- PCL相关库 ---
#include <pcl/sample_consensus/method_types.h> // for SAC_RANSAC
#include <pcl/sample_consensus/model_types.h>  // for SACMODEL_PLANE
#include <pcl/segmentation/sac_segmentation.h> // for SACSegmentation
#include <pcl/filters/extract_indices.h>       // for ExtractIndices

// --- C++ 标准库和第三方库 ---
#include <iostream>
#include <vector>
#include <random>
#include <Eigen/Dense> // CPU端的Eigen仍然需要，用于处理小规模计算

#include <chrono>

using Clock = std::chrono::high_resolution_clock;

#define TIMEPOINT() Clock::now()
#define DURATION_MS(start, end) \
    std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
// 计时器结构体
struct ScopedTimer
{
    std::string name;
    std::chrono::high_resolution_clock::time_point start;

    ScopedTimer(const std::string &n) : name(n), start(std::chrono::high_resolution_clock::now()) {}
    ~ScopedTimer()
    {
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "[TIMER] " << name << " elapsed: " << elapsed.count() << " s" << std::endl;
    }
};

// ==========================================================================
// 第一部分: 基础设施 (Infrastructure)
// ==========================================================================

/**
 * @brief CUDA API调用的错误检查宏。
 *        这是编写健壮CUDA程序的必备工具。它会检查函数返回值，如果出错则打印详细信息并终止程序。
 */
inline void cudaSafeCall(cudaError_t err, const char *file, int line)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error: %s at %s:%d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}
#define CUDA_SAFE_CALL(err) (cudaSafeCall(err, __FILE__, __LINE__))

/**
 * @brief 一个用于将模型参数从CPU传递到GPU Kernel的简单结构体。
 *        Kernel不能直接使用Eigen::Matrix4f这样的复杂C++对象。
 */
struct QuadricModelData
{
    float coeffs[16];
};
// 定义一个公共可见的 Functor（放在 .cu 文件或命名空间作用域中，不进类体）
struct IsIndexInSortedList
{
    const int *sorted_data;
    size_t num_elements;

    __host__ __device__ bool operator()(int idx) const
    {
        // 手写二分，避免 Thrust binary_search 在 device lambda 里被卡
        int left = 0, right = static_cast<int>(num_elements) - 1;
        while (left <= right)
        {
            int mid = (left + right) / 2;
            int val = sorted_data[mid];
            if (val == idx)
                return true;
            if (val < idx)
                left = mid + 1;
            else
                right = mid - 1;
        }
        return false;
    }
};

__host__ QuadricModelData matrixToQuadricModelData(const Eigen::Matrix4f &mat)
{
    QuadricModelData model;
    // Eigen 默认列主序存储，这里按行主序存进 coeffs
    for (int r = 0; r < 4; ++r)
    {
        for (int c = 0; c < 4; ++c)
        {
            model.coeffs[r * 4 + c] = mat(r, c);
        }
    }
    return model;
}

/**
 * @brief __host__ 函数: 一个在CPU端运行的辅助函数。
 *        将一个10维的参数向量(SVD求解结果)转换为一个4x4的二次曲面Q矩阵。
 * @param q 包含二次曲面10个系数的向量
 * @return 对应的4x4对称Q矩阵
 */
__host__ Eigen::Matrix4f
vectorToQMatrix(const Eigen::VectorXd &q)
{
    // TODO: 在这里实现从10维向量到4x4 Eigen矩阵的转换逻辑。
    //       这个函数的代码和您CPU版本中的完全一样。
    Eigen::Matrix4f Q;
    Q.setZero();

    // 对角线元素
    Q(0, 0) = q(0); // A: x^2 coeff
    Q(1, 1) = q(1); // B: y^2 coeff
    Q(2, 2) = q(2); // C: z^2 coeff

    // 上三角 + 对称
    Q(0, 1) = Q(1, 0) = q(3); // D: xy
    Q(0, 2) = Q(2, 0) = q(4); // E: xz
    Q(1, 2) = Q(2, 1) = q(5); // F: yz

    // 齐次坐标的线性项（x, y, z）
    Q(0, 3) = Q(3, 0) = q(6); // G: x
    Q(1, 3) = Q(3, 1) = q(7); // H: y
    Q(2, 3) = Q(3, 2) = q(8); // I: z

    // 常数项
    Q(3, 3) = q(9); // J

    return Q;
}

// ==========================================================================
// 第二部分: GPU并行计算核心 (The Kernels)
// ==========================================================================

/**
 * @brief KERNEL: RANSAC模型验证。
 *        由成百上千个GPU线程并行执行，每个线程负责一个点。
 *        计算每个点到候选模型的距离，并统计内点的总数。
 * @param all_points          指向存储在GPU上的所有点云数据的指针
 * @param num_points          点云总数
 * @param model               要验证的二次曲面模型
 * @param threshold           判断是否为内点的距离阈值
 * @param out_inlier_count    指向GPU内存中一个整数的指针，用于原子性地累加内点数量
 */
__global__ void countInliers_Kernel(
    const float3 *all_points,
    const int *remaining_indices,
    int num_points,
    QuadricModelData model,
    float threshold,
    int *out_inlier_count)
{
    // TODO: 在这里实现单个线程的工作：
    // 1. 计算当前线程负责的点的全局索引(idx)。
    // 2. 如果索引有效，从all_points[idx]读取点坐标。
    // 3. 计算该点到模型(model)的代数距离。
    // 4. 如果距离的绝对值小于threshold，就使用 atomicAdd(out_inlier_count, 1) 来安全地增加内点计数。
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx > num_points)
        return;
    // 正确的访问方式：
    // 1. 获取当前线程应该处理的那个点的“全局索引”
    int global_point_index = remaining_indices[idx];

    // 2. 使用“全局索引”从 all_points 中获取点坐标
    float x = all_points[global_point_index].x;
    float y = all_points[global_point_index].y;
    float z = all_points[global_point_index].z;

    // 后续的计算逻辑保持不变
    float px = x, py = y, pz = z, pw = 1.0f;

    // tmp = Q * p
    float tx = model.coeffs[0] * px + model.coeffs[1] * py + model.coeffs[2] * pz + model.coeffs[3] * pw;
    float ty = model.coeffs[4] * px + model.coeffs[5] * py + model.coeffs[6] * pz + model.coeffs[7] * pw;
    float tz = model.coeffs[8] * px + model.coeffs[9] * py + model.coeffs[10] * pz + model.coeffs[11] * pw;
    float tw = model.coeffs[12] * px + model.coeffs[13] * py + model.coeffs[14] * pz + model.coeffs[15] * pw;

    float dist = fabsf(px * tx + py * ty + pz * tz + pw * tw);

    if (dist < threshold)
    {
        atomicAdd(out_inlier_count, 1);
    }
}

/**
 * @brief KERNEL: 为SVD求解并行构建矩阵A。
 *        每个线程负责一个内点，计算二次曲面方程对应的10个系数，并填充到矩阵A的对应行。
 * @param inlier_points      指向只包含内点的GPU数组的指针
 * @param num_inliers        内点的总数
 * @param A_matrix           指向将在GPU上构建的巨大矩阵A的指针 (大小为 num_inliers x 10)
 */
__global__ void buildQuadricMatrix_Kernel(
    const float3 *all_points,  // [in] GPU 上全部点的坐标数组
    const int *inlier_indices, // [in] 内点索引数组（全局索引）
    int num_inliers,           // [in] 内点数量
    double *A_matrix           // [out] 大矩阵 A (num_inliers x 10) 列主序
)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_inliers)
        return;
    // 内点的全局 index
    int pt_index = inlier_indices[row];
    float3 p = all_points[pt_index];

    double x = (double)p.x;
    double y = (double)p.y;
    double z = (double)p.z;
    // 10 维系数
    double a0 = x * x;       // x²
    double a1 = y * y;       // y²
    double a2 = z * z;       // z²
    double a3 = 2.0 * x * y; // 2xy
    double a4 = 2.0 * x * z; // 2xz
    double a5 = 2.0 * y * z; // 2yz
    double a6 = 2.0 * x;     // 2x
    double a7 = 2.0 * y;     // 2y
    double a8 = 2.0 * z;     // 2z
    double a9 = 1.0;         // 常数项
    // 按列主序写入
    A_matrix[0 * num_inliers + row] = a0;
    A_matrix[1 * num_inliers + row] = a1;
    A_matrix[2 * num_inliers + row] = a2;
    A_matrix[3 * num_inliers + row] = a3;
    A_matrix[4 * num_inliers + row] = a4;
    A_matrix[5 * num_inliers + row] = a5;
    A_matrix[6 * num_inliers + row] = a6;
    A_matrix[7 * num_inliers + row] = a7;
    A_matrix[8 * num_inliers + row] = a8;
    A_matrix[9 * num_inliers + row] = a9;
}

/**
 * @brief KERNEL: 提取所有内点的索引。
 *        在找到一个最终模型后，用此Kernel来找出所有属于该模型的点的索引。
 * @param all_points              所有点的GPU数组指针
 * @param num_points              总点数
 * @param final_model             最终确定的二次曲面模型
 * @param threshold               内点距离阈值
 * @param out_inlier_indices      一个足够大的GPU数组，用于存储找到的内点的索引
 * @param out_inlier_atomic_counter 一个原子计数器，用于为每个内点分配在out_inlier_indices中的存储位置
 */
__global__ void extractInlierIndices_Kernel(
    const float3 *all_points,      // [in] GPU 上全部点的坐标数组
    const int *remaining_indices,  // [in] 剩余候选点的索引数组
    int num_remaining,             // [in] 剩余点数量
    QuadricModelData final_model,  // [in] 拟合好的 Q 矩阵（GPU-friendly 格式）
    float threshold,               // [in] 内点判定距离阈值
    int *out_inlier_indices,       // [out] 内点索引数组（存的是全局点索引）
    int *out_inlier_atomic_counter // [in/out] 原子计数器（统计内点数量）
)
{

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_remaining)
        return;
    // 根据剩余索引取到真实点的全局 index
    int pt_index = remaining_indices[idx];
    float3 p = all_points[pt_index];
    // 构造齐次坐标 [x, y, z, 1]
    float4 pt_h = make_float4(p.x, p.y, p.z, 1.0f);
    // 计算 pt_h^T * Q * pt_h
    float q_mult[4];
#pragma unroll
    for (int row = 0; row < 4; ++row)
    {
        q_mult[row] = final_model.coeffs[row * 4 + 0] * pt_h.x +
                      final_model.coeffs[row * 4 + 1] * pt_h.y +
                      final_model.coeffs[row * 4 + 2] * pt_h.z +
                      final_model.coeffs[row * 4 + 3] * pt_h.w;
    }
    float dist = fabsf(pt_h.x * q_mult[0] +
                       pt_h.y * q_mult[1] +
                       pt_h.z * q_mult[2] +
                       pt_h.w * q_mult[3]);

    // 如果是内点，则写入输出数组
    if (dist < threshold)
    {
        int write_pos = atomicAdd(out_inlier_atomic_counter, 1);
        out_inlier_indices[write_pos] = pt_index; // 存全局索引
    }
}

// ==========================================================================
// 第三部分: PIMPL实现类的完整定义
// ==========================================================================
class MinimalSampleQuadric_GPU::MinimalSampleQuadric_GPU_Impl
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    using PointT = pcl::PointXYZ;
    using PointCloud = pcl::PointCloud<PointT>;
    using PointCloudPtr = PointCloud::Ptr;

public:
    // --- 公有方法 (供外层桥接类调用) ---
    MinimalSampleQuadric_GPU_Impl(const DetectorParams &params);
    ~MinimalSampleQuadric_GPU_Impl();
    bool processCloud(const PointCloudConstPtr &input_cloud);
    const std::vector<DetectedPrimitive, Eigen::aligned_allocator<DetectedPrimitive>> &getDetectedPrimitives() const;
    PointCloudPtr getFinalCloud() const;

    void removeFoundPoints(const std::vector<int> &indices_to_remove);

private:
    // --- 私有核心算法流程 ---
    void uploadCloudToGPU(const PointCloudConstPtr &cloud);
    void findQuadrics_GPU();
    Eigen::Matrix4f refineModel_SVD_GPU(const Eigen::Matrix4f &initial_model, std::vector<int> &out_inlier_indices);
    void detectPlanes(PointCloudPtr &remain_cloud);
    Eigen::Matrix4f fitQuadricFromIndices(const std::vector<int> &sample_indices);
    void extractInliers_GPU(const QuadricModelData &modelData,
                            float distance_threshold,
                            std::vector<int> &out_indices);
    // --- 私有成员变量 (类的状态和资源) ---
    const DetectorParams &params_; // 存储用户配置参数的引用

    // GPU数据: 使用Thrust的device_vector来自动管理GPU内存
    thrust::device_vector<float3> d_all_points_;     // 存储在GPU上的原始点云(x,y,z)
    thrust::device_vector<int> d_remaining_indices_; // 存储当前还未被识别的点的索引，这是算法循环的核心

    // GPU计算资源
    cusolverDnHandle_t cusolver_handle_; // cuSOLVER库的句柄，用于SVD计算

    // CPU资源
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    std::vector<DetectedPrimitive, Eigen::aligned_allocator<DetectedPrimitive>> detected_primitives_;
    // 存储所有检测结果
    std::mt19937 rand_engine_; // 用于RANSAC采样的随机数生成器
};

// ==========================================================================
// 第四部分: 核心算法流程的实现 (空函数体)
// ==========================================================================

MinimalSampleQuadric_GPU::MinimalSampleQuadric_GPU_Impl::MinimalSampleQuadric_GPU_Impl(const DetectorParams &params)
    : params_(params), rand_engine_(std::random_device{}())
{
    std::cout << "GPU 实现类 (Impl) 已构造。" << std::endl;
    // TODO: 在这里调用 cusolverDnCreate(&cusolver_handle_) 来初始化cuSOLVER。
    //       并使用 CUDA_SAFE_CALL 宏进行错误检查。
    cusolverStatus_t status = cusolverDnCreate(&cusolver_handle_);
    if (status != CUSOLVER_STATUS_SUCCESS)
    {
        std::cerr << "cuSolver 初始化失败: " << status << std::endl;
        exit(EXIT_FAILURE);
    }
}

MinimalSampleQuadric_GPU::MinimalSampleQuadric_GPU_Impl::~MinimalSampleQuadric_GPU_Impl()
{
    std::cout << "[Impl] GPU 实现类析构开始..." << std::endl;

    try
    {
        // 先清理 GPU 内存
        std::cout << "[Impl] 析构中清理 GPU device_vectors..." << std::endl;
        d_remaining_indices_.clear();
        d_all_points_.clear();
        std::cout << "[Impl] GPU device_vectors 清理完成" << std::endl;

        // 清理 detected_primitives_
        std::cout << "[Impl] 清理 detected_primitives_..." << std::endl;
        detected_primitives_.clear();
        std::cout << "[Impl] detected_primitives_ 清理完成" << std::endl;

        // 最后清理 cuSolver 句柄
        if (cusolver_handle_)
        {
            std::cout << "[Impl] 准备销毁 cuSolver 句柄..." << std::endl;
            cusolverStatus_t status = cusolverDnDestroy(cusolver_handle_);
            if (status != CUSOLVER_STATUS_SUCCESS)
            {
                std::cerr << "销毁 cuSolver 句柄失败" << std::endl;
            }
            else
            {
                std::cout << "[Impl] cuSolver 句柄销毁成功" << std::endl;
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "[ERROR] 析构过程中异常: " << e.what() << std::endl;
    }

    std::cout << "[Impl] GPU 实现类析构完成" << std::endl;
}

void MinimalSampleQuadric_GPU::MinimalSampleQuadric_GPU_Impl::uploadCloudToGPU(const PointCloudConstPtr &cloud)
{
    // TODO: 实现将PCL点云从CPU拷贝到GPU的 d_all_points_ 中的逻辑。
    // 步骤:
    // 1. 创建一个 thrust::host_vector<float3>。
    // 2. 遍历PCL点云，填充这个host_vector。
    // 3. 将host_vector直接赋值给 d_all_points_，Thrust会自动处理CPU到GPU的拷贝。
    // 4. 初始化 d_remaining_indices_，使其包含从 0 到 num_points-1 的所有索引。
    auto t_start = TIMEPOINT();
    thrust::host_vector<float3> h_points;
    h_points.reserve(cloud->size());
    for (const auto &pt : cloud->points)
    {
        float3 p;
        p.x = pt.x;
        p.y = pt.y;
        p.z = pt.z;
        h_points.push_back(p);
    }
    d_all_points_ = h_points;
    d_remaining_indices_.resize(cloud->size());
    thrust::sequence(d_remaining_indices_.begin(), d_remaining_indices_.end(), 0);
    std::cout << "[Impl] 点云上传至 GPU, 点数: " << cloud->size() << std::endl;

    auto t_end = TIMEPOINT();
    std::cout << "[TIMER] Upload to GPU: " << DURATION_MS(t_start, t_end) << " ms" << std::endl;
}

void MinimalSampleQuadric_GPU::MinimalSampleQuadric_GPU_Impl::findQuadrics_GPU()
{

    // TODO: 实现RANSAC主循环。
    // while (剩余点数足够) {
    //     // 1. RANSAC迭代，找出最佳候选模型
    //     for (i=0 to max_iterations) {
    //         a. CPU: 从 d_remaining_indices_ 中随机采样9个点的索引。
    //         b. CPU: 根据索引从GPU取回这9个点的数据。
    //         c. CPU: 使用Eigen的SVD求解一个候选模型。
    //         d. GPU: 调用 countInliers_Kernel 在所有剩余点上验证该模型，得到内点数。
    //         e. CPU: 更新最佳模型和最佳内点数。
    //     }
    //
    //     // 2. 如果找到了一个不错的模型，就用所有内点去精炼它
    //     if (best_inlier_count > threshold) {
    //         std::vector<int> final_inlier_indices;
    //         Eigen::Matrix4f refined_model = refineModel_SVD_GPU(best_model, final_inlier_indices);
    //
    //         // 3. 保存结果，并从剩余点云中移除这些内点
    //         // ... 保存 refined_model 和内点到 detected_primitives_ ...
    //         removeFoundPoints(final_inlier_indices);
    //     } else {
    //         // 找不到更多模型了，退出循环
    //         break;
    //     }
    // }
    // =======================
    // Step 1: RANSAC 主循环
    // =======================
    std::cout << "[findQuadrics_GPU] 启动 GPU RANSAC 二次曲面检测" << std::endl;
    size_t num_remaining = d_remaining_indices_.size();
    int bad_model_rounds = 0;
    const int max_bad_model_rounds = 3; // 允许连续出现 3 个小模型

    while (num_remaining >= static_cast<size_t>(params_.min_remaining_points_percentage * d_all_points_.size()))
    {
        // LO-RANSAC三个参数
        double desired_prob = 0.99;                               // 目标成功概率
        int sample_size = 9;                                      // 二次曲面拟合需要的点
        int required_iterations = params_.quadric_max_iterations; // 初始要求迭代数

        int best_inlier_count = 0;
        Eigen::Matrix4f best_model = Eigen::Matrix4f::Identity();

        // 预分配vectors以避免频繁内存分配
        thrust::device_vector<int> d_sample_indices(9);
        thrust::device_vector<float3> d_sample_points(9);
        thrust::host_vector<float3> h_sample_points(9);
        thrust::host_vector<int> h_remaining_indices;
        std::vector<int> h_sample_indices_vec;
        h_sample_indices_vec.reserve(9);
        std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> sample_points;
        sample_points.reserve(9);

        for (int iter = 0; iter < required_iterations; ++iter)
        {
            // 重用预分配的vectors
            h_remaining_indices = d_remaining_indices_;
            h_sample_indices_vec.clear();
            std::uniform_int_distribution<int> dist(0, num_remaining - 1);
            for (int s = 0; s < 9; ++s)
            {
                h_sample_indices_vec.push_back(h_remaining_indices[dist(rand_engine_)]);
            }

            // b. GPU: 使用 thrust::gather 高效地一次性获取这9个点的数据
            //    b.1. 将采样到的索引从CPU传回GPU
            thrust::copy(h_sample_indices_vec.begin(), h_sample_indices_vec.end(), d_sample_indices.begin());

            //    b.3. 执行 gather 操作！GPU会并行地根据索引从 d_all_points_ 中取出数据
            thrust::gather(
                d_sample_indices.begin(),
                d_sample_indices.end(),
                d_all_points_.begin(),
                d_sample_points.begin());

            //    b.4. 将这9个点的数据一次性从GPU拷回CPU
            thrust::copy(d_sample_points.begin(), d_sample_points.end(), h_sample_points.begin());

            // c. CPU: 将拷回的数据填充到Eigen向量中
            sample_points.clear();
            for (const auto &p : h_sample_points)
            {
                sample_points.emplace_back(p.x, p.y, p.z);
            }

            // ② 拟合模型（SVD）
            Eigen::MatrixXd A(9, 10);
            for (int i = 0; i < 9; ++i)
            {
                double x = sample_points[i](0);
                double y = sample_points[i](1);
                double z = sample_points[i](2);
                A(i, 0) = x * x;
                A(i, 1) = y * y;
                A(i, 2) = z * z;
                A(i, 3) = 2 * x * y;
                A(i, 4) = 2 * x * z;
                A(i, 5) = 2 * y * z;
                A(i, 6) = 2 * x;
                A(i, 7) = 2 * y;
                A(i, 8) = 2 * z;
                A(i, 9) = 1.0;
            }
            Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
            Eigen::VectorXd q = svd.matrixV().col(9);
            Eigen::Matrix4f modelQ = vectorToQMatrix(q);
            // if (iter % 100 == 0)
            // {
            //     std::cout << "[DBG] modelQ from SVD:\n"
            //               << modelQ << std::endl;
            // }

            // ③ 模型转 GPU 可用格式
            QuadricModelData modelData = matrixToQuadricModelData(modelQ);

            // ④ GPU 并行内点评估
            thrust::device_vector<int> d_inlier_count(1, 0); // 计数器在GPU上

            dim3 block(256);
            dim3 grid((num_remaining + block.x - 1) / block.x);
            countInliers_Kernel<<<grid, block>>>(thrust::raw_pointer_cast(d_all_points_.data()),
                                                 thrust::raw_pointer_cast(d_remaining_indices_.data()),
                                                 num_remaining,
                                                 modelData,
                                                 params_.quadric_distance_threshold,
                                                 thrust::raw_pointer_cast(d_inlier_count.data()));
            cudaDeviceSynchronize();
            int h_inlier_count = d_inlier_count[0]; // 从GPU取回（隐式拷贝）

            // === LO-RANSAC 更新逻辑 ===
            if (h_inlier_count > best_inlier_count)
            {
                best_inlier_count = h_inlier_count;
                best_model = modelQ;

                double inlier_ratio = static_cast<double>(best_inlier_count) / static_cast<double>(d_all_points_.size());
                if (inlier_ratio > 0 && inlier_ratio < 1.0)
                {
                    int new_required = 1.5 * static_cast<int>(std::ceil(std::log(1 - desired_prob) / std::log(1 - std::pow(inlier_ratio, sample_size))));
                    if (new_required < required_iterations)
                        required_iterations = std::max(new_required, iter + 1); // 确保不会比当前迭代还小
                }

                // === 局部优化可选版本 ===
                if (params_.enable_local_optimization && inlier_ratio > params_.lo_min_inlier_ratio)
                {
                    // 从当前最佳内点集采更多点重拟合（示例性伪代码）
                    std::vector<int> current_inlier_indices;
                    extractInliers_GPU(modelData, params_.quadric_distance_threshold, current_inlier_indices);

                    if (current_inlier_indices.size() >= sample_size)
                    {
                        // 随机更大样本数，例如 15 个
                        std::vector<int> lo_sample_indices;
                        lo_sample_indices.reserve(15);
                        std::uniform_int_distribution<int> dist_lo(0, current_inlier_indices.size() - 1);
                        for (int s = 0; s < 15; ++s)
                            lo_sample_indices.push_back(current_inlier_indices[dist_lo(rand_engine_)]);

                        // 拟合新的 Q 并重新评估
                        Eigen::Matrix4f lo_modelQ = fitQuadricFromIndices(lo_sample_indices);
                        QuadricModelData lo_modelData = matrixToQuadricModelData(lo_modelQ);
                        thrust::device_vector<int> d_lo_inlier_count(1, 0);
                        countInliers_Kernel<<<grid, block>>>(thrust::raw_pointer_cast(d_all_points_.data()),
                                                             thrust::raw_pointer_cast(d_remaining_indices_.data()),
                                                             num_remaining,
                                                             lo_modelData,
                                                             params_.quadric_distance_threshold,
                                                             thrust::raw_pointer_cast(d_lo_inlier_count.data()));
                        cudaDeviceSynchronize();
                        int lo_inlier_count = d_lo_inlier_count[0];
                        if (lo_inlier_count > best_inlier_count)
                        {
                            best_inlier_count = lo_inlier_count;
                            best_model = lo_modelQ;
                        }
                    }
                }
            }

            // === 提前结束判断 ===
            if (iter >= required_iterations - 1)
            {
                if (params_.verbosity > 0)
                    std::cout << "[LO-RANSAC] Early exit at iter " << iter + 1
                              << " / " << params_.quadric_max_iterations
                              << " (required = " << required_iterations << ")\n";
                break;
            }
            if (iter % 200 == 0)
            {
                std::cout << "[DBG] Iter " << iter
                          << " h_inlier_count = " << h_inlier_count
                          << "  best_so_far = " << best_inlier_count
                          << std::endl;
            }
        }
        // =======================
        // Step 2: 检查模型有效性
        // =======================
        double inlier_percentage_remaining = static_cast<double>(best_inlier_count) / num_remaining;
        bool is_small_model =
            (inlier_percentage_remaining < params_.min_quadric_inlier_percentage ||
             best_inlier_count < params_.min_quadric_inlier_count_absolute);

        if (is_small_model)
        {
            bad_model_rounds++;
            // std::cout << "[findQuadrics_GPU] 小模型("
            //           << inlier_percentage_remaining * 100 << "%, "
            //           << best_inlier_count << "点)，连续第 "
            //           << bad_model_rounds << " 次" << std::endl;

            if (bad_model_rounds >= max_bad_model_rounds * 5)
            {
                std::cout << "[findQuadrics_GPU] 已连续 "
                          << bad_model_rounds << " 次小模型，结束检测" << std::endl;
                break; // 结束外层 while
            }

            continue; // 跳过保存，直接下一轮 while
        }

        bad_model_rounds = 0;

        // =======================
        // Step 3: 精炼模型
        // =======================
        std::cout << "debug1" << std::endl;
        std::vector<int> final_inlier_indices;
        std::cout << "debug2" << std::endl;

        Eigen::Matrix4f refined_model = refineModel_SVD_GPU(best_model, final_inlier_indices);

        std::cout << "debug3" << std::endl;
        // 保存结果 - 使用 emplace_back 避免拷贝构造问题
        std::cout << "debug4" << std::endl;

        // 使用 emplace_back 直接在容器中构造对象，避免拷贝
        detected_primitives_.emplace_back();
        DetectedPrimitive &prim = detected_primitives_.back();

        std::cout << "debug4.5" << std::endl;

        prim.type = "quadric";
        prim.model_coefficients = refined_model;

        // 安全地初始化 inliers 并填充内点数据
        prim.inliers.reset(new pcl::PointCloud<pcl::PointXYZ>());
        prim.inliers->reserve(final_inlier_indices.size());
        
        // 从 GPU 获取内点数据并填充到 inliers 中
        if (!final_inlier_indices.empty()) {
            thrust::device_vector<int> d_inlier_indices(final_inlier_indices.begin(), final_inlier_indices.end());
            thrust::device_vector<float3> d_inlier_points(final_inlier_indices.size());
            
            // 使用 gather 从 d_all_points_ 中获取内点
            thrust::gather(d_inlier_indices.begin(), d_inlier_indices.end(),
                          d_all_points_.begin(), d_inlier_points.begin());
            
            // 拷贝到 CPU
            thrust::host_vector<float3> h_inlier_points = d_inlier_points;
            
            // 填充到 PCL 点云中
            prim.inliers->resize(h_inlier_points.size());
            for (size_t i = 0; i < h_inlier_points.size(); ++i) {
                (*prim.inliers)[i].x = h_inlier_points[i].x;
                (*prim.inliers)[i].y = h_inlier_points[i].y;
                (*prim.inliers)[i].z = h_inlier_points[i].z;
            }
        }

        std::cout << "debug5 - inliers 已正确填充，大小: " << prim.inliers->size() << std::endl;
        std::cout << "debug6" << std::endl;

        // =======================
        // Step 4: 从剩余索引中移除内点
        // =======================
        removeFoundPoints(final_inlier_indices);
        std::cout << "debug7" << std::endl;

        num_remaining = d_remaining_indices_.size();
        std::cout << "debug7.5" << std::endl;

        std::cout << "[findQuadrics_GPU] Final best model (before exit), inliers = "
                  << best_inlier_count << "\n"
                  << best_model << std::endl;
    }
    std::cout << "debug8" << std::endl;
}

Eigen::Matrix4f MinimalSampleQuadric_GPU::MinimalSampleQuadric_GPU_Impl::refineModel_SVD_GPU(const Eigen::Matrix4f &initial_model, std::vector<int> &out_inlier_indices)
{
    // TODO: 实现GPU加速的模型精炼。
    // 1. 调用 extractInlierIndices_Kernel 找出 initial_model 的所有内点索引。
    // =========================
    // Step 1: 提取所有内点 (GPU)
    // =========================
    QuadricModelData modelData = matrixToQuadricModelData(initial_model);
    thrust::device_vector<int> d_inlier_indices(d_remaining_indices_.size());
    thrust::device_vector<int> d_inlier_count(1, 0);

    {
        dim3 block(256);
        dim3 grid((d_remaining_indices_.size() + block.x - 1) / block.x);

        extractInlierIndices_Kernel<<<grid, block>>>(
            thrust::raw_pointer_cast(d_all_points_.data()),
            thrust::raw_pointer_cast(d_remaining_indices_.data()),
            d_remaining_indices_.size(),
            modelData,
            params_.quadric_distance_threshold,
            thrust::raw_pointer_cast(d_inlier_indices.data()),
            thrust::raw_pointer_cast(d_inlier_count.data()));
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
    }
    int h_inlier_count = d_inlier_count[0]; // 隐式 GPU→CPU 拷贝
    if (h_inlier_count < 10)
    {
        std::cerr << "[refineModel_SVD_GPU] 内点数太少，返回原模型" << std::endl;
        out_inlier_indices.clear();
        return initial_model;
    }
    // 2. 根据内点数量，在GPU上分配一个巨大的矩阵A (thrust::device_vector<double> d_A)。
    // 这里矩阵是 m × 10，列主序存储
    int m = h_inlier_count;
    int n = 10;
    thrust::device_vector<double> d_A(m * n, 0.0);

    {
        dim3 block(256);
        dim3 grid((m + block.x - 1) / block.x);

        // 3. 调用 buildQuadricMatrix_Kernel，让GPU并行填充这个矩阵 d_A。
        buildQuadricMatrix_Kernel<<<grid, block>>>(
            thrust::raw_pointer_cast(d_all_points_.data()),
            thrust::raw_pointer_cast(d_inlier_indices.data()),
            m,
            thrust::raw_pointer_cast(d_A.data()));
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
    }

    // 4. 调用 cusolverDnDgesvd() 函数，对 d_A 进行SVD分解。这是一个核心调用。
    // 注意: cuSolver 使用列主序存储，m 行 n 列，lda = m
    thrust::device_vector<double> d_S(n, 0.0); // 奇异值
    // 只需要右奇异向量V，不需要完整的U矩阵来节省内存
    thrust::device_vector<double> d_U(1, 0.0);     // 最小分配，因为我们不需要U
    thrust::device_vector<double> d_V(n * n, 0.0); // 右奇异向量
    int ldu = 1;                                   // 由于不计算U，设为最小值
    int ldv = n;
    int lda = m;

    // cuSolver 工作区大小
    int work_size = 0;
    cusolverDnDgesvd_bufferSize(cusolver_handle_, m, n, &work_size);
    thrust::device_vector<double> d_work(work_size);
    thrust::device_vector<int> d_info(1);

    signed char jobu = 'N';  // 不计算 U 来节省内存
    signed char jobvt = 'A'; // 全部 V^T
    cusolverStatus_t status = cusolverDnDgesvd(
        cusolver_handle_,
        jobu, jobvt,
        m, n,
        thrust::raw_pointer_cast(d_A.data()), lda,
        thrust::raw_pointer_cast(d_S.data()),
        thrust::raw_pointer_cast(d_U.data()), ldu,
        thrust::raw_pointer_cast(d_V.data()), ldv,
        thrust::raw_pointer_cast(d_work.data()), work_size,
        nullptr, // rwork（双精度不需要）
        thrust::raw_pointer_cast(d_info.data()));
    if (status != CUSOLVER_STATUS_SUCCESS)
    {
        throw std::runtime_error("cuSolver SVD failed in refineModel_SVD_GPU");
    }
    // 5. 从cuSOLVER的结果中，拷贝出最后一个奇异向量(代表最优解)到CPU。
    thrust::host_vector<double> h_V = d_V; // GPU→CPU
    Eigen::VectorXd q_vec(n);
    for (int i = 0; i < n; ++i)
    {
        q_vec(i) = h_V[i + (n - 1) * n];
    }

    // 6. 使用 vectorToQMatrix 将这个向量转换为新的、更精确的 Eigen::Matrix4f 模型。
    if (fabs(q_vec(9)) > 1e-8)
    {
        q_vec /= q_vec(9); // 固定最后一项为1
    }
    else
    {
        q_vec.normalize(); // fall back
    }
    Eigen::Matrix4f refined_model = vectorToQMatrix(q_vec).cast<float>();

    // 7. 将找到的内点索引(out_inlier_indices)和精炼后的模型返回。
    out_inlier_indices.resize(h_inlier_count);
    thrust::copy(d_inlier_indices.begin(), d_inlier_indices.begin() + h_inlier_count,
                 out_inlier_indices.begin());

    return refined_model;
}

void MinimalSampleQuadric_GPU::MinimalSampleQuadric_GPU_Impl::removeFoundPoints(const std::vector<int> &indices_to_remove)
{
    // TODO: 实现从 d_remaining_indices_ 中移除已找到的点的索引。
    //       这在Thrust中通常使用 thrust::remove_if 或 thrust::copy_if 配合自定义的functor来高效完成。
    if (indices_to_remove.empty())
        return;
    std::cout << "debug6.1" << std::endl;

    thrust::device_vector<int> d_remove(indices_to_remove.begin(), indices_to_remove.end());
    std::cout << "debug6.2" << std::endl;

    thrust::sort(d_remove.begin(), d_remove.end());
    std::cout << "debug6.3" << std::endl;

    IsIndexInSortedList predicate{
        thrust::raw_pointer_cast(d_remove.data()),
        d_remove.size()};
    std::cout << "debug6.4" << std::endl;

    auto end_it = thrust::remove_if(
        d_remaining_indices_.begin(),
        d_remaining_indices_.end(),
        predicate);
    std::cout << "debug6.5" << std::endl;

    d_remaining_indices_.erase(end_it, d_remaining_indices_.end());
}

void MinimalSampleQuadric_GPU::MinimalSampleQuadric_GPU_Impl::detectPlanes(PointCloudPtr &remain_cloud)
{
    // 如果初始点云为空或点数不足，则直接返回
    if (!remain_cloud || remain_cloud->size() < 3)
        return;

    std::cout << "---Starting Plane Detection---" << std::endl;

    // 创建一个临时点云指针，用于在循环中操作，避免直接修改传入的引用
    // 使用 pcl::copyPointCloud 而不是拷贝构造函数来避免 boost::shared_ptr 问题
    PointCloudPtr current_cloud(new PointCloud());
    pcl::copyPointCloud(*remain_cloud, *current_cloud);

    pcl::SACSegmentation<PointT> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE);
    seg.setMethodType(pcl::SAC_RANSAC);

    // 使用 DetectorParams 中的参数
    seg.setMaxIterations(params_.plane_max_iterations);
    seg.setDistanceThreshold(params_.plane_distance_threshold);

    pcl::PointIndices::Ptr inlier_indices(new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::ExtractIndices<PointT> extract;

    while (true)
    {
        size_t current_point_count = current_cloud->size();

        if (current_point_count < 3)
        {
            std::cout << "Remaining points (" << current_point_count
                      << ") are too few to form a plane. Stopping." << std::endl;
            break;
        }

        seg.setInputCloud(current_cloud);
        seg.segment(*inlier_indices, *coefficients);

        if (inlier_indices->indices.empty())
        {
            std::cout << "RANSAC could not find any planar model in the remaining "
                      << current_point_count << " points. Stopping." << std::endl;
            break;
        }

        double inlier_percentage = static_cast<double>(inlier_indices->indices.size()) / current_point_count;
        if (inlier_percentage < params_.min_plane_inlier_percentage)
        {
            std::cout << "Found a plane with " << inlier_indices->indices.size()
                      << " inliers (below threshold of "
                      << params_.min_plane_inlier_percentage * 100
                      << "%). Stopping plane detection." << std::endl;
            break;
        }

        std::cout << "Plane detected with " << inlier_indices->indices.size() << " inliers." << std::endl;

        DetectedPrimitive plane_primitive;
        plane_primitive.type = "plane";

        // 将平面系数存到 4x4 矩阵的最后一列
        plane_primitive.model_coefficients.setZero();
        plane_primitive.model_coefficients(0, 3) = coefficients->values[0]; // A
        plane_primitive.model_coefficients(1, 3) = coefficients->values[1]; // B
        plane_primitive.model_coefficients(2, 3) = coefficients->values[2]; // C
        plane_primitive.model_coefficients(3, 3) = coefficients->values[3]; // D

        // 提取内点为 PointXYZ
        pcl::PointCloud<pcl::PointXYZ>::Ptr inliers_xyz(new pcl::PointCloud<pcl::PointXYZ>());
        extract.setInputCloud(current_cloud);
        extract.setIndices(inlier_indices);
        extract.setNegative(false);
        extract.filter(*inliers_xyz);

        // 使用安全的 copyPointCloud 而不是直接拷贝构造来避免 boost::shared_ptr 问题
        pcl::copyPointCloud(*inliers_xyz, *(plane_primitive.inliers));

        detected_primitives_.push_back(plane_primitive);

        // 移除内点
        extract.setNegative(true);
        PointCloudPtr remaining_points(new PointCloud());
        extract.filter(*remaining_points);

        // 使用安全的方法交换点云，避免拷贝构造
        current_cloud.swap(remaining_points);
    }

    remain_cloud.swap(current_cloud);

    std::cout << "--- Plane Detection Finished. "
              << remain_cloud->size() << " points remain. ---" << std::endl;
}

bool MinimalSampleQuadric_GPU::MinimalSampleQuadric_GPU_Impl::processCloud(const PointCloudConstPtr &input_cloud)
{
    if (!input_cloud || input_cloud->empty())
        return false;

    // 暂时注释掉预处理和平面检测，直接进行二次曲面检测
    // ==== Step 1: 预处理 ====
    auto t1_start = TIMEPOINT();
    std::cout << "[DEBUG GPU] Creating preprocessor" << std::endl;
    
    PointCloudPreprocessor preprocessor(params_.preprocessing);
    std::cout << "[DEBUG GPU] Preprocessor created" << std::endl;

    // 2. 执行预处理 - 使用安全的拷贝方法和分离的输入输出指针
    std::cout << "[DEBUG GPU] Creating input and output clouds" << std::endl;
    PointCloudPtr input_for_preprocessing(new PointCloud());
    PointCloudPtr processed_cloud(new PointCloud());
    std::cout << "[DEBUG GPU] Input cloud at: " << input_for_preprocessing.get() 
              << ", Output cloud at: " << processed_cloud.get() << std::endl;
    
    std::cout << "[DEBUG GPU] Copying input cloud (size: " << input_cloud->size() << ")" << std::endl;
    pcl::copyPointCloud(*input_cloud, *input_for_preprocessing); // 使用 copyPointCloud 而非拷贝构造
    std::cout << "[DEBUG GPU] Input copied, calling preprocessor.process" << std::endl;

    if (!preprocessor.process(input_for_preprocessing, processed_cloud) ||
        processed_cloud->empty())
    {
        std::cout << "[DEBUG GPU] Preprocessing failed or empty result" << std::endl;
        return false;
    }
    
    std::cout << "[DEBUG GPU] Preprocessing successful" << std::endl;
    auto t1_end = TIMEPOINT();
    std::cout << "[TIMER] Preprocessing: "
              << DURATION_MS(t1_start, t1_end) << " ms" << std::endl;

    // // ==== Step 2: 平面检测 ====
    // auto t2_start = TIMEPOINT();
    // detected_primitives_.clear();
    // detectPlanes(processed_cloud);
    // auto t2_end = TIMEPOINT();
    // std::cout << "[TIMER] Plane detection: "
    //           << DURATION_MS(t2_start, t2_end) << " ms" << std::endl;

    // // 直接使用输入点云进行二次曲面检测
    // PointCloudPtr processed_cloud(new PointCloud());
    // pcl::copyPointCloud(*input_cloud, *processed_cloud); // 安全拷贝输入点云

    detected_primitives_.clear(); // 清空之前的检测结果

    // ==== Step 3: 上传到GPU ====
    auto t3_start = TIMEPOINT();
    uploadCloudToGPU(processed_cloud);
    auto t3_end = TIMEPOINT();
    std::cout << "[TIMER] Upload to GPU: "
              << DURATION_MS(t3_start, t3_end) << " ms" << std::endl;

    // ==== Step 4: 二次曲面检测 ====
    auto t4_start = TIMEPOINT();
    findQuadrics_GPU();
    auto t4_end = TIMEPOINT();
    std::cout << "[TIMER] Quadric detection: "
              << DURATION_MS(t4_start, t4_end) << " ms" << std::endl;
    std::cout << "debugf" << std::endl;
    return true;
}

const std::vector<DetectedPrimitive, Eigen::aligned_allocator<DetectedPrimitive>> &
MinimalSampleQuadric_GPU::MinimalSampleQuadric_GPU_Impl::getDetectedPrimitives() const
{
    return detected_primitives_;
}

MinimalSampleQuadric_GPU::PointCloudPtr MinimalSampleQuadric_GPU::MinimalSampleQuadric_GPU_Impl::getFinalCloud() const
{
    // TODO: 实现将GPU上的 d_remaining_indices_ 和 d_all_points_ 下载回CPU，并构建成PCL点云返回。
    PointCloudPtr final_cloud(new PointCloud());

    if (d_all_points_.empty() || d_remaining_indices_.empty())
    {
        std::cout << "[Impl] GPU 上无剩余点，返回空点云" << std::endl;
        return final_cloud;
    }

    // ====== 1. 先在 GPU 上 gather 到 device_vector ======
    thrust::device_vector<float3> d_final_points(d_remaining_indices_.size());
    thrust::gather(
        d_remaining_indices_.begin(),
        d_remaining_indices_.end(),
        d_all_points_.begin(),
        d_final_points.begin());

    // ====== 2. 一次性拷回 CPU ======
    thrust::host_vector<float3> h_final_points = d_final_points;

    // ====== 3. 转换成 PCL 格式 ======
    final_cloud->resize(h_final_points.size());
    for (size_t i = 0; i < h_final_points.size(); ++i)
    {
        (*final_cloud)[i].x = h_final_points[i].x;
        (*final_cloud)[i].y = h_final_points[i].y;
        (*final_cloud)[i].z = h_final_points[i].z;
        // 如果 PointT 有法线字段，还可以在这里赋 nx, ny, nz
    }

    std::cout << "[Impl] 下载剩余点云完成，共 " << final_cloud->size() << " 个点" << std::endl;
    return final_cloud;
}

Eigen::Matrix4f MinimalSampleQuadric_GPU::MinimalSampleQuadric_GPU_Impl::
    fitQuadricFromIndices(const std::vector<int> &sample_indices)
{
    size_t m = sample_indices.size();
    Eigen::MatrixXd A(m, 10);

    // 一次性从GPU取回样本点
    thrust::device_vector<int> d_sample_indices(sample_indices.begin(), sample_indices.end());
    thrust::device_vector<float3> d_sample_points(m);
    thrust::gather(d_sample_indices.begin(),
                   d_sample_indices.end(),
                   d_all_points_.begin(),
                   d_sample_points.begin());
    thrust::host_vector<float3> h_sample_points = d_sample_points;

    for (size_t i = 0; i < m; ++i)
    {
        double x = h_sample_points[i].x;
        double y = h_sample_points[i].y;
        double z = h_sample_points[i].z;
        A(i, 0) = x * x;
        A(i, 1) = y * y;
        A(i, 2) = z * z;
        A(i, 3) = 2 * x * y;
        A(i, 4) = 2 * x * z;
        A(i, 5) = 2 * y * z;
        A(i, 6) = 2 * x;
        A(i, 7) = 2 * y;
        A(i, 8) = 2 * z;
        A(i, 9) = 1.0;
    }

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
    Eigen::VectorXd q = svd.matrixV().col(9);
    return vectorToQMatrix(q);
}

void MinimalSampleQuadric_GPU::MinimalSampleQuadric_GPU_Impl::
    extractInliers_GPU(const QuadricModelData &modelData,
                       float distance_threshold,
                       std::vector<int> &out_indices)
{
    thrust::device_vector<int> d_inlier_indices(d_remaining_indices_.size(), -1);
    thrust::device_vector<int> d_inlier_count(1, 0);

    dim3 block(256);
    dim3 grid((d_remaining_indices_.size() + block.x - 1) / block.x);
    extractInlierIndices_Kernel<<<grid, block>>>(
        thrust::raw_pointer_cast(d_all_points_.data()),
        thrust::raw_pointer_cast(d_remaining_indices_.data()),
        d_remaining_indices_.size(),
        modelData,
        distance_threshold,
        thrust::raw_pointer_cast(d_inlier_indices.data()),
        thrust::raw_pointer_cast(d_inlier_count.data()));
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    int h_inlier_count = d_inlier_count[0];
    if (h_inlier_count <= 0)
    {
        out_indices.clear();
        return;
    }
    // 回传有效的 inlier 索引
    out_indices.resize(h_inlier_count);
    thrust::copy(d_inlier_indices.begin(),
                 d_inlier_indices.begin() + h_inlier_count,
                 out_indices.begin());
}

// ==========================================================================
// 第五部分: 公共接口桥接的实现
// ==========================================================================

// --- 公有构造函数: 创建PIMPL实现类的实例 ---
MinimalSampleQuadric_GPU::MinimalSampleQuadric_GPU(const DetectorParams &params)
    : impl_(std::make_unique<MinimalSampleQuadric_GPU_Impl>(params))
{
    // 构造函数的函数体是空的，所有工作都由 unique_ptr 在初始化列表中完成。
}

// --- 公有析构函数: unique_ptr 会自动处理 _Impl 对象的销毁 ---
MinimalSampleQuadric_GPU::~MinimalSampleQuadric_GPU() = default;

// --- 移动构造和移动赋值: 允许我们的类被高效移动 ---
MinimalSampleQuadric_GPU::MinimalSampleQuadric_GPU(MinimalSampleQuadric_GPU &&) noexcept = default;
MinimalSampleQuadric_GPU &MinimalSampleQuadric_GPU::operator=(MinimalSampleQuadric_GPU &&) noexcept = default;

// --- 公共方法的实现: 简单地将调用转发给PIMPL实现对象 ---
bool MinimalSampleQuadric_GPU::processCloud(const PointCloudConstPtr &input_cloud)
{
    return impl_->processCloud(input_cloud);
}

const std::vector<DetectedPrimitive, Eigen::aligned_allocator<DetectedPrimitive>> &
MinimalSampleQuadric_GPU::getDetectedPrimitives() const
{
    return impl_->getDetectedPrimitives();
}
MinimalSampleQuadric_GPU::PointCloudPtr MinimalSampleQuadric_GPU::getFinalCloud() const
{
    return impl_->getFinalCloud();
}