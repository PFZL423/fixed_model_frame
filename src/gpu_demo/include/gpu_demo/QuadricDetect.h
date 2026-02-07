#pragma once
#include <vector>
#include <memory>
#include <Eigen/Core>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>   // 添加这个
#include <thrust/sequence.h>
#include <cusolverDn.h>
#include <curand_kernel.h>
#include <ctime>  // 添加这个用于time()函数
#include <thrust/set_operations.h>
#include <thrust/sort.h>
#include <chrono>  // 添加计时器支持

// 使用GPUPreprocessor中的GPUPoint3f定义
#include "gpu_demo/GPUPreprocessor.h"

// ========================================
// 计时器工具结构体
// ========================================

/**
 * @brief 高精度计时器，用于性能分析
 * 提供毫秒级精度的GPU操作计时功能
 */


// ========================================
// GPU数据结构定义 (使用GPUPreprocessor中的定义)
// ========================================

// GPUPoint3f定义在GPUPreprocessor.h中

/**
 * @brief GPU端二次曲面模型数据结构
 * 将4x4对称矩阵Q展开为16个float，避免在GPU kernel中使用Eigen
 * 存储格式：行主序，coeffs[i*4+j] = Q(i,j)
 */
struct GPUQuadricModel
{
    float coeffs[16]; // 4x4二次曲面矩阵Q的展开形式
};

// ========================================
// 算法参数配置结构体
// ========================================
namespace quadric {


// ========================================
// 计时器工具结构体
// ========================================

/**
 * @brief 高精度计时器，用于性能分析
 * 提供毫秒级精度的GPU操作计时功能
 */
struct GPUTimer
{
    std::chrono::high_resolution_clock::time_point start_time;
    std::chrono::high_resolution_clock::time_point end_time;
    
    /**
     * @brief 开始计时
     */
    void start() {
        start_time = std::chrono::high_resolution_clock::now();
    }
    
    /**
     * @brief 结束计时并返回经过的毫秒数
     * @return 经过的时间（毫秒）
     */
    double stop() {
        end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        return duration.count() / 1000.0; // 转换为毫秒
    }
    
    /**
     * @brief 获取当前已经过的时间（不停止计时器）
     * @return 经过的时间（毫秒）
     */
    double elapsed() const {
        auto current_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(current_time - start_time);
        return duration.count() / 1000.0; // 转换为毫秒
    }
};
using GPUtimer = quadric::GPUTimer;

/**
 * @brief 二次曲面检测算法的完整参数配置
 * 涵盖RANSAC、LO-RANSAC、GPU batch处理等所有关键参数
 */
struct DetectorParams
{
    // === 核心RANSAC参数 ===
    double min_remaining_points_percentage = 0.03;  ///< 剩余点数阈值(相对于总点数)
    double quadric_distance_threshold = 0.02;       ///< 二次曲面内点距离阈值
    int min_quadric_inlier_count_absolute = 500;    ///< 有效模型的最小内点数(绝对值)
    int quadric_max_iterations = 5000;              ///< RANSAC最大迭代次数
    double min_quadric_inlier_percentage = 0.05;    ///< 有效模型的最小内点比例

    // === LO-RANSAC局部优化参数 ===
    bool enable_local_optimization = true;          ///< 是否启用LO-RANSAC精炼
    double lo_min_inlier_ratio = 0.6;              ///< 触发局部优化的最小内点比例
    double desired_prob = 0.99;                    ///< RANSAC成功概率目标
    int lo_sample_size = 15;                       ///< 局部优化采样点数

    // === 调试和输出控制 ===
    int verbosity = 1;                             ///< 详细输出级别 (0=静默, 1=正常, 2=详细)
};

/**
 * @brief 检测到的几何体基元结果结构
 * 用于存储检测到的二次曲面(椭球、椭圆抛物面、双曲面等)的完整信息
 */
struct DetectedPrimitive
{
    std::string type;                               ///< 几何体类型 ("quadric", "plane", etc.)
    Eigen::Matrix4f model_coefficients;            ///< 4x4二次曲面矩阵Q或平面参数
    pcl::PointCloud<pcl::PointXYZI>::Ptr inliers;   ///< 属于该几何体的内点点云

    DetectedPrimitive()
    {
        inliers.reset(new pcl::PointCloud<pcl::PointXYZI>());
        model_coefficients.setZero();
    }
};
} // namespace quadric
/**
 * @brief 全GPU加速的二次曲面检测器 - point_cloud_generator的升级版
 * 
 * 核心创新：
 * 1. 批量GPU RANSAC：1024个模型并行采样+验证，取代传统串行RANSAC
 * 2. GPU batch SVD：cuSolver批量矩阵分解，避免CPU-GPU频繁传输
 * 3. 全流水线GPU化：从采样到精炼全程GPU resident
 * 4. 内存优化：thrust智能管理，避免point包的内存对齐问题
 * 
 * 性能提升预期：
 * - RANSAC采样速度: ~100x加速 (1024并行 vs 串行)
 * - 内点验证速度: ~50x加速 (批量2D grid vs 单模型)
 * - 内存带宽: ~10x减少 (GPU resident vs 频繁传输)
 */
class QuadricDetect
{
public:
    void launchRemovePointsKernel();
    QuadricDetect(const quadric::DetectorParams &params);
    ~QuadricDetect();

    // ========================================
    // 公有接口 (与point包兼容)
    // ========================================
    
    /**
     * @brief 处理输入点云，检测所有二次曲面
     * @param input_cloud 输入的PCL点云 (PointXYZRGB格式)
     * @return true表示处理成功，false表示输入无效或处理失败
     */
    bool processCloud(const pcl::PointCloud<pcl::PointXYZI>::ConstPtr &input_cloud);
    bool processCloud(const thrust::device_vector<GPUPoint3f> &input_cloud);

    /**
     * @brief 获取检测到的所有几何体基元
    * @return 检测结果的const引用，包含二次曲面的模型参数和内点
    */
    const std::vector<quadric::DetectedPrimitive, Eigen::aligned_allocator<quadric::DetectedPrimitive>> &getDetectedPrimitives() const;

    /**
     * @brief 获取处理后的剩余点云
     * @return 移除所有检测到的几何体后的剩余点云
     */
    pcl::PointCloud<pcl::PointXYZI>::Ptr getFinalCloud() const;

    /**
     * @brief 设置CUDA流，绑定所有GPU操作到此流
     * @param stream CUDA流句柄
     */
    void setStream(cudaStream_t stream);

    /**
     * @brief 零拷贝直接处理接口：从外部GPU缓冲区直接进行二次曲面检测
     * @param d_points 外部GPU点云缓冲区指针（已压实）
     * @param count 点云数量
     * @return true表示处理成功，false表示输入无效或处理失败
     */
    bool processCloudDirect(GPUPoint3f* d_points, size_t count);

private:
    // 添加这个新函数的声明
    void validateInversePowerResults(int batch_size);
    void outputBestModelDetails(const GPUQuadricModel &best_model, int inlier_count, int model_idx, int iteration);

    pcl::PointCloud<pcl::PointXYZI>::Ptr extractInlierCloud() const;
    
    // GPU 辅助函数：在 .cu 文件中实现
    void gatherInliersToCompact() const;  // 将内点聚集到 d_compact_inliers_
    void gatherRemainingToCompact() const; // 将剩余点聚集到 d_compact_inliers_
    // 🆕 添加到QuadricDetect.h的public部分
    void performBatchInversePowerIteration(int batch_size);
    void launchComputeATA(int batch_size);
    void launchBatchQR(int batch_size);
    void launchBatchInversePower(int batch_size);
    void launchExtractQuadricModels(int batch_size);

    // 添加临时存储成员变量
    mutable thrust::device_vector<int> d_temp_inlier_indices_;
    mutable int current_inlier_count_;
    mutable thrust::device_vector<GPUPoint3f> d_compact_inliers_;  // GPU内部聚集的内点缓冲区

    // ========================================
    // 核心数据成员
    // ========================================
    quadric::DetectorParams params_;                                                                      ///< 算法参数配置
    std::vector<quadric::DetectedPrimitive, Eigen::aligned_allocator<quadric::DetectedPrimitive>> detected_primitives_;  ///< 检测结果存储

    // ========================================
    // GPU内存管理 (thrust智能指针，自动清理)
    // ========================================
    thrust::device_vector<GPUPoint3f> d_all_points_;       ///< GPU上的原始点云数据
    thrust::device_vector<int> d_remaining_indices_;       ///< 当前未分配的点索引列表
    thrust::device_vector<float> d_batch_matrices_;        ///< 批量A矩阵存储 (batch_size × 9 × 10)
    thrust::device_vector<GPUQuadricModel> d_batch_models_; ///< 批量拟合的二次曲面模型
    thrust::device_vector<int> d_batch_inlier_counts_;     ///< 每个模型的内点计数
    thrust::device_vector<curandState> d_rand_states_;     ///< GPU随机数生成器状态
    
    // 存储最优结果
    thrust::device_vector<int> d_best_model_index_;        ///< 最优模型在batch中的索引
    thrust::device_vector<int> d_best_model_count_;        ///< 最优模型的内点数

    // 反幂迭代所需的额外GPU内存

    thrust::device_vector<float> d_batch_ATA_matrices_; // 1024个10×10的A^T*A矩阵
    thrust::device_vector<float> d_batch_R_matrices_;   // 1024个10×10的R矩阵(QR分解)
    thrust::device_vector<float> d_batch_eigenvectors_; // 1024个10维特征向量

    // 两阶段RANSAC竞速相关（参考平面检测）
    thrust::device_vector<int> d_indices_full_;       ///< 完整索引序列（用于排序，预分配batch_size）
    thrust::device_vector<int> d_top_k_indices_;      ///< Top-K模型索引（用于精选，预分配128）
    thrust::device_vector<int> d_fine_inlier_counts_; ///< 精选阶段内点计数（预分配128）
    thrust::device_vector<GPUQuadricModel> d_candidate_models_; ///< 候选模型数组（预分配128）

    // ========================================
    // CUDA计算资源
    // ========================================
    cusolverDnHandle_t cusolver_handle_;                    ///< cuSolver句柄，用于批量SVD分解
    cudaStream_t stream_;                                   ///< CUDA流，用于异步计算
    bool owns_stream_;                                      ///< 是否拥有流的生命周期
    bool is_external_memory_;                              ///< 标记是否使用外部显存（零拷贝模式）
    GPUPoint3f* d_external_points_;                        ///< 外部GPU点云指针（零拷贝模式）
    uint8_t* d_valid_mask_;                                ///< 掩码缓冲区，标记点是否有效（1=有效，0=已移除）
    size_t max_points_capacity_;                           ///< 掩码缓冲区最大容量
    size_t original_total_count_;                          ///< 初始总点数（用于掩码缓冲区分配）

    // ========================================
    // 数据转换层 (PCL ↔ GPU格式)
    // ========================================
    
    /**
     * @brief 将PCL点云转换为GPU格式并上传
     * @param cloud 输入的PCL点云
     */
    void convertPCLtoGPU(const pcl::PointCloud<pcl::PointXYZI>::ConstPtr &cloud);
    
    /**
     * @brief 将GPU二次曲面模型转换为Eigen矩阵
     * @param gpu_model GPU格式的模型 (16个float)
     * @return 4x4的Eigen二次曲面矩阵
     */
    Eigen::Matrix4f convertGPUModelToEigen(const GPUQuadricModel &gpu_model);
    
    /**
     * @brief 获取点云指针（支持外部内存）
     * @return GPU点云指针
     */
    GPUPoint3f* getPointsPtr() const;

    /**
     * @brief 将10维SVD解向量转换为4×4二次曲面矩阵
     * @param q 10维解向量 (二次曲面系数)
     * @return 4x4对称二次曲面矩阵Q
     */
    Eigen::Matrix4f convertSolutionToQMatrix(const Eigen::VectorXf &q);

    // ========================================
    // 核心算法流程 (全GPU化升级版)
    // ========================================
    
    /**
     * @brief 批量GPU RANSAC主流程
     * 创新点：1024个模型并行处理，取代传统的串行RANSAC
     * 流程：批量采样 → 批量SVD → 批量验证 → 最优选择 → LO-RANSAC精炼
     */
    void findQuadrics_BatchGPU();
    
    /**
     * @brief LO-RANSAC局部优化精炼 (🚧待实现)
     * @param best_model [in/out] 待精炼的最优模型
     * @param best_inlier_count [in/out] 对应的内点数
     */
    void performLO_RANSAC(GPUQuadricModel &best_model, int &best_inlier_count);
    
    

    // ========================================
    // CUDA内核包装函数 (CPU调用，GPU执行)
    // ========================================
    
    /**
     * @brief 初始化GPU内存和计算资源
     * @param batch_size 批处理大小 (通常为1024)
     */
    void initializeGPUMemory(int batch_size);
    
    /**
     * @brief 初始化剩余索引序列（使用 kernel，支持流绑定）
     * @param count 索引数量
     */
    void initializeRemainingIndices(size_t count);
    
    /**
     * @brief 上传点云数据到GPU并初始化索引
     * @param h_points CPU端的点云数据
     */
    void uploadPointsToGPU(const std::vector<GPUPoint3f>& h_points);
    void uploadPointsToGPU(const thrust::device_vector<GPUPoint3f> &h_points);

    /**
     * @brief 初始化GPU随机数生成器状态
     * @param batch_size 需要初始化的随机数状态数量
     */
    void launchInitCurandStates(int batch_size);

    /**
     * @brief 启动批量采样和矩阵构建内核
     * 每个GPU线程负责采样9个点并构建9×10的A矩阵
     * @param batch_size 并行处理的模型数量
     */
    void launchSampleAndBuildMatrices(int batch_size);
    
    /**
     * @brief 启动批量内点计数内核（粗筛阶段）
     * 使用2D Grid架构：Y维度对应模型，X维度对应点云
     * @param batch_size 需要验证的模型数量
     * @param stride 采样步长（1=全量，50=2%采样）
    */
    void launchCountInliersBatch(int batch_size, int stride);
    
    /**
     * @brief 两阶段RANSAC：选择Top-K模型
     * 从粗筛结果中选出内点数最高的k个模型
     * @param k 选择的模型数量
     */
    void launchSelectTopKModels(int k);
    
    /**
     * @brief 两阶段RANSAC：精选阶段内点计数
     * 对Top-K候选模型进行全量验证
     * @param k 候选模型数量
     */
    void launchFineCountInliersBatch(int k);
    
    /**
     * @brief 启动最优模型查找内核
     * GPU并行reduce找出内点数最多的模型
     * @param batch_size 参与比较的模型数量
     */
    void launchFindBestModel(int batch_size);
    
   
    /**
     * @brief 启动内点提取内核 (🚧待完善实现)
     * @param model 用于提取内点的二次曲面模型
     */
    void launchExtractInliers(const GPUQuadricModel *model);
    
    /**
     * @brief 从GPU获取最优模型结果
     * @param h_best_index [out] 最优模型索引
     * @param h_best_count [out] 最优模型内点数
     */
    void getBestModelResults(thrust::host_vector<int>& h_best_index, thrust::host_vector<int>& h_best_count);

    // ========================================
    // 辅助功能函数
    // ========================================
    
    /**
     * @brief 从剩余点云中移除已检测的内点
     * @param indices_to_remove 需要移除的点的全局索引列表
     */
    void removeFoundPoints(const std::vector<int> &indices_to_remove);
    
    /**
     * @brief 使用PCL进行平面检测 (可选的后处理步骤)
     */
    void findPlanes_PCL();
};