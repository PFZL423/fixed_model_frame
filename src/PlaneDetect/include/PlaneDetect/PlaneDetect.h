#pragma once
#include <vector>
#include <memory>
#include <Eigen/Core>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h> // 添加这个
#include <thrust/sequence.h>
#include <curand_kernel.h>
#include <ctime> // 添加这个用于time()函数
#include <thrust/set_operations.h>
#include <thrust/sort.h>
#include <chrono> // 添加计时器支持
// 使用GPUPreprocessor中的GPUPoint3f定义
#include "gpu_demo/GPUPreprocessor.h"

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
    void start()
    {
        start_time = std::chrono::high_resolution_clock::now();
    }

    /**
     * @brief 结束计时并返回经过的毫秒数
     * @return 经过的时间（毫秒）
     */
    double stop()
    {
        end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        return duration.count() / 1000.0; // 转换为毫秒
    }

    /**
     * @brief 获取当前已经过的时间（不停止计时器）
     * @return 经过的时间（毫秒）
     */
    double elapsed() const
    {
        auto current_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(current_time - start_time);
        return duration.count() / 1000.0; // 转换为毫秒
    }
};

/**
 * @brief GPU端平面模型数据结构
 * 避免在GPU kernel中使用Eigen
 */
struct GPUPlaneModel
{
    float coeffs[4]; // 平面方程的系数（Ax + By + Cz + D = 0）
};

/**
 * @brief 平面检测算法的完整参数配置
 * 简化版RANSAC参数，专注于平面检测
 */
struct DetectorParams
{
    // === 核心RANSAC参数 ===
    double min_remaining_points_percentage = 0.03; ///< 剩余点数阈值(相对于总点数)
    double plane_distance_threshold = 0.02;        ///< 平面内点距离阈值
    int min_plane_inlier_count_absolute = 500;     ///< 有效模型的最小内点数(绝对值)
    int plane_max_iterations = 2000;               ///< RANSAC最大迭代次数
    double min_plane_inlier_percentage = 0.05;     ///< 有效模型的最小内点比例

    // === Batch-RANSAC参数 ===
    int batch_size = 2048;                         ///< 每批次处理的样本数量
    
    // === 两阶段RANSAC竞速参数 ===
    double ransac_coarse_ratio = 0.02;             ///< 粗筛采样率（默认2%，stride=50）
    int ransac_fine_k = 20;                       ///< 精选阶段候选模型数量（默认20）
    
    // === 调试和输出控制 ===
    int verbosity = 1;                             ///< 详细输出级别 (0=静默, 1=正常, 2=详细)
};

/**
 * @brief 检测到的几何体基元结果结构
 * 用于存储检测到的平面的完整信息
 */
template<typename PointT>
struct DetectedPrimitive
{
    std::string type="plane";                            ///< 几何体类型 ("quadric", "plane", etc.)
    float model_coefficients[4];          ///< 4平面参数
    typename pcl::PointCloud<PointT>::Ptr inliers; ///< 属于该几何体的内点点云

    DetectedPrimitive()
    {
        inliers.reset(new pcl::PointCloud<PointT>());
        model_coefficients[0] = 0;
        model_coefficients[1] = 0;
        model_coefficients[2] = 0;
        model_coefficients[3] = 0;
    }
};

/**
 * @brief 全GPU加速的平面检测器 - 二次曲面检测的简化版
 *
 * 核心简化：
 * 1. 批量GPU RANSAC：2048个平面模型并行采样+验证
 * 2. 简化采样：每个模型只需3个点（vs 9个点用于二次曲面）
 * 3. 直接解析解：平面方程直接计算（无需SVD分解）
 * 4. 内存优化：thrust智能管理，避免复杂矩阵操作
 *
 * 性能优势：
 * - 采样速度: 更快（3点 vs 9点）
 * - 模型拟合: 直接解析（无需反幂迭代）
 * - 内存使用: 大幅减少（4参数 vs 16参数）
 */
template<typename PointT>
 class PlaneDetect
{
public:
    PlaneDetect(const DetectorParams &params);
    ~PlaneDetect();

    /**
     * @brief 处理输入点云，检测所有二次曲面
     * @param input_cloud 输入的PCL点云 (PointXYZ格式)
     * @return true表示处理成功，false表示输入无效或处理失败
     */
    bool processCloud(const typename pcl::PointCloud<PointT>::ConstPtr &input_cloud);
    bool processCloud(const thrust::device_vector<GPUPoint3f> &input_cloud);

    /**
     * @brief 获取检测到的所有几何体基元
     * @return 检测结果的const引用，包含二次曲面的模型参数和内点
    */
   const std::vector<DetectedPrimitive<PointT>> &getDetectedPrimitives() const;

   /**
    * @brief 获取处理后的剩余点云
    * @return 移除所有检测到的几何体后的剩余点云
    */
   typename pcl::PointCloud<PointT>::Ptr getFinalCloud() const;

   /**
    * @brief 获取检测到的平面数量
    * @return 检测到的平面数量
    */
   size_t getDetectedPlaneCount() const;

   /**
    * @brief 获取指定索引的平面参数
    * @param index 平面索引
    * @return 平面方程系数 [A, B, C, D]，满足 Ax + By + Cz + D = 0
    */
   std::vector<float> getPlaneCoefficients(size_t index) const;

   /**
    * @brief 获取所有检测到的平面参数（打包返回）
    * @return 所有平面的系数矩阵，每行4个值[A, B, C, D]
    */
   std::vector<std::vector<float>> getAllPlaneCoefficients() const;

   /**
    * @brief 获取指定索引平面的内点点云
    * @param index 平面索引  
    * @return 该平面的内点点云
    */
   typename pcl::PointCloud<PointT>::Ptr getPlaneInliers(size_t index) const;

   /**
    * @brief 获取指定索引平面的内点数量
    * @param index 平面索引
    * @return 该平面的内点数量
    */
   size_t getPlaneInlierCount(size_t index) const;

    /**
     * @brief 获取剩余点云的数量
     * @return 剩余点的数量
     */
    size_t getRemainingPointCount() const;

    /**
     * @brief 设置CUDA流（用于异步操作和流隔离）
     * @param stream CUDA流指针
     */
    void setStream(cudaStream_t stream);

    /**
     * @brief 获取当前CUDA流
     * @return CUDA流指针
     */
    cudaStream_t getStream() const { return stream_; }

    /**
     * @brief 零拷贝接口：从外部GPU缓冲区直接进行平面检测
     * 借用外部显存指针，避免数据拷贝开销
     * @param d_external_points 外部GPU点云缓冲区指针
     * @param count 点云数量
     */
    void findPlanesFromGPU(GPUPoint3f* d_external_points, size_t count);


private:
    // 添加临时存储成员变量
    mutable thrust::device_vector<int> d_temp_inlier_indices_;
    mutable int current_inlier_count_;

    // ========================================
    // GPU显存预分配（避免每帧malloc/free开销）
    // ========================================
    GPUPoint3f* d_points_buffer_;        ///< 预分配的GPU显存缓冲区
    uint8_t* d_valid_mask_;              ///< 点云有效性掩码（1=有效，0=已移除）
    size_t max_points_capacity_;        ///< 最大容量（默认200万点）
    size_t current_point_count_;        ///< 当前点云数量（替代d_all_points_.size()）
    GPUPoint3f* h_pinned_buffer_;       ///< CPU端锁页内存缓冲区（预分配，DMA直接访问，避免驱动层拷贝）
    cudaStream_t stream_;               ///< CUDA流，用于异步操作和流隔离
    bool is_external_memory_;           ///< 标记是否使用外部显存（零拷贝模式）
    GPUPoint3f* d_points_backup_;       ///< 备份原始 d_points_buffer_ 指针（用于零拷贝恢复）

    // ========================================
    // 核心数据成员
    // ========================================
    DetectorParams params_;                                                                           ///< 算法参数配置
    std::vector<DetectedPrimitive<PointT>> detected_primitives_; ///< 检测结果存储

    // ========================================
    // GPU内存管理 (thrust智能指针，自动清理)
    // ========================================
    thrust::device_vector<int> d_remaining_indices_;        ///< 当前未分配的点索引列表
    thrust::device_vector<GPUPlaneModel> d_batch_models_;   ///< 批量拟合的平面模型
    thrust::device_vector<int> d_batch_inlier_counts_;      ///< 每个模型的内点计数
    thrust::device_vector<curandState> d_rand_states_;      ///< GPU随机数生成器状态

    // 存储最优结果
    thrust::device_vector<int> d_best_model_index_; ///< 最优模型在batch中的索引
    thrust::device_vector<int> d_best_model_count_; ///< 最优模型的内点数

    // 两阶段RANSAC竞速相关
    thrust::device_vector<int> d_indices_full_;       ///< 完整索引序列（用于排序，预分配batch_size）
    thrust::device_vector<int> d_top_k_indices_;      ///< Top-K模型索引（用于精选，预分配128）
    thrust::device_vector<int> d_fine_inlier_counts_; ///< 精选阶段内点计数（预分配128）
    thrust::device_vector<GPUPlaneModel> d_candidate_models_; ///< 候选模型数组（预分配128）

    /**
     * @brief 将PCL点云转换为GPU格式并上传
     * @param cloud 输入的PCL点云
     */
    void convertPCLtoGPU(const typename pcl::PointCloud<PointT>::ConstPtr &cloud);

    /**
     * @brief 批量GPU RANSAC主流程
     * 简化版：批量采样3点 → 直接解析拟合 → 批量验证 → 最优选择
     */
    void findPlanes_BatchGPU();

    /**
     * @brief 从GPU提取内点构建点云
     * @return 内点的PCL点云
     */
    typename pcl::PointCloud<PointT>::Ptr extractInlierCloud() const;

    // ========================================
    // CUDA内核包装函数 (CPU调用，GPU执行)
    // ========================================

    /**
     * @brief 初始化GPU内存和计算资源
     * @param batch_size 批处理大小 (通常为2048)
     */
    void initializeGPUMemory(int batch_size);

    /**
     * @brief 上传点云数据到GPU并初始化索引（异步，使用流隔离）
     * @param h_points CPU端的点云数据（锁页内存）
     * @param point_count 点云数量
     */
    void uploadPointsToGPU(const GPUPoint3f* h_points, size_t point_count);
    void uploadPointsToGPU(const thrust::device_vector<GPUPoint3f> &h_points);

    /**
     * @brief 初始化GPU随机数生成器状态
     * @param batch_size 需要初始化的随机数状态数量
     */
    void launchInitCurandStates(int batch_size);

    /**
     * @brief 启动批量采样和平面拟合内核
     * 每个GPU线程负责采样3个点并直接计算平面方程
     * @param batch_size 并行处理的模型数量
     */
    void launchSampleAndFitPlanes(int batch_size);

    /**
     * @brief 启动批量内点计数内核（粗筛阶段）
     * 使用2D Grid架构：Y维度对应模型，X维度对应点云
     * 通过stride参数实现子采样，大幅降低计算量
     * @param batch_size 需要验证的模型数量
     */
    void launchCountInliersBatch(int batch_size);

    /**
     * @brief 在GPU端选出coarse_score最高的k个模型索引
     * 使用thrust::sort_by_key进行排序并提取Top-K
     * @param k 需要选择的候选模型数量
     */
    void launchSelectTopKModels(int k);

    /**
     * @brief 启动精选阶段内点计数内核
     * 只对前k个候选模型执行全量点云计数
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
     * @brief 启动内点提取内核
     * @param model 用于提取内点的平面模型
     */
    void launchExtractInliers(const GPUPlaneModel *model);

    /**
     * @brief 从GPU获取最优模型结果
     * @param h_best_index [out] 最优模型索引
     * @param h_best_count [out] 最优模型内点数
     */
    void getBestModelResults(thrust::host_vector<int> &h_best_index, thrust::host_vector<int> &h_best_count);

    // ========================================
    // 辅助功能函数
    // ========================================

    /**
     * @brief 从剩余点云中移除已检测的内点
     * @param indices_to_remove 需要移除的点的全局索引列表
     */
    void removeFoundPoints(const std::vector<int> &indices_to_remove);

    /**
     * @brief 启动GPU内核移除内点（使用掩码标记）
     */
    void launchRemovePointsKernel();

    /**
     * @brief 下载紧凑点云到CPU缓冲区 - 避免ABI不一致
     * 在.cu内部完成copy_if，然后直接下载到CPU缓冲区
     * @param h_out_buffer CPU端输出缓冲区（必须预分配足够空间）
     * @return 实际下载的有效点数量
     */
    int downloadCompactPoints(GPUPoint3f* h_out_buffer) const;

    /**
     * @brief 计算有效点数量 - 根据掩码统计
     * 使用thrust::count统计掩码为1的点数，必须在.cu文件中实现
     * @return 有效点数量
     */
    int countValidPoints() const;

    /**
     * @brief 调整辅助缓冲区大小（支持外部内存模式）
     * 当使用外部显存时，跳过 d_points_buffer_ 的分配/释放
     * @param point_count 点云数量
     */
    void resizeBuffers(size_t point_count);
};