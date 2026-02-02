#pragma once

#include <memory>
#include <vector>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

struct GPUPoint3f
{
    float x, y, z;
    float intensity; // 强度值（用于PointXYZI）
    // 4个float正好16字节，自然对齐，满足CUDA内存对齐要求
};

struct GPUPointNormal3f
{
    float x, y, z;
    float normal_x, normal_y, normal_z;
};

// 预处理配置
struct PreprocessConfig
{
    // 体素下采样
    bool enable_voxel_filter = true;
    float voxel_size = 0.05f;

    // 离群点移除
    bool enable_outlier_removal = true;
    enum OutlierMethod
    {
        STATISTICAL,
        RADIUS
    } outlier_method = STATISTICAL;
    int statistical_k = 50;
    float statistical_stddev = 1.0f;
    float radius_search = 0.1f;
    int min_radius_neighbors = 5;

    // 法线计算 (关键开关)
    bool compute_normals = false;
    float normal_radius = 0.1f;
    int normal_k = 20;

    // 其他选项
    bool enable_ground_removal = false;
    float ground_threshold = 0.02f;
};

// 处理结果包装类
class ProcessingResult
{
public:
    ProcessingResult() = default;
    ~ProcessingResult() = default;

    // 禁止拷贝，只允许移动
    ProcessingResult(const ProcessingResult &) = delete;
    ProcessingResult &operator=(const ProcessingResult &) = delete;
    ProcessingResult(ProcessingResult &&) = default;
    ProcessingResult &operator=(ProcessingResult &&) = default;

    // 数据访问接口
    bool hasNormals() const { return has_normals_; }
    size_t getPointCount() const { return point_count_; }

    // 获取基础点云 (总是可用)
    thrust::device_vector<GPUPoint3f> &getPoints()
    {
        return *d_points_;
    }
    const thrust::device_vector<GPUPoint3f> &getPoints() const
    {
        return *d_points_;
    }

    // 获取带法线点云 (仅当计算了法线时可用)
    // thrust::device_vector<GPUPointNormal3f> &getPointsWithNormals()
    // {
    //     if (!has_normals_)
    //     {
    //         throw std::runtime_error("Normals not computed! Check hasNormals() first.");
    //     }
    //     return *d_points_normal_;
    // }
    // const thrust::device_vector<GPUPointNormal3f> &getPointsWithNormals() const
    // {
    //     if (!has_normals_)
    //     {
    //         throw std::runtime_error("Normals not computed! Check hasNormals() first.");
    //     }
    //     return *d_points_normal_;
    // }

    // 下载到CPU (可选接口)
    std::vector<GPUPoint3f> downloadPoints() const;
    std::vector<GPUPointNormal3f> downloadPointsWithNormals() const;

private:
    friend class GPUPreprocessor;

    bool has_normals_ = false;
    size_t point_count_ = 0;
    thrust::device_vector<GPUPoint3f> *d_points_ = nullptr;
    thrust::device_vector<GPUPointNormal3f> *d_points_normal_ = nullptr;

    void setPointsRef(thrust::device_vector<GPUPoint3f> *points) { d_points_ = points; }
    void setPointsNormalRef(thrust::device_vector<GPUPointNormal3f> *points_normal)
    {
        // d_points_normal_ = points_normal;
        // has_normals_ = true;
    }
    void setPointCount(size_t count) { point_count_ = count; }
};

// GPU预处理器主类
class GPUPreprocessor
{
public:
    GPUPreprocessor();
    ~GPUPreprocessor();

    // 禁止拷贝
    GPUPreprocessor(const GPUPreprocessor &) = delete;
    GPUPreprocessor &operator=(const GPUPreprocessor &) = delete;

    // 主要处理接口
    ProcessingResult process(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cpu_cloud,
                             const PreprocessConfig &config);

    // 内存管理
    void reserveMemory(size_t max_points);
    void clearMemory();

    // Stream 管理接口
    void setStream(cudaStream_t stream);
    cudaStream_t getStream() const { return stream_; }

    // 零拷贝接口：暴露输出缓冲区指针
    GPUPoint3f* getOutputBuffer() const;
    size_t getOutputCount() const;

    // 性能统计
    struct PerformanceStats
    {
        float upload_time_ms = 0.0f;
        float voxel_filter_time_ms = 0.0f;
        float outlier_removal_time_ms = 0.0f;
        float normal_estimation_time_ms = 0.0f;
        float total_time_ms = 0.0f;
    };
    const PerformanceStats &getLastStats() const { return last_stats_; }

    // CUDA核函数的包裹函数声明
    void cuda_convertToPointsWithNormals(GPUPoint3f *input_points, GPUPointNormal3f *output_points, size_t point_count);
    size_t cuda_compactValidPoints(
        GPUPoint3f *input_points, bool *valid_flags,
        GPUPoint3f *output_points, size_t input_count);
    size_t cuda_performGroundRemoval(
        GPUPoint3f *input_points, size_t input_count,
        GPUPoint3f *output_points, float threshold);
    void cuda_performNormalEstimation(
        GPUPoint3f *points, GPUPointNormal3f *points_with_normals,
        size_t point_count, float radius, int k);
    size_t cuda_performOutlierRemoval(
        GPUPoint3f *input_points, size_t input_count,
        GPUPoint3f *output_points, const PreprocessConfig &config);
    size_t cuda_performVoxelDownsampling(
        GPUPoint3f *input_points, size_t input_count,
        GPUPoint3f *output_points, float voxel_size);
    // ✅ 新增：把这些函数也移到.cu文件中实现
    void cuda_launchVoxelFilter(float voxel_size);
    void cuda_launchOutlierRemoval(const PreprocessConfig &config);
    void cuda_launchGroundRemoval(float threshold);
    void cuda_compactValidPoints();
    void cuda_initializeMemory(size_t max_points);

    bool cpuFallbackSort(size_t input_count);

    void processVoxelCentroids(size_t input_count);

private:
    // CPU端锁页内存缓冲区（预分配，DMA直接访问）
    GPUPoint3f* h_pinned_buffer_;
    size_t max_points_capacity_;  ///< 最大容量（用于预分配）
    
    // CUDA流（用于异步操作和流隔离）
    cudaStream_t stream_;
    bool owns_stream_;  ///< 是否拥有stream的所有权

    // GPU内存缓冲区
    thrust::device_vector<GPUPoint3f> d_input_points_;
    thrust::device_vector<GPUPoint3f> d_temp_points_;
    thrust::device_vector<GPUPoint3f> d_output_points_;
    thrust::device_vector<GPUPointNormal3f> d_output_points_normal_;

    // 体素下采样相关
    thrust::device_vector<uint64_t> d_voxel_keys_;
    thrust::device_vector<int> d_voxel_boundaries_;
    thrust::device_vector<uint64_t> d_unique_keys_;

    // 离群点移除相关
    thrust::device_vector<int> d_neighbor_counts_;
    thrust::device_vector<bool> d_valid_flags_;

    // 法线估计相关
    thrust::device_vector<int> d_knn_indices_;
    thrust::device_vector<float> d_knn_distances_;

    // 性能统计
    mutable PerformanceStats last_stats_;
    void launchNormalEstimation(float radius, int k);

    // 内部处理流程
    void preprocessOnGPU(const PreprocessConfig &config);
    ProcessingResult createResult(const PreprocessConfig &config);

    // 独立转换函数（返回转换的点数，数据直接写入 h_pinned_buffer_）
    size_t convertPCLToGPU(const pcl::PointCloud<pcl::PointXYZ>::Ptr &cpu_cloud);

    // GPU上传函数（在.cu中实现，使用异步上传和 pinned memory）
    void cuda_uploadGPUPoints(const GPUPoint3f* h_pinned_points, size_t count);

    // 获取GPU结果的引用（供后续模块使用）
    const thrust::device_vector<GPUPoint3f> &getOutputPoints() const { return d_output_points_; }
    const thrust::device_vector<GPUPoint3f> &getTempPoints() const { return d_temp_points_; }

    // 工具函数
    void convertToPointsWithNormals();
    size_t getCurrentPointCount() const;
};
