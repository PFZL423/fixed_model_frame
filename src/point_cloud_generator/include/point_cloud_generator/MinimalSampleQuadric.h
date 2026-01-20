#ifndef MINIMAL_SAMPLE_QUADRIC_H
#define MINIMAL_SAMPLE_QUADRIC_H
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <Eigen/Core>
#include <random>
#include <ceres/ceres.h>
#include "point_cloud_generator/Point_cloud_preprocessor.h"
#include <vector>
#include <string>
struct QuadricGeometricError
{
    const double x_, y_, z_;

    QuadricGeometricError(double x, double y, double z) : x_(x), y_(y), z_(z) {}

    template <typename T>
    bool operator()(const T *const q, T *residual) const
    {
        
        T F = q[0] * T(x_ * x_) + q[1] * T(y_ * y_) + q[2] * T(z_ * z_) +
              q[3] * T(2 * x_ * y_) + q[4] * T(2 * x_ * z_) + q[5] * T(2 * y_ * z_) +
              q[6] * T(2 * x_) + q[7] * T(2 * y_) + q[8] * T(2 * z_) + q[9];

        T gx = T(2.0) * (q[0] * T(x_) + q[3] * T(y_) + q[4] * T(z_) + q[6]);
        T gy = T(2.0) * (q[1] * T(y_) + q[3] * T(x_) + q[5] * T(z_) + q[7]);
        T gz = T(2.0) * (q[2] * T(z_) + q[4] * T(x_) + q[5] * T(y_) + q[8]);

        T grad_norm_sq = gx * gx + gy * gy + gz * gz;

        if (grad_norm_sq < T(1e-12))
        {
            residual[0] = T(0.0);
        }
        else
        {
            residual[0] = F / ceres::sqrt(grad_norm_sq);
        }

        return true;
    }
};

//1 配置参数的结构体
struct DetectorParams
{
    // 我认为需要有ransac的单次循环次数，单次ransac的那个最小点数，判断内点的阈值距离，可以补充....
    // --- 预处理参数部分 ---
    PreprocessingParams preprocessing; // <<<<<<< 核心修改：直接包含预处理参数结构体

    // --- 平面检测(RANSAC)参数 ---
    int plane_max_iterations = 1000;           // RANSAC最大迭代次数
    double plane_distance_threshold = 0.02;    // 判断平面内点的距离阈值
    double min_plane_inlier_percentage = 0.10; // 一个平面被接受所需的最少内点比例（相对于当前点云）

    // --- 二次曲面检测(RANSAC)参数 ---
    int quadric_max_iterations = 5000;           // RANSAC最大迭代次数
    double quadric_distance_threshold = 0.02;    // 判断二次曲面内点的距离阈值
    double min_quadric_inlier_percentage = 0.05; // 一个二次曲面被接受所需的最少内点比例

    // --- 循环控制参数 ---
    double min_remaining_points_percentage = 0.03; // 当剩余点数少于原始点数的这个比例时，停止循环

    // --- 投票机制参数 (针对三点法) ---
    float voting_bin_size = 0.1f; // 投票空间（λ）的直方图bin的大小
};

//2用于存储结果的结构体
struct DetectedPrimitive
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    std::string type; // "plane" 或 "quadric"

    // 模型参数，统一使用4x4 Q矩阵存储。对于平面，可以只用前几项。
    Eigen::Matrix4f model_coefficients;

    // 内点（包含位置和法线）
    pcl::PointCloud<pcl::PointNormal>::Ptr inliers;

    DetectedPrimitive() 
    {
        // 先初始化 PCL 智能指针
        inliers.reset(new pcl::PointCloud<pcl::PointNormal>());
        
        // 再初始化 Eigen 矩阵
        model_coefficients.setZero();
    }
};
class MinimalSampleQuadric
{
   
    using PointT = pcl::PointNormal; // **关键修正：需要同时处理点和法线，所以用PointNormal
    using PointCloud = pcl::PointCloud<PointT>;
    using PointCloudPtr = PointCloud::Ptr;

public:

    MinimalSampleQuadric(const DetectorParams& params);  
    /**
     * @brief 主处理函数，执行完整的检测流程
     * @param input_cloud 待处理的原始点云（只需要XYZ，法线会在内部计算）
     * @return 如果处理成功则返回true
     */
    bool processCloud(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr &input_cloud);
    /**
     * @brief 获取检测到的所有几何基元
     * @return 一个包含所有基元信息的vector的常量引用
     */
    const std::vector<DetectedPrimitive> &getDetectedPrimitives() const
    {
        return detected_primitives_;
    }
    /**
     * @brief 获取最终剩余的、未被分类的点云
     * @return 指向剩余点云的智能指针
     */
    PointCloudPtr getFinalCloud() const
    {
        return final_remaining_cloud_;
    }


private:

    // **我新增的私有预处理函数声明**
    /**
     * @brief 对输入的原始点云执行预处理，并返回一个包含位置和法线的新点云
     * @param input_cloud 待处理的原始点云 (仅需要XYZ)
     * @param preprocessed_cloud_out 输出的处理后并合并了法线的点云 (PointNormal)
     * @return 如果预处理成功且点云非空，返回true
     */
    bool preProcess(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr &input_cloud, PointCloudPtr &preprocessed_cloud_out);
    // void preProcess(PointCloudPtr &input_cloud);
    PointCloudPreprocessor preprocessor_;
    void detectPlanes(PointCloudPtr &remain_cloud);
    void detectQuadric(PointCloudPtr &remain_cloud);
    DetectorParams params_;
    std::vector<DetectedPrimitive> detected_primitives_; // 存储所有检测结果
    PointCloudPtr final_remaining_cloud_;                // 存储最终剩余的点云

    size_t initial_point_count_ = 0; // 存储初始点云数量，用于计算百分比

    Eigen::Matrix4f vectorToQMatrix(const Eigen::VectorXd &q_vec)
    {
        Eigen::Matrix4f Q = Eigen::Matrix4f::Zero();
        Q(0, 0) = q_vec(0);                 // A
        Q(1, 1) = q_vec(1);                 // B
        Q(2, 2) = q_vec(2);                 // C
        Q(0, 1) = Q(1, 0) = q_vec(3) / 2.0; // D
        Q(0, 2) = Q(2, 0) = q_vec(4) / 2.0; // E
        Q(1, 2) = Q(2, 1) = q_vec(5) / 2.0; // F
        Q(0, 3) = Q(3, 0) = q_vec(6) / 2.0; // G
        Q(1, 3) = Q(3, 1) = q_vec(7) / 2.0; // H
        Q(2, 3) = Q(3, 2) = q_vec(8) / 2.0; // I
        Q(3, 3) = q_vec(9);                 // J
        return Q;
    }
    Eigen::VectorXd QMatrixToVector(const Eigen::Matrix4f &Q) const;
    /**
     * @brief 在给定的点云中寻找单个最佳二次曲面模型。
     * @param cloud 带有法线的输入点云。
     * @param best_model_coefficients [输出] 找到的最佳模型的4x4 Q矩阵。
     * @param best_inlier_indices [输出] 最佳模型的内点索引。
     * @return 如果成功找到一个满足条件的模型，则返回true。
     */
    bool findQuadric(const PointCloudPtr &cloud,
                     Eigen::Matrix4f &best_model_coefficients,
                     pcl::PointIndices::Ptr &best_inlier_indices);
};
#endif