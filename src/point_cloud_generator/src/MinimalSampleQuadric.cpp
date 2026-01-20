#include "point_cloud_generator/MinimalSampleQuadric.h"
#include <iostream>
#include <pcl/sample_consensus/method_types.h> // for SAC_RANSAC
#include <pcl/sample_consensus/model_types.h>  // for SACMODEL_PLANE
#include <pcl/segmentation/sac_segmentation.h> // for SACSegmentation
#include <pcl/filters/extract_indices.h>       // for ExtractIndices
#include <iomanip>
#include <algorithm>
#include <cmath> // 包含此头文件
#include <chrono>
struct MSQParams
{
    int quadric_max_iterations = 5000;
    double quadric_min_inlier_probability = 0.99;
    double quadric_distance_threshold = 0.05;
};
// 一个简单的RAII计时器
struct ScopedTimer
{
    std::chrono::time_point<std::chrono::high_resolution_clock> start;
    std::string message;

    // 构造函数：记录开始时间并保存要打印的消息
    ScopedTimer(const std::string &msg) : message(msg)
    {
        start = std::chrono::high_resolution_clock::now();
    }

    // 析构函数：在对象生命周期结束时自动调用
    ~ScopedTimer()
    {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        // 您可以使用 PCL_INFO 或者 std::cout
        PCL_INFO("%s took %lld ms.", message.c_str(), duration_ms);
    }
};

using MSQ = MinimalSampleQuadric;
MSQ::MinimalSampleQuadric(const DetectorParams &params)
    // 使用成员初始化列表（Member Initializer List）来初始化成员变量
    : params_(params),                          // 将传入的参数直接赋值给成员变量params_
      preprocessor_(params.preprocessing),      // 使用params中的特定值初始化preprocessor_对象
      final_remaining_cloud_(new PointCloud()), // 初始化指向点云的智能指针
      initial_point_count_(0)                   // 初始化计数器为0
{
    // 这里可以放一些额外的逻辑，比如打印日志等。
    std::cout << "MinimalSampleQuadric detector initialized." << std::endl;
}

bool MSQ::processCloud(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr &input_cloud)
{
    detected_primitives_.clear();
    if (final_remaining_cloud_)
    {
        final_remaining_cloud_->clear();
    }
    else
    {
        final_remaining_cloud_.reset(new PointCloud());
    }
    if (!input_cloud || input_cloud->empty())
    {
        std::cerr << "Error: Input cloud for processCloud is null or empty." << std::endl;
        return false;
    }
    // 记录最原始的点云数量（预处理前）
    initial_point_count_ = input_cloud->size();
    std::cout << "Starting detection process with " << initial_point_count_ << " initial points." << std::endl;

    PointCloudPtr working_cloud(new PointCloud());
    if (!preProcess(input_cloud, working_cloud) || working_cloud->empty())
    {
        std::cout << "Preprocessing resulted in an empty point cloud. Stopping." << std::endl;
        // 注意：即使预处理后为空，也应将空的 working_cloud 赋给 final_remaining_cloud_
        pcl::copyPointCloud(*working_cloud, *final_remaining_cloud_);
        return true; // 处理流程正常结束，只是没有点可检测
    }
    std::cout << "Preprocessing finished. Working cloud has " << working_cloud->size() << " points with normals." << std::endl;

    detectPlanes(working_cloud);
    detectQuadric(working_cloud);

    // pcl::copyPointCloud(*working_cloud, *final_remaining_cloud_);
    final_remaining_cloud_ = working_cloud;

    std::cout << "Detection process finished. " << detected_primitives_.size() << " primitives detected." << std::endl;
    std::cout << "Final remaining cloud has " << final_remaining_cloud_->size() << " points." << std::endl;

    return true;
}
bool MSQ::preProcess(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr &input_cloud, PointCloudPtr &preprocessed_cloud_out)
{
    pcl::PointCloud<pcl::PointXYZ>::Ptr processed_xyz(new pcl::PointCloud<pcl::PointXYZ>());

    if (!preprocessor_.process(input_cloud, processed_xyz))
    {
        // 如果预处理失败（例如，输入点云为空），则直接返回失败
        std::cerr << "Error during preprocessing step." << std::endl;
        return false;
    }
    // 检查处理完是否为空
    if (processed_xyz->empty())
    {
        std::cout << "Warning: Point cloud is empty after preprocessing." << std::endl;
        preprocessed_cloud_out->clear(); // 确保输出为空
        return true;
    }
    
    // 直接将处理后的点云拷贝到输出，不再合并法线
    pcl::copyPointCloud(*processed_xyz, *preprocessed_cloud_out);

    return true;
}
void MSQ::detectPlanes(PointCloudPtr &remain_cloud)
{
    // 如果初始点云为空或点数不足，则直接返回
    if (!remain_cloud || remain_cloud->size() < 3)
    {
        return;
    }
    std::cout << "---Starting Plane Detection---" << std::endl;

    // 创建一个临时点云指针，用于在循环中操作，避免直接修改传入的引用
    // 使用 pcl::copyPointCloud 而不是拷贝构造函数来避免 boost::shared_ptr 问题
    PointCloudPtr current_cloud(new PointCloud());
    pcl::copyPointCloud(*remain_cloud, *current_cloud);

    pcl::SACSegmentation<PointT> seg;
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_PLANE); // 设置拟合类型为平面
    seg.setMethodType(pcl::SAC_RANSAC);    // 方法为RANSAC

    // 使用我们在DetectorParams中定义的参数
    seg.setMaxIterations(params_.plane_max_iterations);
    seg.setDistanceThreshold(params_.plane_distance_threshold);

    pcl::PointIndices::Ptr inlier_indices(new pcl::PointIndices);
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);

    pcl::ExtractIndices<PointT> extract; // 过滤器

    while (true)
    {
        size_t current_point_count = current_cloud->size();

        // 1. 终止条件检查：如果剩余点数太少，无法形成平面，则退出循环
        if (current_point_count < 3)
        {
            std::cout << "Remaining points (" << current_point_count << ") are too few to form a plane. Stopping." << std::endl;
            break;
        }
        // 2. 在当前点云上执行RANSAC分割
        seg.setInputCloud(current_cloud);
        seg.segment(*inlier_indices, *coefficients);

        // 3. 检查是否找到了内点
        if (inlier_indices->indices.empty())
        {
            std::cout << "RANSAC could not find any planar model in the remaining " << current_point_count << " points. Stopping." << std::endl;
            break;
        }
        // 4. 决策：检查找到的平面是否足够“显著”
        double inlier_percentage = static_cast<double>(inlier_indices->indices.size()) / current_point_count;
        if (inlier_percentage < params_.min_plane_inlier_percentage)
        {
            std::cout << "Found a plane with " << inlier_indices->indices.size() << " inliers ("
                      << "which is below the threshold of " << params_.min_plane_inlier_percentage * 100
                      << "%. Stopping plane detection." << std::endl;
            break;
        }

        std::cout << "Plane detected with " << inlier_indices->indices.size() << " inliers." << std::endl;

        DetectedPrimitive plane_primitive;
        plane_primitive.type = "plane";

        // 将平面系数(a,b,c,d)存入4x4矩阵。对于平面，很多项是0。
        // 对于平面 Ax+By+Cz+D=0，可以将其表示为 q^T*x = 0，其中 x=[x,y,z,1]^T
        // q = [A, B, C, D]^T。我们将其存储在矩阵的最后一列。
        plane_primitive.model_coefficients.setZero();
        plane_primitive.model_coefficients(0, 3) = coefficients->values[0]; // A
        plane_primitive.model_coefficients(1, 3) = coefficients->values[1]; // B
        plane_primitive.model_coefficients(2, 3) = coefficients->values[2]; // C
        plane_primitive.model_coefficients(3, 3) = coefficients->values[3]; // D

        extract.setInputCloud(current_cloud);
        extract.setIndices(inlier_indices);
        extract.setNegative(false); // false = 提取内点
        extract.filter(*(plane_primitive.inliers));

        detected_primitives_.push_back(plane_primitive);

        // 6. 更新点云：移除已找到的内点，为下一次迭代做准备
        extract.setNegative(true); // true = 移除内点，保留剩余部分
        PointCloudPtr remaining_points(new PointCloud());
        extract.filter(*remaining_points);
        current_cloud.swap(remaining_points); // 用剩余点云替换当前点云
    }

    remain_cloud.swap(current_cloud);

    std::cout << "--- Plane Detection Finished. " << remain_cloud->size() << " points remain. ---" << std::endl;
}

bool MSQ::findQuadric(const PointCloudPtr &cloud,
                      Eigen::Matrix4f &best_model_coefficients,
                      pcl::PointIndices::Ptr &best_inlier_indices)
{
    // ====================================================================
    // 步骤 0: 入口检查与参数初始化
    // ====================================================================
    const size_t total_points = cloud->size();
    const int MINIMUM_POINTS_REQUIRED = 9;

    if (total_points < MINIMUM_POINTS_REQUIRED)
    {
        // PCL_WARN("[findQuadric] Not enough points (%zu) to fit quadric. Minimum required is %d.", total_points, MINIMUM_POINTS_REQUIRED);
        return false;
    }

    // 初始化RANSAC的最佳结果
    size_t best_inlier_count = 0;
    best_inlier_indices.reset(new pcl::PointIndices());

    // 使用现代C++的随机数生成器
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distrib(0, total_points - 1);

    int max_iterations = params_.quadric_max_iterations;

    // ====================================================================
    // RANSAC 主循环
    // ====================================================================
    for (int iter = 0; iter < max_iterations; ++iter)
    {
        // --- 步骤 A: 生成候选模型 (Hypothesis Generation) ---

        // A.1. 不重复地随机采样9个点
        int sample_indices[MINIMUM_POINTS_REQUIRED];
        for (int i = 0; i < MINIMUM_POINTS_REQUIRED; ++i)
        {
            int current_idx;
            bool is_unique;
            do
            {
                is_unique = true;
                current_idx = distrib(gen);
                for (int j = 0; j < i; ++j)
                {
                    if (sample_indices[j] == current_idx)
                    {
                        is_unique = false;
                        break;
                    }
                }
            } while (!is_unique);
            sample_indices[i] = current_idx;
        }

        // A.2. 构建约束矩阵 A_basis
        Eigen::Matrix<double, MINIMUM_POINTS_REQUIRED, 10> A_basis;
        bool sample_valid = true;
        for (int i = 0; i < MINIMUM_POINTS_REQUIRED; ++i)
        {
            const auto &p = cloud->points[sample_indices[i]];
            if (!pcl::isFinite(p) || std::isnan(p.x) || std::isnan(p.y) || std::isnan(p.z))
            {
                sample_valid = false;
                break;
            }
            double x = p.x, y = p.y, z = p.z;
            A_basis.row(i) << x * x, y * y, z * z, 2 * x * y, 2 * x * z, 2 * y * z, 2 * x, 2 * y, 2 * z, 1.0;
        }
        if (!sample_valid)
        {
            continue; // 如果样本点无效，跳过本次迭代
        }

        // A.3. SVD求解
        Eigen::JacobiSVD<Eigen::Matrix<double, MINIMUM_POINTS_REQUIRED, 10>> svd(A_basis, Eigen::ComputeFullV);
        Eigen::VectorXd q = svd.matrixV().col(9);

        Eigen::Matrix4f candidate_model = vectorToQMatrix(q).cast<float>();

        // SVD求解结果的范数理论上为1，但为保险起见，检查并归一化
        float norm = candidate_model.norm();
        if (norm < 1e-6)
        {
            continue; // 退化模型，跳过
        }
        candidate_model /= norm;

        // --- 步骤 B: 验证候选模型 (Verification) ---
        pcl::PointIndices::Ptr current_inliers(new pcl::PointIndices());

        // 【优化点 1: 验证提前终止】
        // 在进入验证循环前，获取到目前为止找到的最佳内点数。
        const size_t best_inlier_count_snapshot = best_inlier_count;

        for (size_t i = 0; i < total_points; ++i)
        {
            // 【优化点 1: 验证提前终止】
            // 检查剩余的点数加上当前已找到的内点数，是否还有可能超过历史最佳。
            size_t remaining_points_to_check = total_points - i;
            if (current_inliers->indices.size() + remaining_points_to_check <= best_inlier_count_snapshot)
            {
                break; // 不可能超越历史最佳了，提前退出本次验证
            }

            const auto &p = cloud->points[i];
            Eigen::Vector4f pt_h(p.x, p.y, p.z, 1.0f);
            float dist = std::abs((pt_h.transpose() * candidate_model * pt_h).coeff(0, 0));

            if (dist < params_.quadric_distance_threshold)
            {
                current_inliers->indices.push_back(i);
            }
        }

        // --- 步骤 C: 更新最佳模型与局部优化 (Update & LO-RANSAC) ---
        bool update_max_iterations_flag = false;
        if (current_inliers->indices.size() > best_inlier_count)
        {
            best_inlier_count = current_inliers->indices.size();
            best_inlier_indices = current_inliers;
            best_model_coefficients = candidate_model;
            update_max_iterations_flag = true;

            // 【优化点 2: LO-RANSAC (局部优化采样)】
            // 如果这个新模型的内点数已经比较可观，我们触发一个短暂的、成本很低的局部搜索阶段。
            const int LO_MIN_INLIERS = MINIMUM_POINTS_REQUIRED * 3;
            const int LO_MAX_ITERATIONS = 10;

            if (best_inlier_count > LO_MIN_INLIERS)
            {
                std::uniform_int_distribution<> local_distrib(0, best_inlier_count - 1);

                for (int lo_iter = 0; lo_iter < LO_MAX_ITERATIONS; ++lo_iter)
                {
                    // A.1. 从【当前找到的内点集】中不重复地随机采样9个点的索引
                    int local_sample_indices[MINIMUM_POINTS_REQUIRED];
                    for (int i = 0; i < MINIMUM_POINTS_REQUIRED; ++i)
                    {
                        int current_idx;
                        bool is_unique;
                        do
                        {
                            is_unique = true;
                            current_idx = local_distrib(gen);
                            for (int j = 0; j < i; ++j)
                            {
                                if (local_sample_indices[j] == current_idx)
                                {
                                    is_unique = false;
                                    break;
                                }
                            }
                        } while (!is_unique);
                        local_sample_indices[i] = current_idx;
                    }

                    // A.2. 构建局部约束矩阵
                    Eigen::Matrix<double, MINIMUM_POINTS_REQUIRED, 10> A_basis_local;
                    bool local_sample_valid = true;
                    for (int i = 0; i < MINIMUM_POINTS_REQUIRED; ++i)
                    {
                        const int point_global_index = best_inlier_indices->indices[local_sample_indices[i]];
                        const auto &p = cloud->points[point_global_index];
                        if (!pcl::isFinite(p) || std::isnan(p.x) || std::isnan(p.y) || std::isnan(p.z))
                        {
                            local_sample_valid = false;
                            break;
                        }
                        double x = p.x, y = p.y, z = p.z;
                        A_basis_local.row(i) << x * x, y * y, z * z, 2 * x * y, 2 * x * z, 2 * y * z, 2 * x, 2 * y, 2 * z, 1.0;
                    }
                    if (!local_sample_valid)
                        continue;

                    // A.3. SVD求解
                    Eigen::JacobiSVD<Eigen::Matrix<double, MINIMUM_POINTS_REQUIRED, 10>> svd_local(A_basis_local, Eigen::ComputeFullV);
                    Eigen::VectorXd q_local = svd_local.matrixV().col(9);
                    Eigen::Matrix4f local_candidate_model = vectorToQMatrix(q_local).cast<float>();
                    float local_norm = local_candidate_model.norm();
                    if (local_norm < 1e-6)
                        continue;
                    local_candidate_model /= local_norm;

                    // 验证这个局部优化的模型
                    pcl::PointIndices::Ptr local_inliers(new pcl::PointIndices());
                    size_t local_best_inlier_count_snapshot = best_inlier_count; // 使用当前的全局最优值进行比较
                    for (size_t i = 0; i < total_points; ++i)
                    {
                        size_t remaining = total_points - i;
                        if (local_inliers->indices.size() + remaining <= local_best_inlier_count_snapshot)
                        {
                            break;
                        }
                        const auto &p = cloud->points[i];
                        Eigen::Vector4f pt_h(p.x, p.y, p.z, 1.0f);
                        float dist = std::abs((pt_h.transpose() * local_candidate_model * pt_h).coeff(0, 0));
                        if (dist < params_.quadric_distance_threshold)
                        {
                            local_inliers->indices.push_back(i);
                        }
                    }

                    // 如果局部搜索找到了一个更好的模型，就再次更新最佳记录
                    if (local_inliers->indices.size() > best_inlier_count)
                    {
                        best_inlier_count = local_inliers->indices.size();
                        best_inlier_indices = local_inliers;
                        best_model_coefficients = local_candidate_model;
                        update_max_iterations_flag = true;
                    }
                } // LO-RANSAC 循环结束
            } // LO-RANSAC 结束
        }

        // --- 步骤 D: 更新动态迭代次数 ---
        if (update_max_iterations_flag)
        {
            double inlier_ratio = static_cast<double>(best_inlier_count) / total_points;
            if (inlier_ratio > 0)
            { // 避免log(0)
                // PCL RANSAC的经典迭代次数更新公式
                double p_no_outliers = 1.0 - pow(inlier_ratio, MINIMUM_POINTS_REQUIRED);
                p_no_outliers = std::max(std::numeric_limits<double>::epsilon(), p_no_outliers);       // 避免log(0)
                p_no_outliers = std::min(1.0 - std::numeric_limits<double>::epsilon(), p_no_outliers); // 避免log(1)
                int new_max_iterations = static_cast<int>(log(1.0 - params_.min_quadric_inlier_percentage) / log(p_no_outliers));
                if (new_max_iterations < max_iterations)
                {
                    // std::cout << "Updating max_iterations from " << max_iterations << " to " << new_max_iterations << std::endl;
                    max_iterations = new_max_iterations;
                }
            }
        }
    } // RANSAC 主循环结束

    // --- 步骤 D: 模型修正与返回 ---
    // (逻辑与您原来的代码完全相同)
    // 在 findQuadric 函数中...
    // RANSAC 主循环结束

    // 在 findQuadric 函数中...
    // RANSAC 主循环结束

    // --- 步骤 D: 迭代式精炼-扩展 与 最终返回 ---
    const double min_inlier_relative = total_points * params_.min_quadric_inlier_percentage;

    if (best_inlier_count > std::max(static_cast<double>(MINIMUM_POINTS_REQUIRED), min_inlier_relative))
    {
    PCL_INFO("[findQuadric] RANSAC found a promising model with %zu inliers. Starting Iterative Refine-and-Expand...", best_inlier_count);

    Eigen::Matrix4f refined_model = best_model_coefficients; // 从RANSAC的最佳模型开始
    pcl::PointIndices::Ptr current_inliers = best_inlier_indices;
    
    const int MAX_REFINEMENT_ITERATIONS = 10; // 防止无限循环
    for (int i = 0; i < MAX_REFINEMENT_ITERATIONS; ++i)
    {
            size_t inliers_before_iteration = current_inliers->indices.size();

            // --- 1. 精炼模型 ---
            Eigen::MatrixXd A_refine(inliers_before_iteration, 10);
            for (size_t j = 0; j < inliers_before_iteration; ++j)
            {
                const auto &p = cloud->points[current_inliers->indices[j]];
                double x = p.x, y = p.y, z = p.z;
                A_refine.row(j) << x * x, y * y, z * z, 2 * x * y, 2 * x * z, 2 * y * z, 2 * x, 2 * y, 2 * z, 1.0;
            }

            Eigen::JacobiSVD<Eigen::MatrixXd> svd_refine(A_refine, Eigen::ComputeFullV);
            Eigen::VectorXd q_refined_vec = svd_refine.matrixV().col(9);
            // 将 Ax=0 转换为 A_sub*x_sub = b
            // Eigen::MatrixXd A_sub = A_refine.leftCols(9);
            // Eigen::VectorXd b = -A_refine.col(9);

            // // 使用速度更快的 ColPivHouseholderQr 分解求解
            // Eigen::VectorXd q_sub_refined = A_sub.colPivHouseholderQr().solve(b);
            // if (q_sub_refined.hasNaN() || q_sub_refined.norm() > 1e6)
            // {
            //     // 如果求解结果包含NaN(非数字)或者其范数(长度)异常大（说明数值爆炸了），
            //     // 意味着这个随机样本组成的矩阵是病态的。我们应该直接放弃这个样本。
            //     continue; // 直接进入下一次RANSAC迭代
            // }
            // Eigen::VectorXd q_refined_vec(10);
            // q_refined_vec.head(9) = q_sub_refined;
            // q_refined_vec(9) = 1.0; // 我们求解的是归一化到常数项为1的方程

            refined_model = vectorToQMatrix(q_refined_vec).cast<float>();
            refined_model.normalize();

            // --- 2. 扩展内点集 ---
            pcl::PointIndices::Ptr expanded_inliers(new pcl::PointIndices());
            expanded_inliers->indices.reserve(total_points);
            for (int k = 0; k < total_points; ++k)
            {
                Eigen::Vector4f pt_h(cloud->points[k].x, cloud->points[k].y, cloud->points[k].z, 1.0f);
                float dist = std::abs((pt_h.transpose() * refined_model * pt_h).coeff(0, 0));
                if (dist < params_.quadric_distance_threshold)
                {
                    expanded_inliers->indices.push_back(k);
                }
            }

            // --- 3. 检查是否收敛 ---
            if (expanded_inliers->indices.size() <= inliers_before_iteration)
            {
            PCL_INFO("[findQuadric] Refinement converged after %d iterations.", i + 1);
            break; // 内点不再增加，模型已稳定
            }

            current_inliers = expanded_inliers; // 更新内点集，进行下一次迭代
        PCL_INFO("[findQuadric] Iteration %d: Inlier set expanded to %zu.", i + 1, current_inliers->indices.size());
    }

    // --- 4. 最终检查与返回 ---
    if (current_inliers->indices.size() > std::max(static_cast<double>(MINIMUM_POINTS_REQUIRED), min_inlier_relative))
    {
            best_model_coefficients = refined_model;
            best_inlier_indices = current_inliers;
            return true;
    }
    }

PCL_INFO("[findQuadric] RANSAC failed. Best model had %zu inliers, but did not pass final checks.", best_inlier_count);
return false;
}

/**
 * @brief 在点云中循环检测二次曲面。
 * @param remain_cloud [输入/输出] 待处理的点云，函数会从中移除已识别的基元。
 */
void MSQ::detectQuadric(PointCloudPtr &remain_cloud)
{
    ScopedTimer total_timer("Total quadric detection process");
    const size_t initial_point_count = remain_cloud->size();
    if (initial_point_count == 0)
        return;
    // --- 循环检测，直到剩余点太少 ---
    while (true)
    {
        // 检查剩余点数是否满足继续检测的条件
        if (remain_cloud->size() < initial_point_count * params_.min_remaining_points_percentage)
        {
            std::cout << "Remaining points too few. Stopping quadric detection." << std::endl;
            break;
        }

        Eigen::Matrix4f model_coefficients;
        pcl::PointIndices::Ptr inlier_indices(new pcl::PointIndices);

        // 调用内部核心函数，尝试在当前剩余点云中找到一个最佳二次曲面
        bool found = findQuadric(remain_cloud, model_coefficients, inlier_indices);

        if (found)
        {
            if (inlier_indices->indices.size() < 75)
            { // 50只是一个例子，您可以设为100
                PCL_WARN("Found a primitive with only %zu inliers. Considering it as noise and stopping detection.", inlier_indices->indices.size());
                break; // 停止检测，因为剩下的都是碎片了
            }
            // 找到了一个有效的二次曲面
            std::cout << "Found a quadric with " << inlier_indices->indices.size() << " inliers." << std::endl;

            // 1. 保存检测结果
            DetectedPrimitive primitive;
            primitive.type = "quadric";
            primitive.model_coefficients = model_coefficients;

            // 从 remain_cloud 中提取内点，保存到基元信息中
            pcl::ExtractIndices<PointT> extract_inliers;
            extract_inliers.setInputCloud(remain_cloud);
            extract_inliers.setIndices(inlier_indices);
            extract_inliers.setNegative(false); // false = 提取内点
            extract_inliers.filter(*(primitive.inliers));

            detected_primitives_.push_back(primitive);

            // 2. 从 remain_cloud 中移除内点，为下一次循环做准备
            pcl::ExtractIndices<PointT> extract_outliers;
            extract_outliers.setInputCloud(remain_cloud);
            extract_outliers.setIndices(inlier_indices);
            extract_outliers.setNegative(true); // true = 移除内点, 保留外点
            PointCloudPtr next_remain_cloud(new PointCloud());
            extract_outliers.filter(*next_remain_cloud);

            remain_cloud = next_remain_cloud; // 更新剩余点云
        }
        else
        {
            // 在当前剩余点云中再也找不到满足条件的二次曲面了
            std::cout << "No more quadrics could be found." << std::endl;
            break; // 结束循环
        }
    }
}
/**
 * @brief 将4x4的二次曲面矩阵Q转换为10维参数向量q。
 * @param Q 输入的4x4对称矩阵。
 * @return 包含10个二次曲面系数的向量 [A,B,C,D,E,F,G,H,I,J]。
 */
Eigen::VectorXd MinimalSampleQuadric::QMatrixToVector(const Eigen::Matrix4f &Q) const
{
    Eigen::VectorXd q_vec(10);

    // 根据映射关系从矩阵中提取系数
    // Q(row, col)
    q_vec(0) = Q(0, 0); // A: x^2 coeff
    q_vec(1) = Q(1, 1); // B: y^2 coeff
    q_vec(2) = Q(2, 2); // C: z^2 coeff

    // 注意：非对角线元素对应的是 2*D*xy, 2*E*xz 等，所以我们直接取上三角部分
    q_vec(3) = Q(0, 1); // D: xy coeff
    q_vec(4) = Q(0, 2); // E: xz coeff
    q_vec(5) = Q(1, 2); // F: yz coeff

    q_vec(6) = Q(0, 3); // G: x coeff
    q_vec(7) = Q(1, 3); // H: y coeff
    q_vec(8) = Q(2, 3); // I: z coeff

    q_vec(9) = Q(3, 3); // J: constant term

    return q_vec;
}