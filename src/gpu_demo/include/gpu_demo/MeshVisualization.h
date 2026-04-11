#pragma once

#include <vector>
#include <string>
#include <opencv2/core.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <Eigen/Core>
#include <visualization_msgs/Marker.h>
#include <std_msgs/Header.h>
#include <std_msgs/ColorRGBA.h>

namespace mesh_viz
{

/// Alpha Shape + Delaunay 可视化参数
struct MeshVizParams
{
    bool use_concave_mesh = true;
    /// Alpha Shape 半径（米）；<=0 时自动取 0.005 * bbox_diagonal（推荐）
    double concave_alpha = 0.0;
    /// Delaunay 三角形最大边长（米），过大则丢弃
    double delaunay_max_edge = 2.0;
    /// 最长边 / 最短边 上限，剔除细长三角；≤0 关闭
    double sliver_max_edge_ratio = 0.0;  // ≤0 关闭；可视化场景无需剔除细长三角
    /// Marker 整体透明度 [0,1]
    float mesh_alpha = 0.7f;
    /// 保留兼容字段（Alpha Shape 不再需要，但调用方可能仍设置）
    bool clip_to_hull = true;
    bool clip_hull_vertices_inside = true;
    /// 平面纯色（a>0 时启用，每个平面传入不同颜色；a=0 则不填 colors，由 marker.color 统一着色）
    std_msgs::ColorRGBA flat_color{};
};

/// 二次曲面：凹包 + Delaunay + 抬升 + tinycolormap（不依赖 QuadricDetect 头，避免 Thrust 链入纯 CXX 目标）
bool buildQuadricVisualizationMarker(const pcl::PointCloud<pcl::PointXYZI>::ConstPtr &inliers_global,
                                       const float explicit_coeffs[6],
                                       const float transform[12],
                                       const std_msgs::Header &header,
                                       const MeshVizParams &params,
                                       visualization_msgs::Marker &marker_out);

/// 平面：内点投到 (u,v)，同流程；法线为常向量
bool buildPlaneVisualizationMarker(const pcl::PointCloud<pcl::PointXYZI>::Ptr &inliers,
                                   const Eigen::Vector3f &p0,
                                   const Eigen::Vector3f &u_axis,
                                   const Eigen::Vector3f &v_axis,
                                   const Eigen::Vector3f &unit_normal,
                                   const std_msgs::Header &header,
                                   const MeshVizParams &params,
                                   visualization_msgs::Marker &marker_out);

} // namespace mesh_viz
