#pragma once

#include <vector>
#include <string>
#include <opencv2/core.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <Eigen/Core>
#include <visualization_msgs/Marker.h>
#include <std_msgs/Header.h>

namespace mesh_viz
{

/// 与 PCL ConcaveHull 的 alpha（米）及 Delaunay 边长上限、透明度等
struct MeshVizParams
{
    bool use_concave_mesh = true;
    /// PCL ConcaveHull alpha（过小可能失败，由调用方回退）
    double concave_alpha = 0.08;
    /// Delaunay 三角形最大边长（米），过大则丢弃该三角形
    double delaunay_max_edge = 2.0;
    /// 最长边 / 最短边 上限，用于剔除细长三角（边缘辐射状毛刺）；≤0 表示关闭
    double sliver_max_edge_ratio = 28.0;
    /// Marker 整体透明度 [0,1]
    float mesh_alpha = 0.7f;
    /// 是否用凹多边形裁剪三角形
    bool clip_to_hull = true;
    /// true：三顶点均在凹包内（边界更整齐，推荐）；false：仅三角形重心在凹包内（旧行为，凹边界处易穿孔）
    bool clip_hull_vertices_inside = true;
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
