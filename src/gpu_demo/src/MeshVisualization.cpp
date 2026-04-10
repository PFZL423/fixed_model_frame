#include "gpu_demo/MeshVisualization.h"
#include "gpu_demo/tinycolormap.hpp"

#include <opencv2/imgproc.hpp>
#include <ros/ros.h>
#include <std_msgs/ColorRGBA.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>

namespace mesh_viz
{
namespace
{

struct Local3
{
    float x, y, z;
};

static Local3 transformToGlobalLocal(const Local3 &pt_local, const float transform[12])
{
    float p[3] = {transform[3], transform[7], transform[11]};
    Local3 g;
    g.x = transform[0]*pt_local.x + transform[1]*pt_local.y + transform[2]*pt_local.z + p[0];
    g.y = transform[4]*pt_local.x + transform[5]*pt_local.y + transform[6]*pt_local.z + p[1];
    g.z = transform[8]*pt_local.x + transform[9]*pt_local.y + transform[10]*pt_local.z + p[2];
    return g;
}

static float edgeMax(float x1, float y1, float x2, float y2, float x3, float y3)
{
    float e01 = std::hypot(x1 - x2, y1 - y2);
    float e12 = std::hypot(x2 - x3, y2 - y3);
    float e20 = std::hypot(x3 - x1, y3 - y1);
    return std::max(e01, std::max(e12, e20));
}

static float edgeMin(float x1, float y1, float x2, float y2, float x3, float y3)
{
    float e01 = std::hypot(x1 - x2, y1 - y2);
    float e12 = std::hypot(x2 - x3, y2 - y3);
    float e20 = std::hypot(x3 - x1, y3 - y1);
    return std::min(e01, std::min(e12, e20));
}

static bool isTriangleSliver(float x1, float y1, float x2, float y2, float x3, float y3, double ratio_max)
{
    if (ratio_max <= 0.0)
        return false;
    float emin = edgeMin(x1, y1, x2, y2, x3, y3);
    float emax = edgeMax(x1, y1, x2, y2, x3, y3);
    if (emin < 1e-15f)
        return true;
    return (emax / emin) > static_cast<float>(ratio_max);
}

/// 去重：将点映射到网格格子，每格只保留一个点
static std::vector<cv::Point2f> dedupePoints2dGrid(const std::vector<cv::Point2f> &in, float cell)
{
    std::map<std::pair<int, int>, cv::Point2f> uniq;
    for (const auto &p : in)
    {
        if (!std::isfinite(p.x) || !std::isfinite(p.y))
            continue;
        int kx = static_cast<int>(std::llround(p.x / cell));
        int ky = static_cast<int>(std::llround(p.y / cell));
        uniq.emplace(std::make_pair(kx, ky), p);
    }
    std::vector<cv::Point2f> out;
    out.reserve(uniq.size());
    for (const auto &kv : uniq)
        out.push_back(kv.second);
    return out;
}

/// 三角形外接圆半径（Alpha Shape 的核心判据）
static float circumradius(float x1, float y1, float x2, float y2, float x3, float y3)
{
    float ax = x2 - x1, ay = y2 - y1;
    float bx = x3 - x1, by = y3 - y1;
    float D = 2.f * (ax * by - ay * bx);
    if (std::fabs(D) < 1e-12f)
        return std::numeric_limits<float>::max();
    float ux = (by * (ax*ax + ay*ay) - ay * (bx*bx + by*by)) / D;
    float uy = (ax * (bx*bx + by*by) - bx * (ax*ax + ay*ay)) / D;
    return std::sqrt(ux*ux + uy*uy);
}

/// Alpha Shape + Delaunay 三角化
/// concave_alpha <= 0 时自动取 0.005 * bbox_diagonal（与论文一致）
static bool triangulateConcaveDelaunay(const std::vector<cv::Point2f> &pts2d,
                                       double concave_alpha,
                                       double max_edge,
                                       bool /*clip_to_hull*/,
                                       bool /*clip_hull_vertices_inside*/,
                                       double sliver_max_edge_ratio,
                                       std::vector<cv::Vec6f> &triangles_out)
{
    triangles_out.clear();
    if (pts2d.size() < 3)
        return false;

    float minx = std::numeric_limits<float>::max();
    float miny = std::numeric_limits<float>::max();
    float maxx = std::numeric_limits<float>::lowest();
    float maxy = std::numeric_limits<float>::lowest();
    for (const auto &p : pts2d)
    {
        if (!std::isfinite(p.x) || !std::isfinite(p.y))
            continue;
        minx = std::min(minx, p.x);
        miny = std::min(miny, p.y);
        maxx = std::max(maxx, p.x);
        maxy = std::max(maxy, p.y);
    }
    if (maxx <= minx || maxy <= miny)
        return false;

    const float spanx = maxx - minx;
    const float spany = maxy - miny;
    const float span = std::max(spanx, spany);
    const float bbox_diag = std::sqrt(spanx * spanx + spany * spany);

    // 自适应 alpha：论文用 0.005 * bbox_diagonal
    float alpha_r = (concave_alpha > 1e-9)
                        ? static_cast<float>(concave_alpha)
                        : 0.005f * bbox_diag;

    const float cell = std::max(1e-6f, span * 1e-5f);
    std::vector<cv::Point2f> insert_pts = dedupePoints2dGrid(pts2d, cell);
    if (insert_pts.size() < 3u)
        return false;

    // 重新计算去重后的 bbox
    minx = std::numeric_limits<float>::max();
    miny = std::numeric_limits<float>::max();
    maxx = std::numeric_limits<float>::lowest();
    maxy = std::numeric_limits<float>::lowest();
    for (const auto &p : insert_pts)
    {
        minx = std::min(minx, p.x);
        miny = std::min(miny, p.y);
        maxx = std::max(maxx, p.x);
        maxy = std::max(maxy, p.y);
    }

    float pad = std::max(alpha_r * 2.f, static_cast<float>(max_edge) * 2.f);
    const float span_margin = span * 0.02f;
    const int kSlackPix = 32;
    int ix0 = static_cast<int>(std::floor(minx - pad - span_margin)) - kSlackPix;
    int iy0 = static_cast<int>(std::floor(miny - pad - span_margin)) - kSlackPix;
    int ix1 = static_cast<int>(std::ceil(maxx + pad + span_margin)) + kSlackPix;
    int iy1 = static_cast<int>(std::ceil(maxy + pad + span_margin)) + kSlackPix;
    int rw = ix1 - ix0;
    int rh = iy1 - iy0;
    if (rw < 1 || rh < 1)
        return false;

    cv::Rect bounds(ix0, iy0, rw, rh);
    cv::Subdiv2D subdiv(bounds);

    const float eps = std::max(1e-7f, std::min(0.25f,
        1e-4f * std::min(static_cast<float>(rw), static_cast<float>(rh))));
    float fxmin = static_cast<float>(bounds.x) + eps;
    float fxmax = static_cast<float>(bounds.x + bounds.width) - eps;
    float fymin = static_cast<float>(bounds.y) + eps;
    float fymax = static_cast<float>(bounds.y + bounds.height) - eps;

    try
    {
        for (const auto &p : insert_pts)
        {
            float px = std::max(fxmin, std::min(p.x, fxmax));
            float py = std::max(fymin, std::min(p.y, fymax));
            subdiv.insert(cv::Point2f(px, py));
        }
    }
    catch (const cv::Exception &e)
    {
        ROS_WARN_THROTTLE(5.0, "[MeshVisualization] Subdiv2D insert failed: %s", e.what());
        return false;
    }

    std::vector<cv::Vec6f> tri_list;
    subdiv.getTriangleList(tri_list);

    for (const auto &t : tri_list)
    {
        float x1 = t[0], y1 = t[1];
        float x2 = t[2], y2 = t[3];
        float x3 = t[4], y3 = t[5];

        // 丢弃 Delaunay 超级三角形残留（重心超出点云范围）
        float cx = (x1 + x2 + x3) / 3.f;
        float cy = (y1 + y2 + y3) / 3.f;
        if (cx < minx - alpha_r || cx > maxx + alpha_r ||
            cy < miny - alpha_r || cy > maxy + alpha_r)
            continue;

        // Alpha Shape 核心：外接圆半径 > alpha 则丢弃
        if (circumradius(x1, y1, x2, y2, x3, y3) > alpha_r)
            continue;

        if (edgeMax(x1, y1, x2, y2, x3, y3) > static_cast<float>(max_edge))
            continue;

        if (isTriangleSliver(x1, y1, x2, y2, x3, y3, sliver_max_edge_ratio))
            continue;

        triangles_out.push_back(t);
    }
    return !triangles_out.empty();
}

} // namespace

bool buildQuadricVisualizationMarker(const pcl::PointCloud<pcl::PointXYZI>::ConstPtr &inliers_global,
                                       const float explicit_coeffs[6],
                                       const float transform[12],
                                       const std_msgs::Header &header,
                                       const MeshVizParams &params,
                                       visualization_msgs::Marker &marker_out)
{
    if (!inliers_global || inliers_global->empty())
        return false;

    std::vector<cv::Point2f> pts2d;
    pts2d.reserve(inliers_global->size());
    for (const auto &pt : inliers_global->points)
    {
        float dx = pt.x - transform[3];
        float dy = pt.y - transform[7];
        float dz = pt.z - transform[11];
        float lx = transform[0]*dx + transform[4]*dy + transform[8]*dz;
        float ly = transform[1]*dx + transform[5]*dy + transform[9]*dz;
        pts2d.emplace_back(lx, ly);
    }

    std::vector<cv::Vec6f> tris;
    if (!triangulateConcaveDelaunay(pts2d, params.concave_alpha, params.delaunay_max_edge,
                                    params.clip_to_hull, params.clip_hull_vertices_inside,
                                    params.sliver_max_edge_ratio, tris))
        return false;

    // 计算曲率 scalar 用于 colormap
    std::vector<float> z_locals, scalars;
    z_locals.reserve(tris.size() * 3);
    scalars.reserve(tris.size() * 3);
    for (const auto &t : tris)
    {
        for (int k = 0; k < 3; ++k)
        {
            float x = t[k * 2], y = t[k * 2 + 1];
            float z = explicit_coeffs[0]*x*x + explicit_coeffs[1]*x*y +
                      explicit_coeffs[2]*y*y + explicit_coeffs[3]*x +
                      explicit_coeffs[4]*y + explicit_coeffs[5];
            z_locals.push_back(z);
            float gx = 2.f*explicit_coeffs[0]*x + explicit_coeffs[1]*y + explicit_coeffs[3];
            float gy = explicit_coeffs[1]*x + 2.f*explicit_coeffs[2]*y + explicit_coeffs[4];
            scalars.push_back(std::sqrt(gx*gx + gy*gy));
        }
    }
    float smin = *std::min_element(scalars.begin(), scalars.end());
    float smax = *std::max_element(scalars.begin(), scalars.end());
    float sden = (smax - smin < 1e-12f) ? 1.f : (smax - smin);

    marker_out.header = header;
    marker_out.ns = "quadric_surfaces";
    marker_out.type = visualization_msgs::Marker::TRIANGLE_LIST;
    marker_out.action = visualization_msgs::Marker::ADD;
    marker_out.pose.orientation.w = 1.0;
    marker_out.scale.x = marker_out.scale.y = marker_out.scale.z = 1.0;
    marker_out.color.a = params.mesh_alpha;

    size_t idx = 0;
    for (const auto &t : tris)
    {
        for (int k = 0; k < 3; ++k)
        {
            float x = t[k*2], y = t[k*2+1], z = z_locals[idx];
            Local3 g = transformToGlobalLocal({x, y, z}, transform);
            geometry_msgs::Point p;
            p.x = g.x; p.y = g.y; p.z = g.z;
            marker_out.points.push_back(p);
            double sm = std::max(0.0, std::min(1.0, (scalars[idx] - smin) / (double)sden));
            tinycolormap::Color tc = tinycolormap::GetColor(sm, tinycolormap::ColormapType::Plasma);
            std_msgs::ColorRGBA c;
            c.r = tc.r(); c.g = tc.g(); c.b = tc.b(); c.a = params.mesh_alpha;
            marker_out.colors.push_back(c);
            ++idx;
        }
    }
    return !marker_out.points.empty();
}

bool buildPlaneVisualizationMarker(const pcl::PointCloud<pcl::PointXYZI>::Ptr &inliers,
                                   const Eigen::Vector3f &p0,
                                   const Eigen::Vector3f &u_axis,
                                   const Eigen::Vector3f &v_axis,
                                   const Eigen::Vector3f &unit_normal,
                                   const std_msgs::Header &header,
                                   const MeshVizParams &params,
                                   visualization_msgs::Marker &marker_out)
{
    if (!inliers || inliers->size() < 3)
        return false;

    std::vector<cv::Point2f> pts2d;
    pts2d.reserve(inliers->size());
    for (const auto &pt : inliers->points)
    {
        Eigen::Vector3f d(pt.x - p0.x(), pt.y - p0.y(), pt.z - p0.z());
        pts2d.emplace_back(u_axis.dot(d), v_axis.dot(d));
    }

    std::vector<cv::Vec6f> tris;
    if (!triangulateConcaveDelaunay(pts2d, params.concave_alpha, params.delaunay_max_edge,
                                    params.clip_to_hull, params.clip_hull_vertices_inside,
                                    params.sliver_max_edge_ratio, tris))
        return false;

    marker_out.header = header;
    marker_out.ns = "planes";
    marker_out.type = visualization_msgs::Marker::TRIANGLE_LIST;
    marker_out.action = visualization_msgs::Marker::ADD;
    marker_out.pose.orientation.w = 1.0;
    marker_out.scale.x = marker_out.scale.y = marker_out.scale.z = 1.0;
    marker_out.color.a = params.mesh_alpha;

    // 平面用 flat_color（若设置了）；否则不填 colors，由调用方的 marker.color 统一着色
    bool use_flat = (params.flat_color.a > 0.f);
    for (const auto &t : tris)
    {
        for (int k = 0; k < 3; ++k)
        {
            float u = t[k*2], v = t[k*2+1];
            Eigen::Vector3f pw = p0 + u * u_axis + v * v_axis;
            geometry_msgs::Point p;
            p.x = pw.x(); p.y = pw.y(); p.z = pw.z();
            marker_out.points.push_back(p);
            if (use_flat)
                marker_out.colors.push_back(params.flat_color);
        }
    }
    return !marker_out.points.empty();
}

} // namespace mesh_viz
