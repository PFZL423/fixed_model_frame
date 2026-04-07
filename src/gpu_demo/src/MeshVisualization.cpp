#include "gpu_demo/MeshVisualization.h"
#include "gpu_demo/tinycolormap.hpp"

#include <pcl/surface/concave_hull.h>
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
    float R[9] = {transform[0], transform[1], transform[2],
                  transform[4], transform[5], transform[6],
                  transform[8], transform[9], transform[10]};
    float p[3] = {transform[3], transform[7], transform[11]};
    Local3 g;
    g.x = R[0] * pt_local.x + R[1] * pt_local.y + R[2] * pt_local.z + p[0];
    g.y = R[3] * pt_local.x + R[4] * pt_local.y + R[5] * pt_local.z + p[1];
    g.z = R[6] * pt_local.x + R[7] * pt_local.y + R[8] * pt_local.z + p[2];
    return g;
}

static bool pointInPolygonXY(float x, float y, const std::vector<cv::Point2f> &poly)
{
    if (poly.size() < 3)
        return false;
    bool inside = false;
    size_t n = poly.size();
    for (size_t i = 0, j = n - 1; i < n; j = i++)
    {
        const cv::Point2f &pi = poly[i];
        const cv::Point2f &pj = poly[j];
        if (((pi.y > y) != (pj.y > y)) && (pj.y != pi.y))
        {
            float x_intersect = (pj.x - pi.x) * (y - pi.y) / (pj.y - pi.y) + pi.x;
            if (x < x_intersect)
                inside = !inside;
        }
    }
    return inside;
}

/// 凹包内或边上（用于三顶点裁剪）；比纯射线法对「顶点落在边上」更稳
static bool pointInHullPolygonInclusive(float x, float y, const std::vector<cv::Point2f> &poly)
{
    if (poly.size() < 3)
        return false;
    double r = cv::pointPolygonTest(poly, cv::Point2f(x, y), false);
    return r >= 0.0;
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

/// 细长三角（长边/短边过大），常见于凹包+Delaunay 在大平面边缘的辐射状毛刺
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

/// 将 ConcaveHull 输出点按极角排序成简单多边形
static bool orderHullRing(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr &hull_pts,
                          std::vector<cv::Point2f> &out_poly)
{
    if (!hull_pts || hull_pts->size() < 3)
        return false;
    float cx = 0.f, cy = 0.f;
    for (const auto &p : hull_pts->points)
    {
        cx += p.x;
        cy += p.y;
    }
    cx /= static_cast<float>(hull_pts->size());
    cy /= static_cast<float>(hull_pts->size());
    std::vector<std::pair<float, cv::Point2f>> ang;
    ang.reserve(hull_pts->size());
    for (const auto &p : hull_pts->points)
    {
        float dx = p.x - cx;
        float dy = p.y - cy;
        ang.push_back({std::atan2(dy, dx), cv::Point2f(p.x, p.y)});
    }
    std::sort(ang.begin(), ang.end(), [](const auto &a, const auto &b) { return a.first < b.first; });
    out_poly.clear();
    out_poly.reserve(ang.size());
    for (const auto &pr : ang)
        out_poly.push_back(pr.second);
    return out_poly.size() >= 3;
}

/// OpenCV Subdiv2D 对重复 (x,y) 极敏感，去重后再 insert 可显著降低 (-201) 概率
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

/// 凹包 + Delaunay；输出三角形顶点 (x,y) 三元组列表
static bool triangulateConcaveDelaunay(const std::vector<cv::Point2f> &pts2d, double concave_alpha,
                                       double max_edge, bool clip_to_hull, bool clip_hull_vertices_inside,
                                       double sliver_max_edge_ratio,
                                       std::vector<cv::Vec6f> &triangles_out)
{
    triangles_out.clear();
    if (pts2d.size() < 3)
        return false;

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    cloud->reserve(pts2d.size());
    for (const auto &p : pts2d)
        cloud->push_back(pcl::PointXYZ(p.x, p.y, 0.f));

    pcl::ConcaveHull<pcl::PointXYZ> ch;
    ch.setInputCloud(cloud);
    ch.setAlpha(static_cast<float>(concave_alpha));
    pcl::PointCloud<pcl::PointXYZ>::Ptr hull_cloud(new pcl::PointCloud<pcl::PointXYZ>);
    ch.reconstruct(*hull_cloud);

    std::vector<cv::Point2f> hull_poly;
    if (!orderHullRing(hull_cloud, hull_poly))
        return false;

    float minx = std::numeric_limits<float>::max();
    float miny = std::numeric_limits<float>::max();
    float maxx = std::numeric_limits<float>::lowest();
    float maxy = std::numeric_limits<float>::lowest();
    for (const auto &p : pts2d)
    {
        if (!std::isfinite(p.x) || !std::isfinite(p.y))
            return false;
        minx = std::min(minx, p.x);
        miny = std::min(miny, p.y);
        maxx = std::max(maxx, p.x);
        maxy = std::max(maxy, p.y);
    }
    const float spanx = maxx - minx;
    const float spany = maxy - miny;
    if (maxx < minx || maxy < miny)
        return false;
    if (spanx < 1e-12f && spany < 1e-12f)
        return false;

    const float span = std::max(spanx, spany);
    const float cell = std::max(1e-6f, span * 1e-5f);
    std::vector<cv::Point2f> insert_pts = dedupePoints2dGrid(pts2d, cell);
    if (insert_pts.size() < 3u)
        return false;

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

    float pad = std::max(1.f, static_cast<float>(max_edge) * 2.f);
    const float span_margin = span * 0.02f;

    // OpenCV Subdiv2D::insert：重复顶点、矩形略小均易触发 (-201)；去重后扩大 float 边距与像素 slack。
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

    const float eps = std::max(1e-7f, std::min(0.25f, 1e-4f * std::min(static_cast<float>(rw), static_cast<float>(rh))));
    float xmin = static_cast<float>(bounds.x) + eps;
    float xmax = static_cast<float>(bounds.x + bounds.width) - eps;
    float ymin = static_cast<float>(bounds.y) + eps;
    float ymax = static_cast<float>(bounds.y + bounds.height) - eps;
    if (xmin >= xmax || ymin >= ymax)
    {
        xmin = static_cast<float>(bounds.x) + 1e-6f;
        xmax = static_cast<float>(bounds.x + bounds.width) - 1e-6f;
        ymin = static_cast<float>(bounds.y) + 1e-6f;
        ymax = static_cast<float>(bounds.y + bounds.height) - 1e-6f;
    }

    try
    {
        for (const auto &p : insert_pts)
        {
            float px = std::max(xmin, std::min(p.x, xmax));
            float py = std::max(ymin, std::min(p.y, ymax));
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
        if (edgeMax(x1, y1, x2, y2, x3, y3) > static_cast<float>(max_edge))
            continue;
        if (clip_to_hull)
        {
            if (clip_hull_vertices_inside)
            {
                if (!pointInHullPolygonInclusive(x1, y1, hull_poly) ||
                    !pointInHullPolygonInclusive(x2, y2, hull_poly) ||
                    !pointInHullPolygonInclusive(x3, y3, hull_poly))
                    continue;
            }
            else
            {
                float cx = (x1 + x2 + x3) / 3.f;
                float cy = (y1 + y2 + y3) / 3.f;
                if (!pointInPolygonXY(cx, cy, hull_poly))
                    continue;
            }
        }
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
        // R^T * (p - p0)，与 QuadricDetect::transformToLocal 一致
        float R00 = transform[0], R01 = transform[1], R02 = transform[2];
        float R10 = transform[4], R11 = transform[5], R12 = transform[6];
        float R20 = transform[8], R21 = transform[9], R22 = transform[10];
        float lx = R00 * dx + R10 * dy + R20 * dz;
        float ly = R01 * dx + R11 * dy + R21 * dz;
        pts2d.emplace_back(lx, ly);
    }

    std::vector<cv::Vec6f> tris;
    if (!triangulateConcaveDelaunay(pts2d, params.concave_alpha, params.delaunay_max_edge, params.clip_to_hull,
                                    params.clip_hull_vertices_inside, params.sliver_max_edge_ratio, tris))
        return false;

    std::vector<float> z_locals;
    std::vector<float> scalars;
    z_locals.reserve(tris.size() * 3);
    scalars.reserve(tris.size() * 3);
    for (const auto &t : tris)
    {
        for (int k = 0; k < 3; ++k)
        {
            float x = t[k * 2];
            float y = t[k * 2 + 1];
            float z = explicit_coeffs[0] * x * x + explicit_coeffs[1] * x * y +
                      explicit_coeffs[2] * y * y + explicit_coeffs[3] * x +
                      explicit_coeffs[4] * y + explicit_coeffs[5];
            z_locals.push_back(z);
            float gx = 2.f * explicit_coeffs[0] * x + explicit_coeffs[1] * y +
                       explicit_coeffs[3];
            float gy = explicit_coeffs[1] * x + 2.f * explicit_coeffs[2] * y +
                       explicit_coeffs[4];
            scalars.push_back(std::sqrt(gx * gx + gy * gy));
        }
    }
    float smin = *std::min_element(scalars.begin(), scalars.end());
    float smax = *std::max_element(scalars.begin(), scalars.end());
    float sden = (smax - smin);
    if (sden < 1e-12f)
        sden = 1.f;

    marker_out.header = header;
    marker_out.ns = "quadric_surfaces";
    marker_out.type = visualization_msgs::Marker::TRIANGLE_LIST;
    marker_out.action = visualization_msgs::Marker::ADD;
    marker_out.pose.orientation.w = 1.0;
    marker_out.scale.x = 1.0;
    marker_out.scale.y = 1.0;
    marker_out.scale.z = 1.0;
    marker_out.color.a = params.mesh_alpha;

    size_t idx = 0;
    for (const auto &t : tris)
    {
        for (int k = 0; k < 3; ++k)
        {
            float x = t[k * 2];
            float y = t[k * 2 + 1];
            float z = z_locals[idx];
            Local3 local{x, y, z};
            Local3 g = transformToGlobalLocal(local, transform);
            geometry_msgs::Point p;
            p.x = g.x;
            p.y = g.y;
            p.z = g.z;
            marker_out.points.push_back(p);
            double sm = (scalars[idx] - smin) / static_cast<double>(sden);
            sm = std::max(0.0, std::min(1.0, sm));
            tinycolormap::Color tc = tinycolormap::GetColor(sm, tinycolormap::ColormapType::Viridis);
            std_msgs::ColorRGBA c;
            c.r = static_cast<float>(tc.r());
            c.g = static_cast<float>(tc.g());
            c.b = static_cast<float>(tc.b());
            c.a = params.mesh_alpha;
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
        Eigen::Vector3f d(pt.x, pt.y, pt.z);
        d -= p0;
        float u = u_axis.dot(d);
        float v = v_axis.dot(d);
        pts2d.emplace_back(u, v);
    }

    std::vector<cv::Vec6f> tris;
    if (!triangulateConcaveDelaunay(pts2d, params.concave_alpha, params.delaunay_max_edge, params.clip_to_hull,
                                    params.clip_hull_vertices_inside, params.sliver_max_edge_ratio, tris))
        return false;

    std::vector<float> radial;
    radial.reserve(tris.size() * 3);
    for (const auto &t : tris)
    {
        for (int k = 0; k < 3; ++k)
        {
            float u = t[k * 2];
            float v = t[k * 2 + 1];
            radial.push_back(std::sqrt(u * u + v * v));
        }
    }
    float rmin = *std::min_element(radial.begin(), radial.end());
    float rmax = *std::max_element(radial.begin(), radial.end());
    float rden = (rmax - rmin);
    if (rden < 1e-12f)
        rden = 1.f;

    marker_out.header = header;
    marker_out.ns = "planes";
    marker_out.type = visualization_msgs::Marker::TRIANGLE_LIST;
    marker_out.action = visualization_msgs::Marker::ADD;
    marker_out.pose.orientation.w = 1.0;
    marker_out.scale.x = 1.0;
    marker_out.scale.y = 1.0;
    marker_out.scale.z = 1.0;
    marker_out.color.a = params.mesh_alpha;

    size_t idx = 0;
    for (const auto &t : tris)
    {
        for (int k = 0; k < 3; ++k)
        {
            float u = t[k * 2];
            float v = t[k * 2 + 1];
            Eigen::Vector3f pw = p0 + u * u_axis + v * v_axis;
            geometry_msgs::Point p;
            p.x = pw.x();
            p.y = pw.y();
            p.z = pw.z();
            marker_out.points.push_back(p);
            double sm = (radial[idx] - rmin) / static_cast<double>(rden);
            sm = std::max(0.0, std::min(1.0, sm));
            tinycolormap::Color tc = tinycolormap::GetColor(sm, tinycolormap::ColormapType::Viridis);
            std_msgs::ColorRGBA c;
            c.r = static_cast<float>(tc.r());
            c.g = static_cast<float>(tc.g());
            c.b = static_cast<float>(tc.b());
            c.a = params.mesh_alpha;
            marker_out.colors.push_back(c);
            ++idx;
        }
    }
    return !marker_out.points.empty();
}

} // namespace mesh_viz
