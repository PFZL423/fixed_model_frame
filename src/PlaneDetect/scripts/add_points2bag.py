#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TUM 风格 RGB-D bag → 带 XYZRGB 的 sensor_msgs/PointCloud2 bag。
可选 YAML 配置话题名、深度比例、是否 passthrough、是否去掉 frame 前导 '/'。
"""
from __future__ import print_function

import argparse
import copy
import os
import sys

import cv2
import numpy as np
import rosbag
import rospy
import yaml
from cv_bridge import CvBridge
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from tf2_msgs.msg import TFMessage

TF_TOPICS = ("/tf", "/tf_static")


def strip_leading_slash(frame_id):
    if frame_id and frame_id.startswith("/"):
        return frame_id[1:]
    return frame_id


def strip_msg_frames(msg, strip_tf):
    """原地修改消息中的 frame_id。"""
    if hasattr(msg, "header") and msg.header is not None:
        msg.header.frame_id = strip_leading_slash(msg.header.frame_id)
    if strip_tf and isinstance(msg, TFMessage):
        for transform in msg.transforms:
            transform.header.frame_id = strip_leading_slash(transform.header.frame_id)
            transform.child_frame_id = strip_leading_slash(transform.child_frame_id)


def create_pc2_numpy(depth_img, rgb_img, fx, fy, cx, cy, depth_scale):
    """由深度图 + BGR 彩色图生成 [(x,y,z,rgb_u32), ...]。"""
    h, w = depth_img.shape[:2]
    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))

    z = depth_img.astype(np.float32) / float(depth_scale)
    z[z <= 0] = np.nan

    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)

    rgb_img = cv2.resize(rgb_img, (w, h), interpolation=cv2.INTER_LINEAR)
    b = rgb_img[:, :, 0].flatten().astype(np.uint32)
    g = rgb_img[:, :, 1].flatten().astype(np.uint32)
    r = rgb_img[:, :, 2].flatten().astype(np.uint32)
    rgb_packed = (r << 16) | (g << 8) | b

    mask = ~np.isnan(points[:, 2])
    if not np.any(mask):
        return []

    vp = points[mask]
    vr = rgb_packed[mask]
    # list of tuples for create_cloud（比逐点 append 快得多）
    return list(zip(vp[:, 0], vp[:, 1], vp[:, 2], vr))


def load_yaml(path):
    if not path:
        return {}
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def camera_k(msg):
    """sensor_msgs CameraInfo K: [fx, 0, cx, 0, fy, cy, ...]"""
    k = msg.K
    return float(k[0]), float(k[4]), float(k[2]), float(k[5])


def main():
    parser = argparse.ArgumentParser(
        description="TUM RGB-D bag → PointCloud2 (XYZRGB) + 可选原话题 passthrough"
    )
    parser.add_argument("-c", "--config", type=str, default=None, help="YAML 配置文件路径")
    parser.add_argument("inputbag", nargs="?", help="输入 bag")
    parser.add_argument("outputbag", nargs="?", help="输出 bag（默认同目录加后缀）")
    parser.add_argument("--start", type=float, default=None, help="跳过前 N 秒（覆盖 yaml）")
    parser.add_argument("--duration", type=float, default=None, help="最长处理时长（秒，覆盖 yaml）")
    parser.add_argument("--nth", type=int, default=None, help="每隔 N 帧取一帧（覆盖 yaml）")
    args = parser.parse_args()

    cfg = load_yaml(args.config) if args.config else {}

    input_bag = args.inputbag or cfg.get("input_bag") or cfg.get("inputbag")
    if not input_bag:
        print("错误：请指定输入 bag（命令行第一个参数）或在 yaml 中设置 input_bag", file=sys.stderr)
        sys.exit(1)

    output_bag = args.outputbag or cfg.get("output_bag") or cfg.get("outputbag")
    if not output_bag:
        output_bag = os.path.splitext(input_bag)[0] + "-points-p3.bag"

    topics = cfg.get("topics") or {}
    t_depth = topics.get("depth_image", "/camera/depth/image")
    t_rgb = topics.get("rgb_image", "/camera/rgb/image_color")
    t_depth_info = topics.get("depth_camera_info", "/camera/depth/camera_info")
    t_rgb_info = topics.get("rgb_camera_info", "/camera/rgb/camera_info")
    t_out_pc = cfg.get("output_pointcloud") or topics.get("output_pointcloud", "/camera/depth/points")

    depth_scale = float(cfg.get("depth_scale", 5000.0))
    sync_slop = float(cfg.get("sync_slop_sec", 0.033))
    nth = int(args.nth if args.nth is not None else cfg.get("nth_frame", 1))
    start_sec = float(args.start if args.start is not None else cfg.get("start_sec", 0.0))
    dur = args.duration
    if dur is None and cfg.get("duration_sec") is not None:
        dur = float(cfg["duration_sec"])

    passthrough_all = bool(cfg.get("passthrough_all", True))
    do_strip = bool(cfg.get("strip_leading_slash", True))
    pc_frame_override = (cfg.get("pointcloud_frame_id") or "").strip()

    rospy.init_node("add_points2bag", anonymous=True, disable_signals=True)
    bridge = CvBridge()

    depth_info = None
    rgb_info = None
    rgb_msg = None
    frame_count = 0
    time_start = None
    pc_written = 0

    print("输入: {}".format(input_bag))
    print("输出: {}".format(output_bag))
    print("深度比例: {} | 同步门限: {} s | passthrough 全部话题: {}".format(
        depth_scale, sync_slop, passthrough_all))

    with rosbag.Bag(input_bag, "r") as inbag:
        with rosbag.Bag(output_bag, "w") as outbag:
            for topic, msg, t in inbag.read_messages():
                if time_start is None:
                    time_start = t

                curr_time = (t - time_start).to_sec()
                if curr_time < start_sec:
                    continue
                if dur is not None and curr_time > (start_sec + dur):
                    break

                if do_strip:
                    strip_msg_frames(msg, topic in TF_TOPICS)

                if topic == t_depth_info:
                    depth_info = msg
                elif topic == t_rgb_info:
                    rgb_info = msg
                elif topic == t_rgb:
                    rgb_msg = msg
                elif topic == t_depth and rgb_msg is not None and (
                    depth_info is not None or rgb_info is not None
                ):
                    cam_info = depth_info if depth_info is not None else rgb_info
                    dt = abs((msg.header.stamp - rgb_msg.header.stamp).to_sec())
                    if dt < sync_slop:
                        frame_count += 1
                        if frame_count % nth != 0:
                            if passthrough_all or topic in TF_TOPICS:
                                outbag.write(topic, msg, t)
                            continue

                        try:
                            cv_depth = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
                            cv_rgb = bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
                        except Exception as e:
                            rospy.logwarn("图像解码失败: %s", e)
                            if passthrough_all or topic in TF_TOPICS:
                                outbag.write(topic, msg, t)
                            continue

                        fx, fy, cx, cy = camera_k(cam_info)
                        pc_data = create_pc2_numpy(cv_depth, cv_rgb, fx, fy, cx, cy, depth_scale)
                        # if len(pc_data) > 0:
                        #     sample = pc_data[len(pc_data) // 2]
                        #     print("DEBUG: fx={}, fy={}, cx={}, cy={}".format(fx, fy, cx, cy))
                        #     print(
                        #         "DEBUG: 样本点坐标: X={:.4f}, Y={:.4f}, Z={:.4f}".format(
                        #             sample[0], sample[1], sample[2]
                        #         )
                        #     )
                        #     sample_depth = cv_depth[cv_depth.shape[0] // 2, cv_depth.shape[1] // 2]
                        #     print(
                        #         "DEBUG: 图像中心原始深度值: {}, 转换后的Z: {}".format(
                        #             sample_depth, float(sample_depth) / depth_scale
                        #         )
                        #     )

                        if not pc_data:
                            if passthrough_all or topic in TF_TOPICS:
                                outbag.write(topic, msg, t)
                            continue

                        hdr = copy.deepcopy(msg.header)
                        if pc_frame_override:
                            hdr.frame_id = pc_frame_override

                        fields = [
                            PointField("x", 0, PointField.FLOAT32, 1),
                            PointField("y", 4, PointField.FLOAT32, 1),
                            PointField("z", 8, PointField.FLOAT32, 1),
                            PointField("rgb", 12, PointField.UINT32, 1),
                        ]
                        pc2_msg = pc2.create_cloud(hdr, fields, pc_data)
                        outbag.write(t_out_pc, pc2_msg, t)
                        pc_written += 1
                        print("已写点云帧: {} | bag 时间 {:.2f} s\r".format(pc_written, curr_time), end="")
                        sys.stdout.flush()

                if passthrough_all:
                    outbag.write(topic, msg, t)
                elif topic in TF_TOPICS:
                    outbag.write(topic, msg, t)

    print("\n完成。输出: {} | 共 {} 帧点云".format(output_bag, pc_written))


if __name__ == "__main__":
    main()
