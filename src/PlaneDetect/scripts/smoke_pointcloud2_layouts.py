#!/usr/bin/env python3
"""
Offline smoke: build minimal PointCloud2 layouts (XYZI vs XYZRGB-like)
that match what plane_test expects (x,y,z required; optional intensity;
rgb present but ignored by the node).

Run: python3 smoke_pointcloud2_layouts.py
Requires: sensor_msgs (ROS Python path or pip ros-noetic-sensor-msgs in env).
"""
from __future__ import print_function

import struct
import sys

try:
    from sensor_msgs.msg import PointCloud2, PointField
except ImportError:
    print("Skip: sensor_msgs not importable (source devel/setup.bash)", file=sys.stderr)
    sys.exit(0)


FLOAT32 = PointField.FLOAT32
UINT32 = PointField.UINT32


def build_cloud(fields, point_step, pack_fn, n=4):
    msg = PointCloud2()
    msg.header.frame_id = "map"
    msg.height = 1
    msg.width = n
    msg.fields = fields
    msg.is_bigendian = False
    msg.point_step = point_step
    msg.row_step = point_step * n
    msg.is_dense = True
    buf = bytearray(msg.row_step)
    for i in range(n):
        pack_fn(buf, i * point_step, float(i) * 0.1, float(i) * 0.2, float(i) * 0.3)
    msg.data = bytes(buf)
    return msg


def pack_xyz_only(buf, off, x, y, z):
    struct.pack_into("<fff", buf, off, x, y, z)


def pack_xyz_intensity(buf, off, x, y, z):
    struct.pack_into("<ffff", buf, off, x, y, z, float(off % 255))


def pack_xyz_rgb(buf, off, x, y, z):
    # PCL-style packed rgb after xyz (offsets 0,4,8,12)
    struct.pack_into("<fff", buf, off, x, y, z)
    rgb = (255 << 16) | (128 << 8) | 64
    struct.pack_into("<I", buf, off + 12, rgb)


def check_parse_like_plane_test(msg, expect_strength):
    """Mirror plane_test.cpp field loop (subset)."""
    xo = yo = zo = io = ro = -1
    for f in msg.fields:
        if f.name == "x":
            xo, xd = f.offset, f.datatype
        elif f.name == "y":
            yo, yd = f.offset, f.datatype
        elif f.name == "z":
            zo, zd = f.offset, f.datatype
        elif f.name == "intensity":
            io = f.offset
        elif f.name == "reflectivity":
            ro = f.offset
    assert xo >= 0 and yo >= 0 and zo >= 0, "missing xyz"
    strength = io if io >= 0 else ro
    if expect_strength == "intensity":
        assert io >= 0, "expected intensity"
    elif expect_strength == "none":
        assert strength < 0, "expected no strength field"
    elif expect_strength == "reflectivity":
        assert ro >= 0 and io < 0, "expected reflectivity only"
    print("OK layout width={} point_step={} strength={}".format(
        msg.width, msg.point_step,
        "intensity" if io >= 0 else ("reflectivity" if ro >= 0 else "none (I=0)")))


def main():
    # XYZ + intensity
    f_xyz_i = [
        PointField("x", 0, FLOAT32, 1),
        PointField("y", 4, FLOAT32, 1),
        PointField("z", 8, FLOAT32, 1),
        PointField("intensity", 12, FLOAT32, 1),
    ]
    m1 = build_cloud(f_xyz_i, 16, pack_xyz_intensity)
    check_parse_like_plane_test(m1, "intensity")

    # XYZ + rgb (no intensity)
    f_xyz_rgb = [
        PointField("x", 0, FLOAT32, 1),
        PointField("y", 4, FLOAT32, 1),
        PointField("z", 8, FLOAT32, 1),
        PointField("rgb", 12, UINT32, 1),
    ]
    m2 = build_cloud(f_xyz_rgb, 16, pack_xyz_rgb)
    check_parse_like_plane_test(m2, "none")

    # XYZ + reflectivity only
    f_xyz_r = [
        PointField("x", 0, FLOAT32, 1),
        PointField("y", 4, FLOAT32, 1),
        PointField("z", 8, FLOAT32, 1),
        PointField("reflectivity", 12, FLOAT32, 1),
    ]
    m3 = build_cloud(f_xyz_r, 16, pack_xyz_intensity)
    check_parse_like_plane_test(m3, "reflectivity")

    print("smoke_pointcloud2_layouts: all checks passed")


if __name__ == "__main__":
    main()
