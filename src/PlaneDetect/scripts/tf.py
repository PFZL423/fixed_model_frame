#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将点云从消息 header 中的坐标系变换到目标坐标系（默认 map）并发布。
依赖: rospy, tf2_ros, tf2_sensor_msgs, sensor_msgs, geometry_msgs

运行（需已 source devel/setup.bash）:
  rosrun PlaneDetect tf.py
注意:
  - TF 树中必须存在 target_frame；若报 map 不存在，请发布 static_transform。
  - 点云 header 中带前导 '/' 的 frame_id 会自动去掉以符合 tf2 要求。
  - 播 bag 时点云时间常为录制时刻，若用「当前时刻发布的静态 TF」，请设
    _use_point_cloud_stamp:=false，lookup 会用「最新 TF」。
  - 更正规做法: rosparam set /use_sim_time true 且 rosbag play --clock your.bag，
    使 TF 与点云时间一致（此时可保持 use_point_cloud_stamp 默认 true）。

可选参数（私有命名空间）:
  _input_topic:=/camera/rgb/points
  _output_topic:=/camera/points_in_map
  _target_frame:=map
  _queue_size:=10
  _use_point_cloud_stamp:=true   # false=按最新 TF 查（适合 bag+静态 map）
"""
from __future__ import print_function

import rospy
import tf2_ros
from tf2_sensor_msgs import do_transform_cloud
from sensor_msgs.msg import PointCloud2


def tf2_frame_id(frame_id):
    """
    tf2 要求 frame_id 不能以 '/' 开头；部分 bag/驱动仍使用 /camera_link 等形式。
    """
    if not frame_id:
        return frame_id
    return frame_id.lstrip("/")


def main():
    rospy.init_node("pcd_to_map_transformer", anonymous=False)

    input_topic = rospy.get_param("~input_topic", "/camera/rgb/points")
    output_topic = rospy.get_param("~output_topic", "/camera/points_in_map")
    target_frame = tf2_frame_id(rospy.get_param("~target_frame", "map"))
    queue_size = int(rospy.get_param("~queue_size", 10))
    lookup_timeout = rospy.Duration(rospy.get_param("~lookup_timeout", 1.0))
    use_point_cloud_stamp = rospy.get_param("~use_point_cloud_stamp", True)

    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)

    pub = rospy.Publisher(output_topic, PointCloud2, queue_size=queue_size, latch=False)

    def callback(msg):
        if not msg.header.frame_id:
            rospy.logwarn_throttle(5.0, "点云 header.frame_id 为空，跳过")
            return
        source_frame = tf2_frame_id(msg.header.frame_id)
        # Time(0) = 使用缓冲区中「最新可用」变换，避免 bag 旧 stamp 与当前发布的静态 TF 对不齐
        lookup_stamp = msg.header.stamp if use_point_cloud_stamp else rospy.Time(0)
        try:
            trans = tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                lookup_stamp,
                lookup_timeout,
            )
            out = do_transform_cloud(msg, trans)
            pub.publish(out)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn_throttle(2.0, "TF 变换失败 (%s -> %s): %s", source_frame, target_frame, e)

    rospy.Subscriber(input_topic, PointCloud2, callback, queue_size=queue_size)
    rospy.loginfo(
        "pcd_to_map_transformer: %s -> %s (target_frame=%s, use_point_cloud_stamp=%s)",
        input_topic,
        output_topic,
        target_frame,
        use_point_cloud_stamp,
    )
    rospy.spin()


if __name__ == "__main__":
    main()
