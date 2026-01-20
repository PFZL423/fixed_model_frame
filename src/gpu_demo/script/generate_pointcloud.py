#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script reads a bag file containing RGBD data, adds the corresponding
PointCloud2 messages, and saves it again into a bag file. Optional arguments
allow to select only a portion of the original bag file.
"""

import argparse
import sys
import os

if __name__ == '__main__':
    
    # parse command line
    parser = argparse.ArgumentParser(description='''
    This script reads a bag file containing RGBD data, 
    adds the corresponding PointCloud2 messages, and saves it again into a bag file. 
    Optional arguments allow to select only a portion of the original bag file.  
    ''')
    parser.add_argument('--start', help='skip the first N seconds of input bag file (default: 0.0)', default=0.0, type=float)
    parser.add_argument('--duration', help='only process N seconds of input bag file (default: off)', type=float)
    parser.add_argument('--nth', help='only process every N-th frame of input bag file (default: 15)', default=15, type=int)
    parser.add_argument('--skip', help='skip N blocks in the beginning (default: 1)', default=1, type=int)
    parser.add_argument('--compress', help='compress output bag file', action='store_true')
    parser.add_argument('inputbag', help='input bag file')
    parser.add_argument('outputbag', nargs='?', help='output bag file')
    args = parser.parse_args()

    import rospy
    import rosbag
    import sensor_msgs.msg
    from cv_bridge import CvBridge, CvBridgeError
    import struct
    import tf2_msgs.msg
    import numpy as np
    
    # 验证输入bag文件
    if not os.path.exists(args.inputbag):
        print(f"错误: 输入bag文件不存在: {args.inputbag}")
        print("请检查文件路径是否正确")
        sys.exit(1)
    
    if not os.path.isfile(args.inputbag):
        print(f"错误: 路径不是文件: {args.inputbag}")
        sys.exit(1)
    
    if not args.inputbag.endswith('.bag'):
        print(f"警告: 输入文件可能不是bag文件: {args.inputbag}")
    
    if not args.outputbag:
        args.outputbag = os.path.splitext(args.inputbag)[0] + "-points.bag"
      
    print("Processing bag file:")
    print("  in:", args.inputbag)
    print("  out:", args.outputbag)
    print("  starting from: %s seconds" % args.start)
        
    if args.duration:
        print("  duration: %s seconds" % args.duration)
        
    print("  saving every %s-th frame" % args.nth)
    print("  skipping %s blocks" % args.skip)

    inbag = rosbag.Bag(args.inputbag, 'r')
    if args.compress:
        param_compression = rosbag.bag.Compression.BZ2
    else:
        param_compression = rosbag.bag.Compression.NONE
        
    outbag = rosbag.Bag(args.outputbag, 'w', compression=param_compression)
    
    depth_camera_info = None
    rgb_camera_info = None
    depth_image = None
    rgb_image_color = None

    nan = float('nan')
    bridge = CvBridge()
    frame = 0 
    transforms = {}
    
    time_start = None
    total_frames_processed = 0
    
    # 获取bag文件总时长信息
    bag_info = inbag.get_type_and_topic_info()
    print(f"开始处理bag文件...")
    
    try:
        for topic, msg, t in inbag.read_messages():
            if time_start is None:
                time_start = t
            if t - time_start < rospy.Duration.from_sec(args.start):
                continue
            if args.duration and (t - time_start > rospy.Duration.from_sec(args.start + args.duration)):
                break
            
            current_time = (t - time_start).to_sec()
            print(f"处理时间: {current_time:.2f}s, 已处理帧数: {total_frames_processed}\r", end='', flush=True)
            
            if topic == "/tf":
                for transform in msg.transforms:
                    transforms[(transform.header.frame_id, transform.child_frame_id)] = transform
                continue
                
            if topic == "/camera/depth/camera_info":
                depth_camera_info = msg
                continue
                
            if topic == "/camera/rgb/camera_info":
                rgb_camera_info = msg
                continue
                
            if topic == "/camera/rgb/image_color" and rgb_camera_info:
                rgb_image_color = msg
                continue
                
            if topic == "/camera/depth/image" and depth_camera_info and rgb_image_color and rgb_camera_info:
                depth_image = msg
                
                # Check time sync between depth and color images
                if abs((depth_image.header.stamp - rgb_image_color.header.stamp).to_sec()) > 1/30.0:
                    continue
                
                frame += 1
                if frame % args.nth == 0:
                    if args.skip > 0:
                        args.skip -= 1
                    else:
                        total_frames_processed += 1
                        print(f"\n正在生成第 {total_frames_processed} 帧的点云数据...")
                        
                        # Store original messages
                        if transforms:
                            tf_msg = tf2_msgs.msg.TFMessage()
                            tf_msg.transforms = list(transforms.values())
                            outbag.write("/tf", tf_msg, t)
                            transforms = {}
                            
                        outbag.write("/camera/depth/camera_info", depth_camera_info, t)
                        outbag.write("/camera/depth/image", depth_image, t)
                        outbag.write("/camera/rgb/camera_info", rgb_camera_info, t)
                        outbag.write("/camera/rgb/image_color", rgb_image_color, t)

                        # Generate monochrome image from color image
                        try:
                            cv_rgb_image_color = bridge.imgmsg_to_cv2(rgb_image_color, "bgr8")
                            cv_rgb_image_mono = bridge.imgmsg_to_cv2(rgb_image_color, "mono8")
                            rgb_image_mono = bridge.cv2_to_imgmsg(cv_rgb_image_mono, "mono8")
                            rgb_image_mono.header = rgb_image_color.header
                            outbag.write("/camera/rgb/image_mono", rgb_image_mono, t)
                        except CvBridgeError as e:
                            print(f"CV Bridge Error: {e}")
                            continue

                        # Generate depth and colored point cloud
                        try:
                            cv_depth_image = bridge.imgmsg_to_cv2(depth_image, "passthrough")
                            
                            centerX = depth_camera_info.K[2]
                            centerY = depth_camera_info.K[5]
                            depthFocalLength = depth_camera_info.K[0]
                            
                            # Generate depth-only point cloud
                            depth_points = sensor_msgs.msg.PointCloud2()
                            depth_points.header = depth_image.header
                            depth_points.width = depth_image.width
                            depth_points.height = depth_image.height
                            depth_points.fields.append(sensor_msgs.msg.PointField(
                                name="x", offset=0, datatype=sensor_msgs.msg.PointField.FLOAT32, count=1))
                            depth_points.fields.append(sensor_msgs.msg.PointField(
                                name="y", offset=4, datatype=sensor_msgs.msg.PointField.FLOAT32, count=1))
                            depth_points.fields.append(sensor_msgs.msg.PointField(
                                name="z", offset=8, datatype=sensor_msgs.msg.PointField.FLOAT32, count=1))
                            depth_points.point_step = 12
                            depth_points.row_step = depth_points.point_step * depth_points.width
                            
                            buffer = []
                            for v in range(depth_image.height):
                                for u in range(depth_image.width):
                                    d = cv_depth_image[v, u]
                                    # 使用numpy的isfinite检查，避免警告
                                    if np.isfinite(d) and d > 0:
                                        ptx = (u - centerX) * d / depthFocalLength
                                        pty = (v - centerY) * d / depthFocalLength
                                        ptz = d
                                        buffer.append(struct.pack('fff', ptx, pty, ptz))
                                    else:
                                        buffer.append(struct.pack('fff', nan, nan, nan))
                            depth_points.data = b"".join(buffer)
                            outbag.write("/camera/depth/points", depth_points, t)
                            
                            # Generate RGB point cloud
                            rgb_points = sensor_msgs.msg.PointCloud2()
                            rgb_points.header = rgb_image_color.header
                            rgb_points.width = depth_image.width
                            rgb_points.height = depth_image.height
                            rgb_points.fields.append(sensor_msgs.msg.PointField(
                                name="x", offset=0, datatype=sensor_msgs.msg.PointField.FLOAT32, count=1))
                            rgb_points.fields.append(sensor_msgs.msg.PointField(
                                name="y", offset=4, datatype=sensor_msgs.msg.PointField.FLOAT32, count=1))
                            rgb_points.fields.append(sensor_msgs.msg.PointField(
                                name="z", offset=8, datatype=sensor_msgs.msg.PointField.FLOAT32, count=1))
                            rgb_points.fields.append(sensor_msgs.msg.PointField(
                                name="rgb", offset=12, datatype=sensor_msgs.msg.PointField.UINT32, count=1))
                            rgb_points.point_step = 16
                            rgb_points.row_step = rgb_points.point_step * rgb_points.width
                            
                            buffer = []
                            for v in range(depth_image.height):
                                for u in range(depth_image.width):
                                    d = cv_depth_image[v, u]
                                    # 使用numpy的isfinite检查，避免警告
                                    if np.isfinite(d) and d > 0:
                                        rgb = cv_rgb_image_color[v, u]
                                        ptx = (u - centerX) * d / depthFocalLength
                                        pty = (v - centerY) * d / depthFocalLength
                                        ptz = d
                                        
                                        # Pack RGB as uint32 (0x00RRGGBB format)
                                        rgb_packed = (int(rgb[2]) << 16) | (int(rgb[1]) << 8) | int(rgb[0])
                                        buffer.append(struct.pack('fffI', ptx, pty, ptz, rgb_packed))
                                    else:
                                        buffer.append(struct.pack('fffI', nan, nan, nan, 0))
                            rgb_points.data = b"".join(buffer)
                            outbag.write("/camera/rgb/points", rgb_points, t)
                            
                        except CvBridgeError as e:
                            print(f"\nCV Bridge Error: {e}")
                            continue
                        except Exception as e:
                            print(f"\nError processing point cloud: {e}")
                            continue
                
                # Reset for next frame
                depth_image = None
                rgb_image_color = None
                continue
                
            # Pass through other topics unchanged
            if topic not in ["/tf", "/camera/depth/camera_info", "/camera/rgb/camera_info",
                           "/camera/rgb/image_color", "/camera/depth/image"]:
                outbag.write(topic, msg, t)
                
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        print(f"\nClosing bags...")
        inbag.close()
        outbag.close()
        print("Done!")
