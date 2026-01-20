#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import PointCloud2  # 这里换成 PointCloud2 类型

def callback(msg):
    # 去掉 frame_id 的开头斜杠，防止 TF 识别异常
    msg.header.frame_id = msg.header.frame_id.lstrip('/')
    pub.publish(msg)

if __name__ == "__main__":
    rospy.init_node('pc2_frame_fix')
    
    # 发布修正后的点云数据话题
    pub = rospy.Publisher('/camera/rgb/points_f', PointCloud2, queue_size=10)
    
    # 订阅原始点云数据话题
    rospy.Subscriber('/camera/rgb/points', PointCloud2, callback)
    
    rospy.spin()
