import rospy
from geometry_msgs.msg import TransformStamped
from scipy.spatial.transform import Rotation
import numpy as np
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
import ros_numpy


class ZedROS:
    def __init__(self):
        # init vars
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.quat = [0.0, 0.0, 0.0, 0.0]
        self.yaw = 0.0

        self.odom_sub = rospy.Subscriber("/zed2i/zed_node/odom", Odometry, self.odom_callback)  # camera position subscriber 

    def odom_callback(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        self.z = msg.pose.pose.position.z

        self.vx = 0 # update
        self.vy = 0

        self.quat = [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]
        self.yaw = self.get_yaw_from_quat(self.quat)

    def get_yaw_from_quat(self, quat):
        rot = Rotation.from_quat(quat)
        yaw = rot.as_euler('xyz', degrees=True)[2]

        return yaw
    

class ViconStateListener:
    def __init__(self, sub_state, forward):
        self.x = 0.0
        self.y = 0.0
        self.forward = forward
        self.sub = rospy.Subscriber(sub_state, TransformStamped, self.callback)

    def callback(self, data):
        if self.forward == 'x':
            # if go1 forward is x
            self.x = -data.transform.translation.y
            self.y = data.transform.translation.x
        elif self.forward == 'y':
            # if go1 forward is y
            self.x = -data.transform.translation.x
            self.y = -data.transform.translation.y