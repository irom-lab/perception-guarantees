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
        self.yaw = 0.0
        self.timestamp = 0.0
        self.forward = forward
        self.sub = rospy.Subscriber(sub_state, TransformStamped, self.callback)

    def callback(self, data):
        if self.forward == 'x':
            # if go1 forward is x
            self.x = -data.transform.translation.y
            self.y = data.transform.translation.x
            quat = [data.transform.rotation.x, data.transform.rotation.y, data.transform.rotation.z, data.transform.rotation.w]
            # TODO: add in correct transformation (not using this orientation at the moment)
            self.yaw = self.get_yaw_from_quat(quat)
            quat = [data.transform.rotation.x, data.transform.rotation.y, data.transform.rotation.z, data.transform.rotation.w]
            self.timestamp = rospy.Time(data.header.stamp.secs, data.header.stamp.nsecs).to_sec()
        elif self.forward == 'y':
            # if go1 forward is y
            self.x = -data.transform.translation.x
            self.y = -data.transform.translation.y
            quat = [data.transform.rotation.x, data.transform.rotation.y, data.transform.rotation.z, data.transform.rotation.w]
            self.yaw = self.get_yaw_from_quat(quat)
            self.timestamp = rospy.Time(data.header.stamp.secs, data.header.stamp.nsecs).to_sec()

    def get_yaw_from_quat(self, quat):
        rot = Rotation.from_quat(quat)
        yaw = rot.as_euler('xyz', degrees=False)[2]

        return yaw


class GroundTruthBB():
    def __init__(self, chair_nums=[]):
            self.chair_numbers = chair_nums
            self.chair_states = {}

            if len(self.chair_numbers) > 0:
                for num in self.chair_numbers:
                    topic_name = "vicon/chair_" + str(num) + "/chair_" + str(num)
                    self.chair_states[num] = ViconStateListener(topic_name, "x")


    def get_true_bb(self):
        
        bb_list = []
        bb_yaws = []
        for num in self.chair_numbers:
            x, y, yaw = self.chair_states[num].x, self.chair_states[num].y, self.chair_states[num].yaw
            if (np.abs(x) < 0.001) and (np.abs(y) < 0.001):
                continue

            if num == 1:
                padding = 0.35 # axis aligned padding val
                padding_x = 0.25
                padding_y = 0.25
                
            elif num == 2:
                padding = 0.48
                padding_x = 0.3
                padding_y = 0.34
            
            elif num == 3 or num == 6 or num == 7:
                padding = 0.43
                padding_x = 0.3
                padding_y = 0.3

            elif num == 4 or num == 5:
                padding = 0.59
                padding_x = 0.41
                padding_y = 0.42
            
            elif num == 8:
                padding = 0.29
                padding_x = 0.2
                padding_y = 0.18

            elif num == 9:
                padding = 0.38
                padding_x = 0.28
                padding_y = 0.22

            elif num == 10:
                padding = 0.41
                padding_x = 0.28
                padding_y = 0.29

            bb = [[x - padding_x, y - padding_y], [x + padding_x, y + padding_x]] 
            bb_list.append(bb)
            bb_yaws.append(yaw)

        return np.array(bb_list), np.array(bb_yaws)

        # # c1x, c1y = self.chair1_state.x, self.chair1_state.y
        # c2x, c2y = self.chair2_state.x, self.chair2_state.y
        # # bb1 = [[c1x-0.5, c1y-0.5], [c1x+0.5, c1y+0.5]]
        # bb2 = [[c2x-0.5, c2y-0.5], [c2x+0.5, c2y+0.5]]
        # bb = []
        # # bb.append(bb1)
        # bb.append(bb2)
        # return np.array(bb)
    

    def project_bb(self):
        # TODO: use markers to get bb in x-y plane
        pass
