from planning.Safe_Planner import *
import rospy
from geometry_msgs.msg import TransformStamped
from scipy.spatial.transform import Rotation
import numpy as np
import sys
from utils.ros_utils import *
import time
from utils.zed_utils import *


class Go1_move():
    def __init__(self, sp, debug=False):
        self.sp = sp
        self.done = False
        self.state = [0.0, 0.0, 0.0, 0.0]
        self.timestamp = 0
        if debug:
            self.vicon_state = ViconStateListener("vicon/cobs_alec/cobs_alec", "y") # TODO: change to Strelka
            # wait for initial state read
            if (self.vicon_state.timestamp == 0.0):
                time.sleep(0.5)
            self.state, self.timestamp = self.get_true_state()
        else:
            # initialize Zed 
            self.camera = Zed()
            # wait for initial state
            if (self.camera.get_pose()[0] == None):
                time.sleep(0.5)
            self.state, self.timestamp = self.get_state()
    
    def get_state(self):
        # TODO: analyze on moving robot closely to double check coordinate system
        # self.camera.get_IMU()
        pos_state, timestamp = self.camera.get_pose()
        vx, vy = self.calc_velocity(self.state[0], self.state[2], self.timestamp, pos_state[0], pos_state[1], timestamp)
        state = [pos_state[0], vx, pos_state[1], vy]

        # update state and timestamp
        self.state = state
        self.timestamp = timestamp
        return state, timestamp

    def get_true_state(self):
        x, y, timestamp = self.vicon_state.x, self.vicon_state.y, self.vicon_state.timestamp
        vx, vy = self.calc_velocity(self.state[0], self.state[2], self.timestamp, x, y, timestamp, units='seconds')
        state = [x, vx, y, vy]
        # update state and timestamp
        self.state = state
        self.timestamp = timestamp
        return state, timestamp
    
    def calc_velocity(self, x1, y1, t1, x2, y2, t2, units='microseconds'):
        if units == 'microseconds':
            delta_t_microseconds = t2 - t1
            delta_t = delta_t_microseconds / 1e6  # convert to seconds
        else:
            delta_t = t2 - t1
        vx = (x2 - x1) / delta_t
        vy = (y2 - y1) / delta_t
        return vx, vy

    def move(self, action):
        # replace with actual moving and 
        # getting actual state
        x, y, vx, vy = self.state
        ux, uy = action
        x_new = x + vx* self.sp.dt
        y_new = y + vy * self.sp.dt
        vx_new = vx-k1*self.sp.dt*vx+k1*ux*self.sp.dt
        vy_new = vy-k2*self.sp.dt*vy+k2*uy*self.sp.dt
        self.state = np.array([x_new, y_new, vx_new, vy_new]) 

        if np.linalg.norm(self.state[0:2]-self.sp.goal[0:2]) < 0.5: # this is arbitrary now
            self.done = True

def state_to_planner(state, sp):
    # convert robot state to planner coordinates
    return np.array([[[0,-1,0,0],[1,0,0,0],[0,0,0,-1],[0,0,1,0]]])@np.array(state) + np.array([sp.world.w/2,0,0,0])

def state_to_go1(state, sp):
    x, y, vx, vy = state
    return np.array([y, -x+sp.world.w/2, vy, -vx])

def boxes_to_planner(boxes, sp):
    boxes_new = np.zeros_like(boxes)
    for i in range(len(boxes)):
        boxes_new[i,:,:] = np.reshape(np.array([[[0,0,0,-1],[1,0,0,0],[0,-1,0,0],[0,0,1,0]]])@np.reshape(boxes[0],(4,1)),(2,2)) + np.array([sp.world.w/2,0])
    return boxes_new

def get_boxes(sp):
    # fake random boxes in planner coordinates
    # replace with camera + 3detr later
    n = np.random.randint(1,5)
    boxes = []
    for i in range(n):
        x0 = np.random.uniform(0,sp.world.w)
        y0 = np.random.uniform(2,sp.world.h)
        x1 = np.random.uniform(0,sp.world.w)
        y1 = np.random.uniform(2,sp.world.h)
        boxes.append(np.array([[min(x0,x1),min(y0,y1)],
                         [max(x0,x1),max(y0,y1)]]))
    return np.array(boxes)


def plan_loop():
    debug = False
    if debug:
        rospy.init_node('listener', anonymous=True)
    
    # planner
    # load pre-computed: need to recompute for actual gains
    f = open('planning/pre_compute/reachable_cost5_newdim.pkl', 'rb')
    reachable = pickle.load(f)
    f = open('planning/pre_compute/Pset_cost5_newdim.pkl', 'rb')
    Pset = pickle.load(f)

    # initialize planner
    sp = Safe_Planner(goal_f=[7.0, -2.0, 0.0, 0.0])
    print(sp.goal)
    sp.load_reachable(Pset, reachable)

    go1 = Go1_move(sp, debug=debug)
    print(go1.state)
    time.sleep(2)
    for t in range(100):
        print(go1.get_state())

    # t = 0
    # cp = 0.59
    # while True:
    #     # perception + cp
    #     # boxes = get_boxes(sp)
    #     boxes = np.array([[[0,0],[0.01,0.01]]])
    #     boxes[:,0,:] -= cp
    #     boxes[:,1,:] += cp
        
    #     # plan
    #     state = state_to_planner(go1.state, sp)
    #     start_idx = np.argmin(cdist(np.array(sp.Pset),state))

    #     # print(start_idx,Pset[start_idx],state)
    #     res = sp.plan(state, boxes)

    #     #fig, ax = sp.world.show()
    #     # plt.show()

    #     # execute
    #     if len(res[0]) > 1:
    #         print(res[0])
    #         fig, ax = sp.show_connection(res[0])
    #         plt.show()
    #         policy_before_trans = np.vstack(res[2])
    #         policy = (np.array([[0,1],[-1,0]])@policy_before_trans.T).T
    #         for step in range(int(sp.sensor_dt/sp.dt)):
    #             action = policy[step]
    #             go1.move(action)
    #             t += sp.sensor_dt
    #             print(go1.state)
    #         if go1.done:
    #             break
    #     else:
    #         for step in range(int(sp.sensor_dt/sp.dt)):
    #             action = [0,0]
    #             go1.move(action)
    #             t += sp.sensor_dt
    #     if t >100:
    #         break

    if debug:    
        rospy.spin()

if __name__ == '__main__':
    plan_loop()