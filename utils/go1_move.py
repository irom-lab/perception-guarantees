from utils.ros_utils import *
import time
from utils.zed_utils import *
import sys

sys.path.append('../lib/python/arm64')
import robot_interface as sdk


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
