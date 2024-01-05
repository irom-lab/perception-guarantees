from utils.ros_utils import *
import time
from utils.zed_utils import *
import sys

sys.path.append('../lib/python/arm64')
import robot_interface as sdk


class Go1_move():
    def __init__(self, sp, vicon=False, state_type='zed'):
        self.sp = sp
        self.done = False # TODO: Set stopping condition
        self.state = [0.0, 0.0, 0.0, 0.0]
        self.true_state = [0.0, 0.0, 0.0, 0.0]
        self.yaw = 0.0
        self.ts_yaw = 0.0
        self.timestamp = 0
        self.ts_timestamp = 0
        self.state_type = state_type

        # initialize states
        if vicon:
            # self.vicon_state = ViconStateListener("vicon/strelka/strelka", "x") 
            self.vicon_state = ViconStateListener("vicon/cobs_alec/cobs_alec", "x") 

            # wait for initial state read
            if (self.vicon_state.timestamp == 0.0):
                time.sleep(0.6)
            self.true_state, self.ts_yaw, self.ts_timestamp = self.get_true_state()
            if self.state_type == 'vicon':
                self.state = self.true_state
                self.timestamp = self.ts_timestamp
                self.yaw = self.ts_yaw

        if self.state_type == 'zed':
            # initialize Zed 
            self.camera = Zed()
            # wait for initial state
            if (self.camera.get_pose()[0] == None):
                time.sleep(0.5)
            self.state, self.timestamp = self.get_state()

        # initialize go1 communication
        HIGHLEVEL = 0xee
        LOWLEVEL  = 0xff

        self.udp = sdk.UDP(HIGHLEVEL, 8080, "192.168.123.161", 8082)

        self.cmd = sdk.HighCmd()
        self.udp.InitCmdData(self.cmd)
    
    def get_state(self):
        # TODO: analyze on moving robot closely to double check coordinate system
        # self.camera.get_IMU()

        if self.state_type == 'zed':
            pos_state, timestamp, self.yaw = self.camera.get_pose()
            if timestamp != self.timestamp:
                vx, vy = self.calc_velocity(self.state[0], self.state[1], self.timestamp, pos_state[0], pos_state[1], timestamp)
                state = [pos_state[0], pos_state[1], vx, vy]

                print("zed YAW", self.yaw)

                # update state and timestamp
                self.state = state
                self.timestamp = timestamp
        
        elif self.state_type == 'vicon':
            self.state, self.yaw, timestamp = self.get_true_state()
            self.timestamp = timestamp

        return self.state, timestamp

    def get_true_state(self):
        x, y, timestamp = self.vicon_state.x, self.vicon_state.y, self.vicon_state.timestamp
        if timestamp != self.ts_timestamp:
            vx, vy = self.calc_velocity(self.true_state[0], self.true_state[1], self.ts_timestamp, x, y, timestamp, units='seconds')
            state = [x, y, vx, vy]
            self.ts_yaw = self.vicon_state.yaw
            # update state and timestamp
            self.true_state = state
            self.ts_timestamp = timestamp

            print("vicon YAW", self.ts_yaw)
        return self.true_state, self.ts_yaw, timestamp
    
    def calc_velocity(self, x1, y1, t1, x2, y2, t2, units='microseconds'):
        if units == 'microseconds':
            delta_t_microseconds = t2 - t1
            delta_t = delta_t_microseconds / 1e6  # convert to seconds
        else:
            delta_t = t2 - t1
            print(t2, t1)
        vx = (x2 - x1) / delta_t
        vy = (y2 - y1) / delta_t
        return vx, vy

    def correct_yaw(self):
        # TODO: add in yaw correction
        return -self.yaw/1.0
        
    def move(self, action):
        ux, uy = action
        # check ux, uy fall between -1, 1
        bound = 0.5 # TODO: change back to 1
        ux = max(-bound, min(ux, bound))
        uy = max(-bound, min(uy, bound))
        yaw_cmd = self.correct_yaw() 

        self.cmd.mode = 2
        self.cmd.gaitType = 1
        self.cmd.velocity = [ux, uy] # -1  ~ +1
        self.cmd.yawSpeed = yaw_cmd
        self.cmd.bodyHeight = 0.0

        self.udp.SetSend(self.cmd)
        self.udp.Send()



        


