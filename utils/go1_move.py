from utils.ros_utils import *
import time
from utils.zed_utils import *
import sys

sys.path.append('../lib/python/arm64')
import robot_interface as sdk


class Go1_move():
    def __init__(self, goal, vicon=False, state_type='zed', save_folder=None):
        # self.sp = sp
        self.state = [0.0, 0.0, 0.0, 0.0]
        self.true_state = [0.0, 0.0, 0.0, 0.0]
        self.yaw = 0.0
        self.ts_yaw = 0.0
        self.timestamp = 0
        self.ts_timestamp = 0
        self.state_type = state_type
        self.goal = goal

        # initialize states
        if vicon:
            self.vicon_state = ViconStateListener("vicon/strelka/strelka", "x") 
            self.chair1_state = ViconStateListener("vicon/chair_1/chair_1", "x") 
            self.chair2_state = ViconStateListener("vicon/chair_2/chair_2", "x") 
            # self.vicon_state = ViconStateListener("vicon/cobs_alec/cobs_alec", "x") 

            # wait for initial state read
            if (self.vicon_state.timestamp == 0.0):
                time.sleep(0.6)
            self.true_state, self.ts_yaw, self.ts_timestamp = self.get_true_state()
            if self.state_type == 'vicon':
                self.state = self.true_state
                self.timestamp = self.ts_timestamp
                self.yaw = self.ts_yaw
                self.camera = Zed(state_ic=self.true_state, yaw_ic=self.ts_yaw, save_folder=save_folder)
            # set zed camera for bounding boxes
            # initialize Zed 
           

        if self.state_type == 'zed':
            # initialize Zed 
            self.camera = Zed(state_ic=self.true_state, yaw_ic=self.ts_yaw, save_folder=save_folder)
            time.sleep(1)
            # wait for initial state
            if (self.camera.get_pose()[0] == None):
                time.sleep(0.5)
            self.state, self.timestamp, self.yaw= self.get_state()

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
                # print("DEBUG: ", self.state, self.timestamp, pos_state, timestamp)
                vx, vy = self.calc_velocity(self.state[0], self.state[1], self.timestamp, pos_state[0], pos_state[1], timestamp)
                state = [pos_state[0], pos_state[1], vx, vy]

                # print("zed YAW", self.yaw)

                # update state and timestamp
                self.state = state
                self.timestamp = timestamp
        
        elif self.state_type == 'vicon':
            self.state, self.yaw, timestamp = self.get_true_state()
            self.timestamp = timestamp
        
        return self.state, timestamp, self.yaw

    def get_true_state(self):
        x, y, timestamp = self.vicon_state.x, self.vicon_state.y, self.vicon_state.timestamp
        if timestamp != self.ts_timestamp:
            vx, vy = self.calc_velocity(self.true_state[0], self.true_state[1], self.ts_timestamp, x, y, timestamp, units='microseconds')
            state = [x, y, vx, vy]
            self.ts_yaw = self.vicon_state.yaw
            # update state and timestamp
            self.true_state = state
            self.ts_timestamp = timestamp
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

    def check_goal(self):
        self.get_state()
        if (np.abs(self.state[0] - self.goal[0]) < 1.5) and (np.abs(self.state[1] - self.goal[1]) < 1.5):
            print("AT GOAL! :)")
            self.stop()

            return True
    
        else:
            return False

    def correct_yaw(self):
        return -self.yaw/2
        
    def move(self, action):
        ux, uy = action
        # check ux, uy fall between -1, 1
        bound = 1.5 # TODO: change back to 1
        ux = max(-bound, min(ux, bound))
        uy = max(-bound, min(uy, bound))

        if np.abs(ux) < 0.2 and np.abs(ux) > 0.1:
            ux = ux / np.abs(ux) * 0.15
        if np.abs(uy) < 0.2 and np.abs(uy) > 0.1:
            uy = uy / np.abs(uy) * 0.15
        
        yaw_cmd = self.correct_yaw() 
        if np.abs(yaw_cmd) > 0.15:
            print("YAW Correction: (yaw, yaw_cmd)", self.yaw, yaw_cmd)

        self.cmd.mode = 2
        self.cmd.gaitType = 1
        self.cmd.velocity = [ux, uy] # -1  ~ +1
        self.cmd.yawSpeed = yaw_cmd
        self.cmd.bodyHeight = 0.0

        self.udp.SetSend(self.cmd)
        self.udp.Send()

    def stop(self):
        self.cmd.mode = 0
        self.cmd.gaitType = 1
        self.cmd.velocity = [0, 0] # -1  ~ +1
        self.cmd.yawSpeed = 0
        self.cmd.bodyHeight = 0.0

        self.udp.SetSend(self.cmd)
        self.udp.Send()