from planning.Safe_Planner import *
import rospy
from geometry_msgs.msg import TransformStamped
from scipy.spatial.transform import Rotation
import numpy as np
from utils.go1_move import *
from utils.plotting import *
import time
import pickle
import copyreg
import copy
import threading

class Robot_Plan:
    def __init__(self):
        # ****************************************************
        # SET DESIRED FLAGS
        self.vicon = False # set if want vicon state active; If true, make sure ./vicon.sh is running from pg_ws (and zed box connected to gas dynamics network)
        self.state_type = 'zed' #'zed' #if want to set the state used as the vicon state, 'zed' 
        self.replan = True # set if want to just follow open loop plan
        self.save_traj = False  # set if want to save trajectory and compare against plan
        self.plot_traj = True  # set if want to visualize trajectory at the end of execution
        self.goal_forrestal = [7.0, 0.0, 0.0, 0.5] # goal in forrestal coordinates
        self.reachable_file = 'planning/pre_compute/reachable_1_10.pkl'
        self.pset_file = 'planning/pre_compute/Pset_1_10.pkl'
        self.num_samples = 1000  # number of samples used for the precomputed files
        self.dt = 0.1 #   planner dt
        self.radius = 0.7 # distance between intermediate goals on the frontier
        self.chairs = [1, 2, 3, 4,5,6, 7, 8, 9, 10, 11, 12]  # list of chair labels to be used to get ground truth bounding boxes
        self.num_detect = 15  # number of boxes for 3DETR to detect
        self.robot_radius = 0.14
        self.cp = 0.02 # 0.73 # 0.61 #0.02
        self.sensor_dt = 0.8 # time in seconds to replan
        self.num_times_detect = 1
        self.max_search_iter = 2000
        #result_dir = 'results/supplementary_middle_goal_dt_08_cp002/' # set to unique trial identifier if saving results
        self.result_dir = None
        self.is_finetune = False
        # ****************************************************

        f = open(self.reachable_file, 'rb')
        self.reachable = pickle.load(f)
        f = open(self.pset_file, 'rb')
        self.Pset = pickle.load(f)

        sp = Safe_Planner(goal_f=self.goal_forrestal, sensor_dt=self.sensor_dt, dt=self.dt, n_samples=self.num_samples, radius=self.radius, max_search_iter=self.max_search_iter, speed=1)
        print("goal (planner coords): ", sp.goal)
        self.go1 = Go1_move(sp.goal_f, vicon=self.vicon, state_type=self.state_type, save_folder=self.result_dir)

        self.current_plan = None
        self.next_plan = None
        self.lock = threading.Lock()
        self.running = True
        self.boxes = []
        np.random.seed(0)

    def state_to_planner(self, state):
        # convert robot state to planner coordinates
        return np.array([[[0,-1,0,0],[1,0,0,0],[0,0,0,-1],[0,0,1,0]]])@np.array(state) + np.array([self.sp.world.w/2,0,0,0])

    def state_to_go1(self, state):
        x, y, vx, vy = state
        return np.array([y, -x+self.sp.world.w/2, vy, -vx])

    def boxes_to_planner(self, boxes):
        boxes_new = np.zeros_like(boxes)
        for i in range(len(boxes)):
            boxes_new[i,:,:] = np.reshape(np.array([[[0,0,0,-1],[1,0,0,0],[0,-1,0,0],[0,0,1,0]]])@np.reshape(boxes[0],(4,1)),(2,2)) + np.array([self.sp.world.w/2,0])
        return boxes_new

    def boxes_to_planner_frame(self, boxes):
        boxes_new = np.zeros_like(boxes)
        for i in range(len(boxes)):
            #boxes_new[i,:,:] = np.reshape(np.array([[[0,0,0,-1],[1,0,0,0],[0,-1,0,0],[0,0,1,0]]])@np.reshape(boxes[0],(4,1)),(2,2)) + np.array([sp.world.w/2,0])
            boxes_new[i,0,0] = -boxes[i,1,1] + self.sp.world.w/2
            boxes_new[i,1,0] = -boxes[i,0,1] + self.sp.world.w/2
            boxes_new[i,:,1] =  boxes[i,:,0]
        return boxes_new

    # def get_boxes(sp):
    #     # fake random boxes in planner coordinates
    #     # replace with camera + 3detr later
    #     n = np.random.randint(1,5)
    #     boxes = []
    #     for i in range(n):
    #         x0 = np.random.uniform(0,sp.world.w)
    #         y0 = np.random.uniform(2,sp.world.h)
    #         x1 = np.random.uniform(0,sp.world.w)
    #         y1 = np.random.uniform(2,sp.world.h)
    #         boxes.append(np.array([[min(x0,x1),min(y0,y1)],
    #                          [max(x0,x1),max(y0,y1)]]))
    #     return np.array(boxes)

    def plan(self):
        while self.running:
            # Perception + planning logic
            state = self.state_to_planner(self.go1.state)
            t_1 = time.time()
            boxes = self.go1.camera.get_boxes(self.cp, self.num_detect, self.is_finetune)
            t_2 = time.time()
            print(f"Inference time: {t_2 - t_1}")
            boxes = boxes[:, :, 0:2]
            boxes = self.boxes_to_planner_frame(boxes)
            t_3 = time.time()
            print(f"Box transformation time: {t_3 - t_2}")
            new_plan = self.sp.plan(state, boxes)
            t_4 = time.time()
            print(f"Planning time: {t_4 - t_3}")

            with self.lock:
                self.next_plan = new_plan

    def execute(self):
        while self.running:
            # Get the current plan to execute
            with self.lock:
                if self.current_plan is None and self.next_plan is not None:
                    self.current_plan = self.next_plan
                    self.next_plan = None
            
            if self.current_plan is not None:
                # Execute the plan
                for action in self.current_plan[0]:
                    self.go1.move(action)
                    if self.go1.check_goal():
                        print("Goal reached. Stopping.")
                        self.running = False
                        break

    def start(self):
        self.plan()

        execution_thread = threading.Thread(target=self.execute)
        execution_thread.start()

        while not self.goal_reached:
            print("Re-planning for next steps...")
            self.plan()

        # Wait for execution thread to finish
        execution_thread.join()


if __name__ == '__main__':
    robot_controller = Robot_Plan()
    robot_controller.start()
