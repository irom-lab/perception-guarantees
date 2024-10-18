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
        self.num_samples = 1050  # number of samples used for the precomputed files
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

        self.goal_reached = False
        self.count_used_last_plan = 1

        f = open(self.reachable_file, 'rb')
        self.reachable = pickle.load(f)
        f = open(self.pset_file, 'rb')
        self.Pset = pickle.load(f)

        self.sp = Safe_Planner(goal_f=self.goal_forrestal, sensor_dt=self.sensor_dt, dt=self.dt, n_samples=self.num_samples, radius=self.radius, max_search_iter=self.max_search_iter, speed=1)
        print("goal (planner coords): ", self.sp.goal)

        self.sp.load_reachable(self.Pset, self.reachable)
        self.go1 = Go1_move(self.sp.goal_f, vicon=self.vicon, state_type=self.state_type, save_folder=self.result_dir)

        self.last_plan = None
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

    def transform_plan(self, arr):
        print("input array", arr)
        if len(arr) == 0:
            print(self.last_plan[2])
            if len(self.last_plan[2]) > 1:
                alternative_plan = np.vstack(self.last_plan[2])
            else:
                alternative_plan = np.array(self.last_plan[2])

            print("alt plan", alternative_plan)
            policy_before_transformation = alternative_plan[20*self.count_used_last_plan:]
            self.count_used_last_plan += 1
            #print("before transform,", policy_before_transformation)
        elif len(arr) > 1:
            policy_before_transformation = np.vstack(arr)
            print("when length >1")
        else:
            print("when else")
            policy_before_transformation = np.array(arr)[0]
        print("before transform", policy_before_transformation)
        policy = (np.array([[0,1],[-1,0]])@policy_before_transformation.T).T
        #print("post transform", policy)

        return policy

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
        # Perception + planning logic
        print("starting planning !!!!!")
        
        with self.lock:
            if self.current_plan is None:
                future_state = self.go1.get_state()[0]
            elif len(self.current_plan[1]) > 1:
                print("current plan, length > 1", self.current_plan[1])
                union_plans = np.vstack(self.current_plan[1])
                print("union plans", union_plans)
                future_state = union_plans[min(19, len(union_plans)-1)]
            elif len(self.current_plan[1]) == 1:
                print("current plan, length == 1", self.current_plan[1][0])
                future_state = self.current_plan[1][0][min(19, len(self.current_plan[1][0])-1)]
            else:
                print("current plan 0,", self.current_plan[1])
                future_state = self.go1.get_state()[0]
            state = self.state_to_planner(future_state)

        
        print("predicted/real state: ", state)
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
        
        if self.current_plan is not None:
            self.sp.show(self.current_plan, boxes)

        with self.lock:
            self.next_plan = new_plan
            print("New plan ready.")
        
        self.plan_ready.set()

    def execute(self):
        while not self.goal_reached:
            self.plan_ready.wait()

            with self.lock:
                if self.current_plan is None and self.next_plan is not None:
                    # Load the next plan into current_plan
                    self.current_plan = self.next_plan
                    self.next_plan = None
            
            if self.current_plan is not None:
                # Execute the current plan
                self.executing = True
                transformed_plan = self.transform_plan(self.current_plan[2])
                num_action = 1
                for action in transformed_plan[:20]:
                    self.go1.move(action)
                    print("action num: ", num_action)
                    if self.go1.check_goal():
                        self.goal_reached = True
                        print("Goal reached. Stopping.")
                        self.executing = False
                        return
                    num_action +=1
                # Mark that execution is finished and ready for a new plan
                self.executing = False
                self.plan_ready.clear()
                with self.lock:
                    if self.current_plan is not None:
                        self.last_plan = self.current_plan
                        self.count_used_last_plan = 1
                    self.current_plan = None



    def start(self):
        # Create an Event object to synchronize planning and execution
        self.plan_ready = threading.Event()

        # Step 1: Initial Plan
        self.plan()  # Create the first plan

        # Step 2: Start execution of the initial plan
        execution_thread = threading.Thread(target=self.execute)
        execution_thread.start()

        # Step 3: Concurrent planning during execution
        while not self.goal_reached:
            if not self.executing:  # Only plan when execution is ongoing
                print("Re-planning during execution...")
                self.plan()  # Re-plan while execution is happening

        # Wait for execution to finish
        execution_thread.join()


if __name__ == '__main__':
    robot_controller = Robot_Plan()
    robot_controller.start()
