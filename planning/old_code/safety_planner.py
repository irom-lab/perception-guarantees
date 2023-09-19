from fmt_dynamics import FMTPlanner_dynamics
from fmt import FMTPlanner
from plan_func import occ_to_map, goal_inter
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx


class Safety_Planner():
    def __init__(
            self,
            occ_space: np.ndarray,
            world_box: np.ndarray = np.array([[0, 0],[1, 1]]),
            dx: float = 0.01,
            radius: float = 0.1,
            FoV: float = np.pi/2,):
        
        self.occ_space = occ_space
        self.world_box = world_box
        self.w = world_box[1,0] - world_box[0,0]
        self.h = world_box[1,1] - world_box[0,1]
        self.dx = dx
        self.radius = radius
        self.FoV = FoV
        
               
        self.map_design = occ_to_map(self.occ_space, self.w, self.h, self.dx)

        self.planner = FMTPlanner(self.map_design, dx=self.dx)
        self.planner_dynamic = FMTPlanner_dynamics(self.map_design, dx=self.dx)


    def find_subgoal(self, start, goal):
        subgoal = goal_inter(self.occ_space, self.world_box, self.planner_dynamic, 
                                  self.planner, start, goal, self.radius, self.FoV)
        subgoal_id = [int(np.floor(subgoal[0]/self.dx)),int(np.floor(subgoal[1]/self.dx)),subgoal[2]]
        return subgoal_id
    
    def plan(self,
            start: np.ndarray,
            goal: np.ndarray,):
        start_id = [int(np.floor(start[0]/self.dx)),int(np.floor(start[1]/self.dx)),start[2]]
        goal_id = [int(np.floor(goal[0]/self.dx)),int(np.floor(goal[1]/self.dx)),goal[2]]

        self.path_info = self.planner_dynamic.plan(start_id, goal_id)
        if self.path_info['goal_flag'] == 0:
            self.path_info = self.planner_dynamic.plan(start_id, self.find_subgoal(start_id, goal_id))
        return self.path_info
    
    def visualize_result(self):
        plt.figure()
        plt.imshow(self.map_design, cmap="gray")
        nx.draw_networkx(self.planner_dynamic.graph, [x[0:2][::-1] for x in self.planner_dynamic.node_list],
                node_size=1,
                alpha=.5,
                with_labels=False)
        path = self.path_info["path"]
        plt.plot(path[:, 1], path[:, 0], 'r-', lw=2)