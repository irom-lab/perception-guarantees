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
            start: np.ndarray,
            goal: np.ndarray,
            world_box: np.ndarray = np.array([[0, 0],[1, 1]]),
            dx: float = 0.01,
            radius: float = 0.1,
            FoV: float = np.pi/2,):
        
        self.occ_space = occ_space
        self.start = start
        self.goal = goal
        self.world_box = world_box
        self.w = world_box[1,0] - world_box[0,0]
        self.h = world_box[1,1] - world_box[0,1]
        self.dx = dx
        self.radius = radius
        self.FoV = FoV

        
        self.map_design = occ_to_map(self.occ_space, self.w, self.h, self.dx)

        self.planner = FMTPlanner(self.map_design)
        self.planner_dynamic = FMTPlanner_dynamics(self.map_design)

        self.subgoal = goal_inter(self.occ_space, self.world_box, self.planner_dynamic, 
                                  self.planner, self.start, self.goal, self.radius, self.FoV)


    def plan(self):
        start_id = [int(np.floor(self.start[0]/self.dx)),int(np.floor(self.start[1]/self.dx)),self.start[2]]
        subgoal_id = [int(np.floor(self.subgoal[0]/self.dx)),int(np.floor(self.subgoal[1]/self.dx)),self.subgoal[2]]
        self.path_info = self.planner_dynamic.plan(start_id, subgoal_id)
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