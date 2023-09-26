import numpy as np

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import time as tm
import ray
import pickle

from utils import non_det_filter, filter_reachable, show_trajectory, find_frontier, find_candidates, gen_trajectory, gen_path

[k1, k2, A, B, R, BRB] = pickle.load(open('sp_var.pkl','rb'))

class World:
    def __init__(self, world_box):
        self.w = world_box[1,0] - world_box[0,0]
        self.h = world_box[1,1] - world_box[0,1]
        self.occ_space = np.array([])

    def update(self, *new_boxes):
        '''Applies nondeterministic filter to update estimate of occupancy space'''
        new_occ_space = np.array(new_boxes)
        if new_occ_space.ndim == 1:
            new_occ_space = np.array([new_occ_space])
        self.occ_space = non_det_filter(self.occ_space, new_occ_space)

    def isValid(self, state):
        '''Collision check'''
        for i in range(self.occ_space.shape[0]):
            if (state[0] >= self.occ_space[i,0,0] 
                and state[0] <= self.occ_space[i,1,0] 
                and state[1] >= self.occ_space[i,0,1] 
                and state[1] <= self.occ_space[i,1,1]):
                return False
        return True
    
    def isValid_multiple(self, states):
        '''Check collision for multiple points'''
        for x in states:
            if not self.isValid(x):
                return False
        return True
    
    def isICSfree(self, state):
        '''Check for inevitable collision set'''
        # TODO: measure and update empirically
        x_brake = state[2]/k1
        y_brake = state[3]/k2
        new_state = np.array([state[0]+x_brake, state[1]+y_brake,0,0])
        if self.isValid(new_state):
            return True
        return False

    def show(self):
        '''Plot occupied space'''
        fig, ax = plt.subplots()
        ax.set_xlim([0,self.w])
        ax.set_ylim([0,self.h])
        for i in range(len(self.occ_space)):
            w = self.occ_space[i,1,0] - self.occ_space[i,0,0]
            h = self.occ_space[i,1,1] - self.occ_space[i,0,1]
            ax.add_patch(Rectangle(self.occ_space[i,0,:],w, h, edgecolor = 'k',fc=(0, 0.4470, 0.7410,0.5)))
        return fig, ax

class Safe_Planner:
    def __init__(self,
                 world_box: np.ndarray = np.array([[0,0],[8,8]]),
                 vx_range: list = [-4,4],
                 vy_range: list = [0,4],
                 goal: list = [6,7.8,0,0],
                 dt: float = 0.05, #time resolution for controls
                 sensor_dt: float = 1, #time resolution for perception update
                 r = 2, #cost threshold for reachability
                 radius = 0.3, #radius for finding intermediate goals
                 FoV = np.pi/2, #field of view
                 n_samples = 1000,
                 max_search_iter = 1000,
                 neighbor_radius = 0.5, #for non-dynamics planning
                 seed = 0):
        # load inputs
        self.world_box = world_box
        self.vx_range = vx_range
        self.vy_range = vy_range
        self.goal = goal
        self.world = World(world_box)
        self.dt = dt
        self.sensor_dt = sensor_dt
        self.r = r
        self.radius = radius
        self.FoV = FoV
        self.n_samples = n_samples
        self.max_search_iter = max_search_iter
        self.neighbor_radius = neighbor_radius
        self.prng = np.random.RandomState(seed)

        # initialize
        self.cost = np.zeros(n_samples)
        self.time = np.zeros(n_samples)
        self.time_to_come = np.zeros(n_samples)
        self.parent = np.arange(0,n_samples,1, dtype=int)
        self.bool_unvisit = np.ones(n_samples, dtype=bool)
        self.bool_unvisit[0] = False
        self.bool_closed = np.zeros(n_samples, dtype=bool)
        self.bool_open = np.zeros(n_samples, dtype=bool)
        self.bool_valid = np.ones(n_samples, dtype=bool)
        self.itr = 0

    # preparation
    def find_all_reachable(self):
        '''Samples points and computes reachable sets for all points'''
        self.Pset = []

        # sample random nodes
        while len(self.Pset) < self.n_samples:
            node = self.prng.uniform((0,0,min(self.vx_range),min(self.vy_range)), #mins
                                     (self.world.w,self.world.h,max(self.vx_range),max(self.vy_range))) #maxs
            self.Pset.append(node)
        self.Pset.append(self.goal)

        # pre-compute reachable sets
        @ray.remote # speed up
        def compute_reachable(node_idx):
            print(node_idx)
            node = self.Pset[node_idx]
            fset, fdist, ftime, ftraj = filter_reachable(node,self.Pset,self.r,self.vx_range,self.vy_range, 'F', self.dt)
            bset, bdist, btime, btraj = filter_reachable(node,self.Pset,self.r,self.vx_range,self.vy_range, 'B', self.dt)
            # Ginv_i = []
            # for j in range(len(btime)):
            #     Ginv_i.append(np.linalg.inv(gramian(btime[j])))
            return (node_idx,(fset, fdist, ftime, ftraj), (bset, bdist, btime, btraj))
        
        ray.init()
        futures = [compute_reachable.remote(node_idx) for node_idx in range(len(self.Pset))]
        self.reachable = ray.get(futures)
        ray.shutdown()

    def load_reachable(self, Pset, reachable):
        '''Load pre-computed reachable sets'''
        self.Pset = Pset
        self.reachable = reachable
        # for debugging

    def filter_neighbors(self, state, states):
        '''Reachability for non-dynamics planning'''
        neighbors = []
        distances = []
        for node_idx in states:
            node = self.Pset[node_idx]
            dist = np.linalg.norm(node[0:2]-state[0:2])
            if dist <= self.neighbor_radius:
                neighbors.append(node)
                distances.append(dist)
        return neighbors, distances
    
    def goal_inter(self, start_idx):
        '''Returns best intermediate goal to explore'''
        start = self.Pset[start_idx]
        v = np.sqrt(start[2]**2 + start[3]**2)
        goal_reachable = self.reachable[-1]       
        start_reachable = self.reachable[0]
        
        frontier = find_frontier(self.world.occ_space, self.world_box, self.Pset[0], self.FoV) # list of [x_start, y_start, x_end, y_end]
        candidates = find_candidates(frontier, self.radius) # list of segment midpoints

        costs = []
        subgoal_idxs = []
        
        for subgoal in candidates:
            subgoal_idx = np.argmin(np.linalg.norm(np.array(self.Pset) - np.append(subgoal, [0,0]),axis=1))
            subgoal_idxs.append(subgoal_idx)
            subgoal = self.Pset[subgoal_idx]
            subgoal_reachable = self.reachable[subgoal_idx]
            # cost to come
            self.Pset[-1] = subgoal
            self.reachable[-1] = subgoal_reachable
            _, _, cost_to_come = self.solve(start_idx, ICS=False)
            # cost to go
            self.Pset[0] = subgoal
            self.reachable[0] = subgoal_reachable
            self.Pset[-1] = self.goal
            self.reachable[-1] = goal_reachable
            _, _, dist_to_go = self.solve(start_idx, dynamics=False, ICS=False)
            # append + return to original
            costs.append(cost_to_come + dist_to_go/v)
            self.Pset[0] = start
            self.reachable[0] = start_reachable
        
        idx_incost = np.argmin(costs)
        return self.Pset[subgoal_idxs[idx_incost]], self.reachable[subgoal_idxs[idx_incost]]
    
    # plots
    def plot_reachable(self, direction):
        '''Plot reachability connections'''
        _, ax = plt.subplots()
        ax.set_xlim([0,self.world.w])
        ax.set_ylim([0,self.world.h])
        for i in range(self.n_samples):
            ax.scatter(self.Pset[i][0],self.Pset[i][1],color = 'k',marker = '.')
            if direction == 'F':
                fset = self.reachable[i][1]
                for j in range(len(fset[0])):
                    show_trajectory(ax, self.Pset[i],
                                    self.Pset[fset[0][j]],fset[1][j],self.dt)

            elif direction == 'B':
                bset = self.reachable[i][2]
                for j in range(len(bset[0])):
                    show_trajectory(ax, self.Pset[bset[0][j]],
                                    self.Pset[i],bset[1][j],self.dt)
        plt.show()
    
    def show(self,idx_solution):
        '''Plot solution'''
        fig, ax = self.world.show()
        for i in range(self.n_samples):
            ax.scatter(self.Pset[i][0],self.Pset[i][1], s=1, color = 'k', marker = '.')
        for i in range(len(idx_solution)-1):
            s0 = idx_solution[i] #idx
            s1 = idx_solution[i+1] #idx
            # Ginv = self.reachable[s1][2][3][self.reachable[s1][2][0].index(s0)]
            show_trajectory(ax, self.Pset[s0],self.Pset[s1],self.time[s1],self.dt, c_ = 'red', linewidth_ = 1)
        plt.show()
    
    def show_connection(self, idx_solution):
        '''Plot connected tree, solution, and world'''
        fig, ax = self.world.show()
        for i in range(self.n_samples):
            ax.scatter(self.Pset[i][0],self.Pset[i][1], s=1, color = 'k',marker = '.')
        for i in range(self.n_samples):
            if self.parent[i] != i:
                # s1 = i #idx
                s0 = self.parent[i] #idx
                # Ginv = self.reachable[i][2][3][self.reachable[i][2][0].index(s0)]
                show_trajectory(ax, self.Pset[s0], self.Pset[i],self.time[i],self.dt)
        for i in range(len(idx_solution)-1):
            s0 = idx_solution[i] #idx
            s1 = idx_solution[i+1] #idx
            Ginv = self.reachable[s1][2][3][self.reachable[s1][2][0].index(s0)]
            show_trajectory(ax, self.Pset[s0],self.Pset[s1],Ginv,self.time[s1],self.dt, c_ = 'red', linewidth_ = 1)
        plt.show()
    
    # safety planning algorithm
    def plan(self, state, *new_boxes): 
        #plan at new sensor time step when we get new boxes
        start = tm.time()
        # apply filter to update the world
        self.world.update(*new_boxes)

        # finds nearest sampled node to current state
        start_idx = np.argmin(np.linalg.norm(np.array(self.Pset) - np.array(state),axis=1))
        self.bool_open[start_idx] = True

        # check collision
        for node_idx in range(self.n_samples):
            node = self.Pset[node_idx]
            if not self.world.isValid(node):
                self.bool_valid[node_idx] = False
        
        # solve
        idx_solution, goal_flag, _ = self.solve(start_idx)
        if goal_flag == 0:
            self.Pset[-1], self.reachable[-1] = self.goal_inter(start_idx)
            idx_solution, goal_flag, _ = self.solve(start_idx)
        
        # output controls
        x_waypoints = []
        u_waypoints = []
        for i in range(len(idx_solution)-1):
            s0 = idx_solution[i] #idx
            s1 = idx_solution[i+1] #idx
            # Ginv = self.reachable[s1][2][3][self.reachable[s1][2][0].index(s0)]
            # x_waypoint, u_waypoint = gen_trajectory(self.Pset[s0],self.Pset[s1],self.time[s1],self.dt)
            x_waypoint, u_waypoint = self.reachable[s0][1][3][self.reachable[s0][1][0].index(s1)]
            x_waypoints.append(x_waypoint)
            u_waypoints.append(u_waypoint)

        end = tm.time()
        print('time: ', end-start)
        return idx_solution, x_waypoints, u_waypoints

    def solve(self, start_idx, dynamics = True, ICS = True):
        '''Main FMT* algorithm'''
        goal_flag = 0
        while self.itr <= self.max_search_iter:
            # print(self.itr)
            self = self.extend(dynamics, ICS)
            if not self.bool_unvisit[-1]: # goal node is visited
                goal_flag = 1
                break
        idx = self.n_samples-1 # goal index
        idx_solution = [idx]
        
        if goal_flag == 1:
            while True:
                idx = self.parent[idx]
                idx_solution.append(idx)
                if idx == start_idx: # start node 
                    break
        return idx_solution[::-1], goal_flag, self.cost[-1]

    def extend(self, dynamics, ICS):
        '''Inner loop of FMT*'''
        self.itr += 1
        
        # check nodes are collision-free
        idxset_open = np.where(self.bool_open & self.bool_valid)[0] # H
        idxset_unvisit = np.where(self.bool_unvisit & self.bool_valid)[0] # W
        idx_lowest = idxset_open[np.argmin(self.cost[idxset_open])] # z <- argmin cost(y)
        
        if dynamics:
            R_plus = self.reachable[idx_lowest][1][0]
            idxset_near = list(set(R_plus) & set(idxset_unvisit)) # X_near <- R+(z) \cap W
        else:
            idxset_near, _ = self.filter_neighbors(self.Pset[idx_lowest], idxset_unvisit)
        # for x in X_near
        for idx_near in idxset_near:
            # Y_near <- R-(x) \cap H
            if dynamics:
                R_minus = self.reachable[idx_near][2]
                idxset_cand = list(set(R_minus[0]) & set(idxset_open)) #index in Pset
                idxset_inR = [R_minus[0].index(i) for i in idxset_cand] #index in R-
                distset_cand = [R_minus[1][i] for i in idxset_inR]
                timeset_cand = [R_minus[2][i] for i in idxset_inR]
                Ginv_candidate = [R_minus[3][i] for i in idxset_inR]
            else:
                idxset_cand, distset_cand = self.filter_neighbors(self.Pset[idx_near], idxset_open)
                timeset_cand = [0]*len(idxset_cand)
            
            if len(idxset_cand) == 0:
                continue
            # ymin <- argmin cost(y) + dist(y,x)
            else:
                idx_incand_costmin = np.argmin(self.cost[idxset_cand] + distset_cand) #index in cand set
                cost_new = min(self.cost[idxset_cand] + distset_cand)
                time_new = timeset_cand[idx_incand_costmin]
                # Ginv = Ginv_candidate[idx_incand_costmin]
                idx_parent = idxset_cand[idx_incand_costmin]
            
                if dynamics:
                    # x_waypoints, _ = gen_trajectory(self.Pset[idx_parent],self.Pset[idx_near], time_new, self.dt)
                    idx_nearinparentfset = self.reachable[idx_parent][1][0].index(idx_near)
                    x_waypoints = self.reachable[idx_parent][1][3][idx_nearinparentfset][0]
                else:
                    x_waypoints = gen_path(self.Pset[idx_parent],self.Pset[idx_near],self.dt*max(self.v_range))

            def connect():
                self.bool_unvisit[idx_near] = False
                self.bool_open[idx_near] = True
                self.cost[idx_near] = cost_new
                self.time[idx_near] = time_new
                self.parent[idx_near] = idx_parent
                self.time_to_come[idx_near] = self.time_to_come[idx_parent] + time_new
            
            # check trajectory is collision-free
            if self.world.isValid_multiple(x_waypoints):
                # check ICS before sensor update
                if ICS and self.time_to_come[idx_parent] + time_new <= self.sensor_dt and self.world.isICSfree(self.Pset[idx_near]):
                    connect()
                else:
                    connect()
        self.bool_open[idx_lowest] = False
        self.bool_closed[idx_lowest] = True
        return self