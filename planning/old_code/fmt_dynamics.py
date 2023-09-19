import numpy as np
import networkx as nx
from scipy.spatial import cKDTree
from pqdict import pqdict


class FMTPlanner_dynamics():
    def __init__(
        self,
        map_design: np.ndarray,
        dt: float = 1,
        dx: float = 0.01,
        # psi_resolution: float = 0.1,
        n_samples: int = 1000,
        # r_n: float = 20.0,
        path_resolution: float = 0.001,
        rr: float = 0.01,
        max_search_iter: int = 10000,
        seed: int = 0,
        u_range = np.array([-0.5,0.5]), # placeholder
        k = np.array([0.1,0.1]), # placeholder
        
        # ux_max: float = 0.2, # placeholder
        # ux_min: float = 0, # placeholder
        # upsi_max: float =  np.pi/2, # placeholder
        # upsi_min: float = -np.pi/2, # placeholder
        # a_max: float = 1, # placeholder
    ):
        """
        Dynamically-constrained Fast Marching Tree Path Planner 

        Args:
            map_design (np.ndarray): Obstacle map described by a binary image. 1: free nodes; 0: obstacle nodes
            dt (float, optional): Time step. Defaults to 0.1.
            dx (float, optional): Spatial resolution. Defaults to 0.01.
            psi_resolution (float, optional): Angular resolution. Defaults to 0.1.
            n_samples (int, optional): Number of nodes to sample. Defaults to 1000.
            r_n (float, optional): Range to find neighbor nodes. Defaults to .0.
            path_resolution (float, optional): Resolution of paths to check collisions. Defaults to 0.1.
            rr (float, optional): Distance threshold to check collisions. Defaults to 1.0.
            max_search_iter (int, optional): Number of maximum iterations. Defaults to 10000.
            seed (int, optional): Random seed. Defaults to 0.
        """

        # hyperparameters
        self.map_size = map_design.shape
        self.dt = dt
        self.dx = dx
        # self.psi_resolution = psi_resolution
        self.path_resolution = path_resolution/dx
        self.rr = rr/dx
        self.n_samples = n_samples
        # self.r_n = r_n
        self.max_search_iter = max_search_iter
        self.prng = np.random.RandomState(seed)  # initialize PRNG

        self.ux_max = max(u_range)/dx # ux_max/dx
        self.ux_min = min(u_range)/dx # ux_min/dx
        self.k1 = k[0]
        self.k2 = k[1]
        self.A = np.array([[0,0,1,0],[0,0,0,1],[0,0,-self.k1,0],[0,0,0,-self.k2]])
        self.B = np.array([[0,0],[0,0],[self.k1,0],[0,self.k2]])
        self.R = np.array([[1,0],[0,1]])

        self.r_n =self.ux_max*self.dt # radius to search within
        # self.upsi_max = upsi_max
        # self.upsi_min = upsi_min
        # self.amax = a_max/dx

        # construct obstacle tree
        obstacles = np.argwhere(map_design == 0)# *self.dx
        self.obstacles_tree = cKDTree(obstacles)

        # initialize graph
        self.graph = nx.Graph()
        self.node_list = list()
        i = 0
        while len(self.node_list) < self.n_samples:
            node = self.prng.uniform(0, (self.map_size[0],self.map_size[1],np.pi*2))# (self.map_size[0]*self.dx,self.map_size[1]*self.dx,np.pi*2))
            if self.check_collision(node, None):
                self.node_list.append(node)
                self.graph.add_node(i)
                i += 1

    def plan(self,
             start: np.ndarray,
             goal: np.ndarray,
             heuristic_weight: int = 0.0,
             dynamics: bool = True,
             ICS: bool = True) -> dict:
        """
        Run path planning

        Args:
            start (np.ndarray): Start location
            goal (np.ndarray): Goal location
            heuristic_weight (int, optional): Weight for Euclidean heuristics. Defaults to 0.0.
            dynamics (bool, optional): Whether to use dynamics constraints. Defaults to True.
            ICS (bool, optional): Whether to check ICS violations. Defaults to True.

        Returns:
            dict:Containing path, number of steps required, and goal flag
        """
        start = np.asarray(start)
        goal = np.asarray(goal)
        assert self.check_collision(start, None)
        assert self.check_collision(goal, None)
        if ICS:
            assert self.check_ICS(start)
            assert self.check_ICS(goal)

        self.graph.remove_edges_from(list(self.graph.edges))
        start_id = len(self.node_list)
        goal_id = start_id + 1
        for n_steps, node in zip([start_id, goal_id], [start, goal]):
            if ICS:
                if self.check_ICS(node):
                    self.graph.add_node(n_steps)
                    self.node_list.append(node)
            else:
                self.graph.add_node(n_steps)
                self.node_list.append(node)
        # print(self.node_list)
        node_tree = cKDTree(self.node_list)
        heuristic = [np.linalg.norm(x - goal) for x in self.node_list]

        # initialize
        goal_flag = 0
        z = start_id
        V_open = pqdict({z: 0.}) # H
        V_closed = list()
        V_unvisited = list(range(len(self.node_list))) # W
        V_unvisited.remove(z) # W <-V\x_init

        # start search
        for n_steps in range(self.max_search_iter): # while z not in X_goal
            # print(n_steps)
            if z == goal_id:
                print("Reached goal")
                goal_flag = 1
                break
            N_z = node_tree.query_ball_point(self.node_list[z], self.r_n)
            if dynamics:
                R_plus = self.filter_reachable(z, N_z, self.forward_box)
            else:
                R_plus = N_z
            X_near = list(set(R_plus) & set(V_unvisited))
            # print('step: ', n_steps, 'z: ', z, 'R_plus: ', len(R_plus), 'X_near: ', len(X_near))

            for x in X_near:
                N_x = node_tree.query_ball_point(self.node_list[x], self.r_n)
                if dynamics:
                    R_minus = self.filter_reachable(x, N_x, self.backward_box)
                else:
                    R_minus = N_x
                Y_near = list(set(R_minus) & set(V_open))
                # print('N_x: ', len(N_x), 'R-: ', len(R_minus), 'V_open: ', list(set(V_open)), 'Y_near:', Y_near)
                if Y_near != []:
                    y_min = Y_near[np.argmin([V_open[y] for y in Y_near])]
                    if self.check_collision(self.node_list[y_min], self.node_list[x]):
                    
                        self.graph.add_edge(y_min, x)
                        if x in V_open:
                            val = V_open[y_min] + np.linalg.norm(self.node_list[y_min] - self.node_list[x]) + heuristic_weight * (-heuristic[y_min] + heuristic[x])
                            V_open.updateitem(x, val)
                        else:
                            V_open.additem(x, V_open[y_min] + np.linalg.norm(self.node_list[y_min] - self.node_list[x]) + heuristic_weight * (-heuristic[y_min] + heuristic[x]))
                        # print('V_open: ', V_open)
                        V_unvisited.remove(x)
            V_open.pop(z)
            V_closed.append(z)
            if len(V_open) == 0:
                print("Search failed")
                break
            z = V_open.top()


        path = np.vstack([
            self.node_list[x]
            for x in nx.shortest_path(self.graph, start_id, z)
        ])

        if goal_flag == 0:
            cost = 10*np.sum(self.map_size)
        else:
            p1 = np.delete(path,0,axis = 0)
            p1 = np.vstack((p1,path[-1,:]))
            cost = np.sum(np.linalg.norm(p1[:,0:2]-path[:,0:2],axis = 1)) #in terms of path length

        return {
            "path": path,
            "n_steps": n_steps,
            "cost": cost,
            "goal_flag": goal_flag,
        }

    def forward_box(self,state,dt):
        # state_t: [x_t, y_t, psi_t]
        xmin = state[0] + dt*min(np.cos(state[2])*self.ux_min,np.cos(state[2])*self.ux_max)
        xmax = state[0] + dt*max(np.cos(state[2])*self.ux_min,np.cos(state[2])*self.ux_max)
        ymin = state[1] + dt*min(np.sin(state[2])*self.ux_min,np.sin(state[2])*self.ux_max)
        ymax = state[1] + dt*max(np.sin(state[2])*self.ux_min,np.sin(state[2])*self.ux_max)
        psimin = state[2] + dt*self.upsi_min
        psimax = state[2] + dt*self.upsi_max
        
        if xmax == xmin:
            xmax += 0.5*self.rr
            xmin -= 0.5*self.rr
        if ymax == ymin:
            ymax += 0.5*self.rr
            ymin -= 0.5*self.rr

        return xmin,xmax,ymin,ymax,psimin,psimax
    
    def backward_box(self,state,dt):
        psimin = state[2]-dt*self.upsi_max
        psimax = state[2]-dt*self.upsi_min

        # these ranges are custom to dt=1, upsi in [-pi/2,pi/2]
        if -np.pi/2 <= psimin <= 0 <= psimax <= np.pi/2:
            # cos is positive, max is 1
            # sin is monotonic increasing
            xmin = state[0] - dt*self.ux_max
            xmax = state[0] - dt*self.ux_min*min(np.cos(psimin),np.cos(psimax))
            ymin = state[1] - dt*self.ux_max*np.sin(psimax)
            ymax = state[1] - dt*self.ux_max*np.sin(psimin)

        elif 0 <= psimin <= np.pi/2 <= psimax <= np.pi:
            # cos is monotonic decreasing
            # sin is positive, max is 1
            xmin = state[0] - dt*self.ux_max*np.cos(psimin)
            xmax = state[0] - dt*self.ux_max*np.cos(psimax)
            ymin = state[1] - dt*self.ux_max
            ymax = state[1] - dt*self.ux_min*min(np.sin(psimin),np.sin(psimax))
        
        elif np.pi/2 <= psimin <= np.pi <= psimax <= 3*np.pi/2:
            # cos is negative, min is -1
            # sin is monotonic decreasing
            xmin = state[0] - dt*self.ux_min*max(np.cos(psimin),np.cos(psimax))
            xmax = state[0] + dt*self.ux_max
            ymin = state[1] - dt*self.ux_max*np.sin(psimin)
            ymax = state[1] - dt*self.ux_max*np.sin(psimax)
        
        elif -np.pi/2 <= psimin <= 0 <= np.pi/2 <= psimax <= np.pi:
            # sin max = 1, sin min = sin(psimin) negative
            # cos max = 1, cos min = cos(psimax) negative
            xmin = state[0] - dt*self.ux_max
            xmax = state[0] - dt*self.ux_max*np.cos(psimax)
            ymin = state[1] - dt*self.ux_max
            ymax = state[1] - dt*self.ux_max*np.sin(psimin)
        
        elif 0 <= psimin <= np.pi/2 <= np.pi <= psimax <= 3*np.pi/2:
            # sin max = 1, sin min = sin(psimax) negative
            # cos min = -1, cos max = cos(psimin) positive
            xmin = state[0] - dt*self.ux_max*np.cos(psimin)
            xmax = state[0] + dt*self.ux_max
            ymin = state[1] - dt*self.ux_max
            ymax = state[1] - dt*self.ux_max*np.sin(psimax)

        elif -np.pi/2 <= psimin <= 0 <= np.pi <= psimax <= 3*np.pi/2:
            # sin max = 1, sin min = min(sin(psimin),sin(psimax)) negative
            # cos min = -1, cos max = 1
            xmin = state[0] - dt*self.ux_max
            xmax = state[0] + dt*self.ux_max
            ymin = state[1] - dt*self.ux_max
            ymax = state[1] - dt*self.ux_max*min(np.sin(psimin),np.sin(psimax))

        else:
            xmin = state[0] - dt*max(np.cos(psimin)*self.ux_min,np.cos(psimin)*self.ux_max,np.cos(psimax)*self.ux_min,np.cos(psimax)*self.ux_max)
            xmax = state[0] - dt*min(np.cos(psimin)*self.ux_min,np.cos(psimin)*self.ux_max,np.cos(psimax)*self.ux_min,np.cos(psimax)*self.ux_max)
            ymin = state[1] - dt*max(np.sin(psimin)*self.ux_min,np.sin(psimin)*self.ux_max,np.sin(psimax)*self.ux_min,np.sin(psimax)*self.ux_max)
            ymax = state[1] - dt*min(np.sin(psimin)*self.ux_min,np.sin(psimin)*self.ux_max,np.sin(psimax)*self.ux_min,np.sin(psimax)*self.ux_max)

        return xmin,xmax,ymin,ymax,psimin,psimax
    
    def filter_reachable(self,state_idx: int, state_set_idx: list, box: callable) -> list:
        """
        Filter reachable states

        Args:
            state (np.ndarray): Current state
            state_set (np.ndarray): Set of states to filter
            dt (float): Time step

        Returns:
            np.ndarray: Filtered set of states
        """
        state = self.node_list[state_idx]
        
        xmin,xmax,ymin,ymax,psimin,psimax = box(state,self.dt)
        state_set_filtered = []
        for idx in state_set_idx:
            state_i = self.node_list[idx]
            if xmin <= state_i[0] <= xmax and ymin <= state_i[1] <= ymax and psimin <= state_i[2] <= psimax:
                state_set_filtered.append(idx)
        return state_set_filtered
    
    def check_collision(self, src: np.ndarray, dst: np.ndarray) -> bool:
        """
        Check collision

        Args:
            src (np.ndarray): Source node
            dst (np.ndarray): Destination node

        Returns:
            bool: True if no collisions were found and False otherwise
        """
        pr = self.path_resolution
        if (dst is None) | np.all(src == dst):
            return self.obstacles_tree.query(src[0:2])[0] > self.rr

        dx, dy = dst[0] - src[0], dst[1] - src[1]
        yaw = np.arctan2(dy, dx)
        d = np.hypot(dx, dy)
        steps = np.arange(0, d, pr).reshape(-1, 1)
        pts = src[0:2] + steps * np.array([np.cos(yaw), np.sin(yaw)])
        pts = np.vstack((pts, dst[0:2]))
        return bool(self.obstacles_tree.query(pts)[0].min() > self.rr)
    
    def check_ICS(self, state: np.ndarray)-> bool:
        """
        Check if node is in ICS

        Args:
            state (np.ndarray): Node to check

        Returns:
            bool: True if no ICS violation
        """
        
        state_t1 = self.A@state
        violation_dist = self.obstacles_tree.query([state_t1[0],state_t1[1]])[0]
        
        return bool(violation_dist > self.rr)