import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import rankdata
from scipy.linalg import expm

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import time as tm
import pickle
from multiprocessing import Pool

from shapely.geometry import Point, MultiPolygon, Polygon, LineString, MultiPoint
from shapely.ops import unary_union
from shapely.geometry.polygon import orient
from shapely.validation import make_valid
from planning.utils import turn_box, non_det_filter, filter_reachable, find_frontier, find_polygon

[k1, k2, A, B, R, BRB] = pickle.load(open('planning/sp_var.pkl','rb'))
num_parallel = 3


class Ray:
    def __init__(self, angle, vert):
        self.start = vert[0]
        self.end = vert[1]
        self.angle = angle
    def find_box(self, geoms, frontier, world):
        buffer = 2e-5 
        self.start_box = LineString([[0,0],[0,0]])
        self.end_box = LineString([[0,0],[0,0]])
        ab = LineString(frontier[self.angle])
        x = ab.intersection(world)
        if not x.is_empty:
            self.end_box = world
        for geom in geoms:
            geom_buff = geom.buffer(buffer)

            x1 = ab.intersection(geom)
            x2 = ab.intersection(geom_buff)
            if not x1.is_empty:
                x_coords = np.array([x1.coords.xy[0][0],x1.coords.xy[1][0]])
                if np.all(abs(self.start-x_coords)<buffer):
                    self.start_box = geom.exterior
                elif np.all(abs(self.end-x_coords)<buffer):
                    self.end_box = geom.exterior
            elif not x2.is_empty:
                x_coords = np.array([x2.coords.xy[0][0],x2.coords.xy[1][0]])
                if np.all(abs(self.start-x_coords)<buffer):
                    self.start_box = geom.exterior
                elif np.all(abs(self.end-x_coords)<buffer):
                    self.end_box = geom.exterior
            
class World:
    def __init__(self, world_box):
        self.w = world_box[1,0] - world_box[0,0]
        self.h = world_box[1,1] - world_box[0,1]
        # self.occ_space = np.array([])
        self.counter = 0
        self.free_space = None
        self.free_space_new = None

    def update(self, new_boxes):
        '''Applies nondeterministic filter to update estimate of occupancy space'''
        new_occ_space = np.array(new_boxes)
        if self.counter == 0:
            self.box_space = unary_union(turn_box(new_boxes))
        else:
            self.box_space = non_det_filter(self.box_space, new_occ_space)
        self.occ_space = unary_union(turn_box(new_boxes))

        if self.occ_space.geom_type == 'Polygon':
            self.occ_space = MultiPolygon([self.occ_space])
        if self.box_space.geom_type == 'Polygon':
            self.box_space = MultiPolygon([self.box_space])

        self.occ_space.simplify(1e-5)
        self.occ_space = MultiPolygon([orient(s, sign=-1.0) for s in self.occ_space.geoms if s.geom_type == 'Polygon'])
        
        self.box_space.simplify(1e-5)
        self.box_space = MultiPolygon([orient(s, sign=-1.0) for s in self.box_space.geoms if s.geom_type == 'Polygon'])

        self.counter += 1

    def isValid(self, state):
        '''Collision check'''
        return self.free_space.buffer(1e-5).contains(Point(state[0],state[1]))

    def isValid_multiple(self, states):
        '''Check collision for multiple points'''
        return self.free_space.buffer(1e-5).contains(MultiPoint(states[:,0:2]))

    def isICSfree(self, state):
        '''Check for inevitable collision set'''
        # TODO: measure and update empirically
        # x_brake = 0.12512712
        # y_brake = 0.1972604
        new_state = expm(A*10**3)@state
        if self.isValid(new_state):
            return True

        # x_brake = state[2]/k1
        # y_brake = state[3]/k2
        # x_brake = 0.5/k1#+0.5
        # y_brake = 0.5/k2#+0.5 # 0.5 is approximate state estimation error
        # new_state = np.array([state[0]+x_brake, state[1]+y_brake,0,0])
        # if self.isValid(new_state):
        #     return True
        # return False
        # return False

    def show(self, true_boxes = None):
        '''Plot occupied space'''
        fig, ax = plt.subplots()
        ax.set_xlim([0,self.w])
        ax.set_ylim([0,self.h])
        ax.set_aspect('equal')
        if true_boxes is not None:
            for box in true_boxes:
                ax.add_patch(Rectangle((box[0,0],box[0,1]),box[1,0]-box[0,0],box[1,1]-box[0,1],edgecolor = 'k',fc=(1,0,0,0.8)))
        if self.occ_space.geom_type == 'Polygon':
            self.occ_space = MultiPolygon([self.occ_space])
        for geom in self.occ_space.geoms:
            xs, ys = geom.exterior.xy
            ax.fill(xs,ys, fc=(1,0,0,0.3))
        for geom in self.box_space.geoms:
            xs, ys = geom.exterior.xy
            ax.fill(xs,ys, fc=(1,1,0,0.9))
        if self.free_space_new is not None:
            if self.free_space_new.geom_type == 'Polygon':
                xs, ys = self.free_space_new.exterior.xy
                ax.fill(xs,ys, edgecolor = 'k', linestyle='--', fc = (0,0,0,0))
            else:
                for geom in self.free_space_new.geoms:
                    if geom.geom_type == 'Polygon':
                        xs, ys = geom.exterior.xy
                        ax.fill(xs,ys, edgecolor = 'k', linestyle='--', fc = (0,0,0,0))
                    else:
                        print('plotting problem:', geom.geom_type)
        if self.free_space is not None:
            if self.free_space.geom_type == 'Polygon':
                xs, ys = self.free_space.exterior.xy
                ax.fill(xs,ys, edgecolor = 'k',fc=(0, 0.4470, 0.7410,0.5))
            elif self.free_space.geom_type == 'MultiPolygon':
                for geom in self.free_space.geoms:
                    xs, ys = geom.exterior.xy
                    ax.fill(xs,ys, edgecolor = 'k',fc=(0, 0.4470, 0.7410,0.5))
            else:
                print(self.free_space.geom_type)
        return fig, ax

    def show_occlusion(self,polygons):
        fig, ax = self.show()
        ax.set_xlim([0,self.w])
        ax.set_ylim([0,self.h])
        for polygon in polygons:
            ax.fill(polygon[0],polygon[1],edgecolor = 'None',alpha=0.5)
        return fig, ax

class Safe_Planner:
    def __init__(self,
                 # Pset: list,
                 world_box: np.ndarray = np.array([[0,0],[8,8]]),
                 vx_range: list = [-0.5,0.5],
                 vy_range: list = [0,1],
                 sr: float = 1.0, # initial clearance
                 init_state: list = [4,1,0,0.5],
                 goal_f: list = [7,-2,0.5,0], # forrestal coordinates
                 dt: float = 0.1, #time resolution for controls
                 sensor_dt: float = 1, #time resolution for perception update
                 r = 3, #cost threshold for reachability
                 radius = 0.5, #radius for finding intermediate goals
                 FoV = 60*np.pi/180, #field of view
                 FoV_range = 5, #range of field of view
                 FoV_close = 1,
                 n_samples = 2000,
                 max_search_iter = 1000,
                 weight = 10,  #5, #weight for cost to go
                 seed = 0):
        # load inputs

        # self.Pset = Pset

        self.world_box = world_box
        self.vx_range = vx_range
        self.vy_range = vy_range
        self.init_state = init_state
        self.world = World(world_box)
        self.goal_f = goal_f
        self.goal = self.state_to_planner(self.goal_f)
        # self.pool = Pool(num_parallel)


        self.world.free_space = Polygon(((init_state[0]-sr, init_state[1]-sr),
                                         (init_state[0]-sr, init_state[1]+sr),
                                         (init_state[0]+sr, init_state[1]+sr),
                                         (init_state[0]+sr, init_state[1]-sr)))
        self.dt = dt
        self.sensor_dt = sensor_dt
        self.r = r
        self.radius = radius
        self.FoV = FoV
        self.FoV_range = FoV_range
        self.FoV_close = FoV_close
        self.n_samples = n_samples
        self.max_search_iter = max_search_iter
        self.weight = weight
        self.prng = np.random.RandomState(seed)

        # initialize
        self.cost = np.zeros(n_samples+1)
        self.time = np.zeros(n_samples+1)
        self.time_to_come = np.zeros(n_samples+1)
        self.parent = np.arange(0,n_samples+1,1, dtype=int)
        self.bool_unvisit = np.ones(n_samples+1, dtype=bool)
        # self.bool_unvisit[0] = False
        self.bool_closed = np.zeros(n_samples+1, dtype=bool)
        self.bool_open = np.zeros(n_samples+1, dtype=bool)
        self.bool_valid = np.ones(n_samples+1, dtype=bool)
        self.itr = 0

    def state_to_planner(self, state):
        # convert robot state to planner coordinates
        # return (np.array([[[0,-1,0,0],[1,0,0,0],[0,0,0,-1],[0,0,1,0]]])@np.array(state) + np.array([self.world.w/2,0,0,0]))[0]
        state = np.array(state)
        state = np.atleast_2d(state)
        origin_shift = np.atleast_2d(np.array([self.world.w/2,0,0,0]))

        state_tf = (np.array([[[0,-1,0,0],[1,0,0,0],[0,0,0,-1],[0,0,1,0]]]) @ state.T + origin_shift.T)
        return state_tf.squeeze()

    # preparation

    def find_all_reachable(self):
        '''Computes reachable sets for all pre-sampled points'''

        import ray

        # sample random nodes
        # while len(self.Pset) < self.n_samples:
        #     node = self.prng.uniform((0,0,min(self.vx_range),min(self.vy_range)), #mins
        #                              (self.world.w,self.world.h,max(self.vx_range),max(self.vy_range))) #maxs
        #     self.Pset.append(node)
        # self.Pset.append(self.goal)

        # sample nodes on grid
        self.Pset = []
        for i in np.linspace(0.5,self.world.w-0.5,20):
            for j in np.linspace(0.5,self.world.h-0.5,20):
                for k in np.linspace(min(self.vx_range),max(self.vx_range),5):
                    # for l in np.linspace(min(self.vy_range),max(self.vy_range),3):
                    self.Pset.append([i,j,k,0.5])
        self.n_samples = len(self.Pset)
        self.Pset.append(self.goal)

        # pre-compute reachable sets
        @ray.remote # speed up
        def compute_reachable(node_idx):
            # print(node_idx)
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

    def find_goal_reachable(self, reachable):
        '''Computes reachable sets for goal node'''
        self.reachable = reachable
        fset, fdist, ftime, ftraj = filter_reachable(self.goal,self.Pset,self.r,self.vx_range,self.vy_range, 'F', self.dt)
        bset, bdist, btime, btraj = filter_reachable(self.goal,self.Pset,self.r,self.vx_range,self.vy_range, 'B', self.dt)
        for idx in range(len(bset)):
            self.reachable[bset[idx]][1][0].append(self.n_samples)
            self.reachable[bset[idx]][1][1].append(bdist[idx])
            self.reachable[bset[idx]][1][2].append(btime[idx])
            self.reachable[bset[idx]][1][3].append(btraj[idx])
        for idx in range(len(fset)):
            self.reachable[fset[idx]][2][0].append(self.n_samples)
            self.reachable[fset[idx]][2][1].append(fdist[idx])
            self.reachable[fset[idx]][2][2].append(ftime[idx])
            self.reachable[fset[idx]][2][3].append(ftraj[idx])
        self.reachable[self.n_samples] = (self.n_samples,(fset, fdist, ftime, ftraj), (bset, bdist, btime, btraj))

    def load_reachable(self, Pset, reachable):
        '''Load pre-computed reachable sets'''
        self.Pset = Pset
        self.reachable = reachable

    def filter_neighbors(self, state, states):
        '''Reachability for non-dynamics planning'''
        neighbors = []
        distances = []
        for node_idx in states:
            node = self.Pset[node_idx]
            dist = np.linalg.norm(np.array(node[0:2])-np.array(state[0:2]))
            if dist <= self.neighbor_radius:
                neighbors.append(node_idx)
                distances.append(dist)
        return neighbors, distances

    def goal_inter(self, start_idx):
        '''Returns best intermediate goal to explore'''
        start = self.Pset[start_idx]
        v = np.sqrt(start[2]**2 + start[3]**2)

        if self.world.free_space.geom_type == 'Polygon':
            candidates = [np.array(self.world.free_space.exterior.interpolate(t).xy).reshape(2) for t in
                          np.linspace(0,self.world.free_space.length,
                                      int(np.floor(self.world.free_space.length/self.radius)),False)]
        else:
            candidates = []
            for geom in self.world.free_space.geoms:
                candidates += [np.array(geom.exterior.interpolate(t).xy).reshape(2) for t in
                               np.linspace(0,geom.length,
                                           int(np.floor(geom.length/self.radius)),False)]

        
        cand_obj = MultiPoint(np.array(candidates))
        candidates = np.array(candidates)[np.where(self.world.occ_space.buffer(1e-5).contains(cand_obj.geoms)==False)[0]]
        # fig, ax = self.world.show()
        # if len(candidates) > 0:
        #    ax.scatter(*zip(*candidates))
        # plt.show()

        costs = []
        subgoal_idxs = []
        for subgoal in candidates:
            subgoal_idx_all = np.where(rankdata(
                cdist(np.array(self.Pset)[:,0:2],subgoal.reshape(1,2)).flatten(), method='min'
                )-1==0)[0]
            valid_idx = np.where(self.bool_valid[subgoal_idx_all])[0]
            subgoal_idx = subgoal_idx_all[valid_idx]
            subgoal_idxs += list(subgoal_idx)
        
        for subgoal_idx in subgoal_idxs:
            if any([all(i == self.Pset[subgoal_idx][0:2]) for i in self.goal_explored]):
                # this subgoal is already explored
                costs.append(np.inf)
            else:
                subgoal = self.Pset[subgoal_idx]
                # cost to come
                self.goal_idx = subgoal_idx
                if self.bool_unvisit[subgoal_idx]==False:
                    _, cost_to_come = self.solve(start_idx)
                else:
                    cost_to_come = np.inf
                # cost to go
                self.goal_idx = self.n_samples
                if subgoal[1]<=self.goal[1]:
                    dist_to_go = np.linalg.norm(np.array(subgoal[0:2])-np.array(self.goal[0:2]))
                else:
                    dist_to_go = np.inf
                # append
                costs.append(cost_to_come + self.weight*dist_to_go/v)
        # print('costs: ', costs)
        if all(np.isinf(costs)):
            return None, None
        else:
            idx_incost = np.argmin(costs)
            self.goal_idx = subgoal_idxs[idx_incost]
            return self.Pset[subgoal_idxs[idx_incost]]

    def occlusion(self, state):
        '''Returns occlusion polygon'''
        start =state[0,0:2]
        world = LineString([[0,0],[0,self.world.h],[self.world.w,self.world.h],[self.world.w,0],[0,0]])

        # frontier = find_frontier(self.world.occ_space, self.world_box, start, self.FoV)
        frontier = find_frontier(self.world.box_space, self.world_box, start, self.FoV)
        rays = list(frontier.keys())
        rays.sort()
        ray_objects = []
        for ray in rays:
            ray_object = Ray(ray,frontier[ray])
            # ray_object.find_box(self.world.occ_space.geoms, frontier, world)
            ray_object.find_box(self.world.box_space.geoms, frontier, world)
            ray_objects.append(ray_object)

        vs = find_polygon(ray_objects, world)
        if len(vs) >= 4:
            return make_valid(Polygon(vs))
        else:
            return None

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
                    # show_trajectory(ax, self.Pset[i],
                    #                 self.Pset[fset[0][j]],fset[1][j],self.dt)
                    traj = self.reachable[i][1][3][j][0]
                    ax.plot(traj[:,0], traj[:,1], c='gray', linewidth=0.5)

            elif direction == 'B':
                bset = self.reachable[i][2]
                for j in range(len(bset[0])):
                    # show_trajectory(ax, self.Pset[bset[0][j]],
                    #                 self.Pset[i],bset[1][j],self.dt)
                    traj = self.reachable[i][2][3][j][0]
                    ax.plot(traj[:,0], traj[:,1], c='gray', linewidth=0.5)
        plt.show()

    def show(self,idx_solution, true_boxes = None):
        '''Plot solution'''
        fig, ax = self.world.show(true_boxes)
        for i in range(len(idx_solution)-1):
            s0 = idx_solution[i] #idx
            s1 = idx_solution[i+1] #idx
            x_waypoints = self.reachable[s0][1][3][self.reachable[s0][1][0].index(s1)][0]
            ax.plot(x_waypoints[:,0], x_waypoints[:,1], c='red', linewidth=1)
        ax.plot(self.Pset[self.goal_idx][0],self.Pset[self.goal_idx][1],'o')
        plt.show()

    def show_connection(self, idx_solution):
        '''Plot connected tree, solution, and world'''
        fig, ax = self.world.show()
        ax.set_aspect('equal')
        for i in range(self.n_samples):
            ax.scatter(self.Pset[i][0],self.Pset[i][1], s=1, color = 'k',marker = '.')
        for i in range(self.n_samples):
            if self.parent[i] != i:
                # s1 = i #idx
                s0 = self.parent[i] #idx
                # Ginv = self.reachable[i][2][3][self.reachable[i][2][0].index(s0)]
                # show_trajectory(ax, self.Pset[s0], self.Pset[i],self.time[i],self.dt)
                x_waypoints = self.reachable[s0][1][3][self.reachable[s0][1][0].index(i)][0]
                ax.plot(x_waypoints[:,0], x_waypoints[:,1], c='gray', linewidth=0.5)
        for i in range(len(idx_solution)-1):
            s0 = idx_solution[i] #idx
            s1 = idx_solution[i+1] #idx
            # Ginv = self.reachable[s1][2][3][self.reachable[s1][2][0].index(s0)]
            # show_trajectory(ax, self.Pset[s0],self.Pset[s1],self.time[s1],self.dt, c_ = 'red', linewidth_ = 1)
            x_waypoints = self.reachable[s0][1][3][self.reachable[s0][1][0].index(s1)][0]
            ax.plot(x_waypoints[:,0], x_waypoints[:,1], c='red', linewidth=1)
        ax.plot(self.Pset[self.goal_idx][0],self.Pset[self.goal_idx][1],'o')
        plt.show()

    def plot_velocity(self, res):
        '''Plot velocity profile'''
        fig, axs = plt.subplots(3,2)
        fig.suptitle('Velocity Profile')
        fig.tight_layout(pad=2.0)
        axs[0,0].set_title('$v_x(t)$')
        axs[0,1].set_title('$u_x(t)$')
        axs[1,0].set_title('$v_y(t)$')
        axs[1,1].set_title('$u_y(t)$')
        axs[2,0].set_title('$v(t)$')
        axs[2,1].set_title('$y(t)$')
        t = 0
        for i in range(len(res[0])-1):

            t_array = np.linspace(t+self.dt,t+len(res[1][i])*self.dt,len(res[1][i]))
            t = t_array[-1]
            axs[0,0].plot(t_array, res[1][i][:,2],'k-')
            axs[0,1].plot(t_array, res[2][i][:,0],'k-')
            axs[1,0].plot(t_array, res[1][i][:,3],'k-')
            axs[1,1].plot(t_array, res[2][i][:,1],'k-')
            axs[2,0].plot(t_array, np.sqrt(res[1][i][:,2]**2 + res[1][i][:,3]**2),'k-')
            axs[2,1].plot(t_array, res[1][i][:,1],'k-')
        plt.show()

    def check_collision(self,node_idx):
            state = self.Pset[node_idx]
            self.bool_valid[node_idx] = self.world.free_space.buffer(1e-5).contains(Point(state[0],state[1]))

    # safety planning algorithm
    def plan(self, state, new_boxes):
        #plan at new sensor time step when we get new boxes
        start = tm.time()

        # apply filter to update the world
        sense_range = state[0,1]+self.FoV_range*np.cos(self.FoV/2)
        toofar = np.array([[[self.world_box[0,0],state[0,1]+sense_range],
                            [self.world_box[1,0],self.world_box[1,1]]]])
        tooclose = Polygon([[state[0,0]-self.FoV_close,state[0,1]-self.FoV_close],
                            [state[0,0]+self.FoV_close,state[0,1]-self.FoV_close],
                            [state[0,0]+self.FoV_close,state[0,1]+self.FoV_close],
                            [state[0,0]-self.FoV_close,state[0,1]+self.FoV_close],
                            [state[0,0]-self.FoV_close,state[0,1]-self.FoV_close]])
        if state[0,1] + sense_range <= self.world_box[1,1]:
            new_boxes = np.append(new_boxes,toofar,axis=0)
    
        self.world.update(new_boxes)
        # occlusion
        if self.world.occ_space.contains(Point(state[0,0],state[0,1])) == False:
            self.world.free_space_new = self.occlusion(state)
            if self.world.free_space_new is not None:
                self.world.free_space_new = self.world.free_space_new.difference(tooclose)
                if self.world.free_space_new.is_valid:            
                    self.world.free_space = self.world.free_space.union(self.world.free_space_new)
                    self.world.free_space.simplify(1e-5)
        if self.world.free_space.geom_type == 'GeometryCollection':
            self.world.free_space = MultiPolygon([geom for geom in self.world.free_space.geoms if geom.geom_type == 'Polygon'])
        
        # initialize planner
        self.cost = np.zeros(self.n_samples+1)
        self.time = np.zeros(self.n_samples+1)
        self.time_to_come = np.zeros(self.n_samples+1)
        self.parent = np.arange(0,self.n_samples+1,1, dtype=int)
        self.bool_unvisit = np.ones(self.n_samples+1, dtype=bool)
        self.bool_closed = np.zeros(self.n_samples+1, dtype=bool)
        self.bool_open = np.zeros(self.n_samples+1, dtype=bool)
        self.bool_valid = np.zeros(self.n_samples+1, dtype=bool)
        self.itr = 0

        # check collision
        point_objects = MultiPoint(np.array(self.Pset)[:,0:2])
        self.bool_valid = self.world.free_space.contains(point_objects.geoms)

        # finds nearest valid sampled node to current state
        start_idx_all = np.argsort(cdist(np.array(self.Pset),np.array(state)), axis=0)
        valid_idx = np.where(self.bool_valid[start_idx_all])[0]
        start_idx = start_idx_all[valid_idx[0]][0]
        self.goal_idx = self.n_samples
        self.bool_open[start_idx] = True

        # solve
        goal_flag = self.build_tree(start_idx)
        self.goal_explored = []
        if goal_flag == 1:
            idx_solution, _ = self.solve(start_idx)
        else:
            while goal_flag == 0:
                goal_loc = self.goal_inter(start_idx)
                if goal_loc is None or self.goal_idx == self.n_samples:
                    goal_flag = -1
                    idx_solution = [self.goal_idx]
                    print('planning failed, stay')
                    break
                else:
                    self.goal_explored.append(self.Pset[self.goal_idx][0:2])
                    print('intermediate goal: ', self.Pset[self.goal_idx])

                    if self.bool_unvisit[self.goal_idx]==False:
                        idx_solution, _ = self.solve(start_idx)
                        break

        # output controls
        x_waypoints = []
        u_waypoints = []
        for i in range(len(idx_solution)-1):
            s0 = idx_solution[i] #idx
            s1 = idx_solution[i+1] #idx
            x_waypoint, u_waypoint = self.reachable[s0][1][3][self.reachable[s0][1][0].index(s1)]
            x_waypoints.append(x_waypoint)
            u_waypoints.append(u_waypoint)

        end = tm.time()
        print('planning time: ', end-start)
        return idx_solution, x_waypoints, u_waypoints

    def build_tree(self, start_idx):
        self.bool_open[start_idx] = True
        goal_flag = 0
        while self.itr <= self.max_search_iter:
            self = self.extend()
            if not self.bool_unvisit[self.goal_idx]: # goal node is visited
                goal_flag = 1
                break
        print('num_iteration:', self.itr)
        self.itr=1
        return goal_flag

    def solve(self, start_idx):
        '''Main FMT* algorithm'''
        idx = self.goal_idx
        idx_solution = [idx]
        while True:
            idx = self.parent[idx]
            idx_solution.append(idx)
            if idx == start_idx: # start node
                break

        tot_cost = self.cost[self.goal_idx]

        return idx_solution[::-1], tot_cost

    def extend(self):
        '''Inner loop of FMT*'''
        self.itr += 1

        # check nodes are collision-free
        self.idxset_open = np.where(self.bool_open & self.bool_valid)[0] # H
        idxset_unvisit = np.where(self.bool_unvisit & self.bool_valid)[0] # W
        if self.idxset_open.size != 0:
            idx_lowest = self.idxset_open[np.argmin(self.cost[self.idxset_open])] # z <- argmin cost(y)

            R_plus = self.reachable[idx_lowest][1][0]
            idxset_near = list(set(R_plus) & set(idxset_unvisit)) # X_near <- R+(z) \cap W
            # for x in X_near
            for idx_near in idxset_near:
                self.process_idx(idx_near)
                
            self.bool_open[idx_lowest] = False
            self.bool_closed[idx_lowest] = True

        return self

    def process_idx(self, idx_near):
        R_minus = self.reachable[idx_near][2]
        idxset_cand = list(set(R_minus[0]) & set(self.idxset_open)) #index in Pset
        idxset_inR = np.where(np.isin(R_minus[0], idxset_cand))[0]
        distset_cand = np.array(R_minus[1])[idxset_inR]
        timeset_cand = np.array(R_minus[2])[idxset_inR]

        if len(idxset_cand) == 0:
            pass

        # ymin <- argmin cost(y) + dist(y,x)
        idx_incand_costmin = np.argmin(self.cost[idxset_cand] + distset_cand) #index in cand set
        cost_new = min(self.cost[idxset_cand] + distset_cand)
        time_new = timeset_cand[idx_incand_costmin]
        idx_parent = idxset_cand[idx_incand_costmin]
        idx_nearinparentfset = self.reachable[idx_parent][1][0].index(idx_near)
        x_waypoints = self.reachable[idx_parent][1][3][idx_nearinparentfset][0]
        
        # check trajectory is collision-free
        if self.world.isValid_multiple(x_waypoints):
            # check ICS before sensor update
            if (self.time_to_come[idx_parent] + time_new <= self.sensor_dt
                and self.world.isICSfree(self.Pset[idx_near])):
                self.connect(idx_near,cost_new,time_new,idx_parent)
            elif self.time_to_come[idx_parent] + time_new > self.sensor_dt:
                self.connect(idx_near,cost_new,time_new,idx_parent)


    def connect(self,idx_near, cost_new, time_new, idx_parent):
        # print("connect")
        self.bool_unvisit[idx_near] = False
        self.bool_open[idx_near] = True
        self.cost[idx_near] = cost_new
        self.time[idx_near] = time_new
        self.parent[idx_near] = idx_parent
        self.time_to_come[idx_near] = self.time_to_come[idx_parent] + time_new