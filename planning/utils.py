import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
from scipy.linalg import expm
import scipy.optimize as opt
from scipy.sparse import bmat

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import pickle

[k1, k2, A, B, R, BRB] = pickle.load(open('sp_var.pkl','rb'))


# WORLD FUNCTIONS
def overlap(a, b): # 2d implementation
    '''
    returns rectangle overlap between two rectangles
    inputs: a, b, rectangles [[xmin, ymin], [xmax, ymax]] (2d)
    returns: overlapped rectangle, or a, b if no overlap
    '''

    ab1x = max(a[0,0], b[0,0])
    ab1y = max(a[0,1], b[0,1])

    dx = min(a[1,0], b[1,0]) - ab1x
    dy = min(a[1,1], b[1,1]) - ab1y
    if (dx>=0) and (dy>=0):
        return np.array([[[ab1x, ab1y], [ab1x+dx, ab1y+dy]]])
    else:
        # return np.array([a, b])
        return None

def non_det_filter(xbar_prev, xhat_now):
    '''
    implements non-deterministic filter
    inputs: xbar_prev, xhat_now
    xbar_prev: previous estimate of occ_space
    xhat_now: current estimate of occ_space
    
    output: xbar_now, updated estimate of occ_space
    '''
    if len(xbar_prev) == 0:
        return xhat_now
    else:
        ab_all = []
        for a in xbar_prev:
            for b in xhat_now:
                # print(ab)
                ab = overlap(a, b)
                if (np.any(np.all(a == ab))) or (np.any(np.all(b == ab))) or (ab is None):
                    continue
                else:
                    ab_all.append(ab)
        if len(ab_all) == 0:
            return []
        else:
            xbar_now = np.concatenate(ab_all)
            return xbar_now
        

# DYNAMICS FUNCTIONS: Reachability
def gramian(t):
    '''computes weighted controllability gramian'''
    def gdot(t,g0):
        g = g0.reshape(4,4)
        res = A@g + g@A.T + BRB
        return res.reshape(-1)
    return solve_ivp(gdot,[0,t],np.zeros(16)).y[:,-1].reshape(4,4)

def cost_optimal(state0, state1, r):
    def cost(t):
        x = state1-expm(A*t)@state0
        G = gramian(t)
        if np.linalg.det(G) == 0:
            # G = G + np.eye(4)
            print(G)
        c =t+x.T@np.linalg.inv(G)@x
        return c
    t_star = opt.minimize(cost, r/2, bounds = [(0,r)], method = 'trust-constr', options={'gtol': 1e-3, 'maxiter': 10}).x[0]
    return cost(t_star), t_star

def forward_box(state, r, vx_range, vy_range):
    '''rough filtering of forward-reachable set'''
    vxmax = max(vx_range)
    vymax = max(vy_range)
    # cost >= time, so t_max = r
    xmax,ymax = state[0:2] + r*np.array([vxmax,vymax])
    xmin,ymin = state[0:2]
    return xmax,ymax,xmin,ymin

def backward_box(state, r, vx_range, vy_range):
    '''rough filtering of back-reachable set'''
    vxmin = min(vx_range)
    vymin = min(vy_range)
    vxmax = max(vx_range)
    vymax = max(vy_range)
    xmax,ymax = state[0:2] - r*np.array([vxmin,vymin])
    xmin,ymin = state[0:2] - r*np.array([vxmax,vymax])
    return xmax,ymax,xmin,ymin

def filter_reachable(state: np.ndarray, state_set: list, r, vx_range, vy_range, direction: str):
    """
    Filter reachable states

    Args:
        state (np.ndarray): Current state
        state_set (list): Set of state indices to filter
        r: cost threshold
        vx_range (list): Range of vx
        vy_range (list): Range of vy
        direction (str): 'F' for forward, 'B' for backward

    Returns:
        state_set_filtered (list): Filtered set of state indices
        cost_set_filtered (list): costs corresponding to filtered state indices
        time_set_filtered (list): times corresponding to filtered state indices
    """
    if direction == 'F':
        box = forward_box
    elif direction == 'B':
        box = backward_box
    
    xmax, ymax, xmin, ymin= box(state,r,vx_range,vy_range)
    state_set_filtered = []
    cost_set_filtered = []
    time_set_filtered = []
    for idx in range(len(state_set)):
        state_i = state_set[idx]
        if np.any(state_i != state):
            if xmin <= state_i[0] <= xmax and ymin <= state_i[1] <= ymax:
                if direction == 'F':
                    cost, time = cost_optimal(state, state_i, r)
                elif direction == 'B':
                    cost, time = cost_optimal(state_i, state, r)
                if cost <= r:
                    state_set_filtered.append(idx)
                    cost_set_filtered.append(cost)
                    time_set_filtered.append(time)
    return state_set_filtered, cost_set_filtered, time_set_filtered


# DYNAMICS FUNCTIONS: Trajectory
def gen_trajectory(s0, s1, Ginv, tau, dt):
    '''
    Generates the optimal trajectory connecting two points in state space
    
    Inputs:
        s0: initial state
        s1: final state
        Ginv: inverse of weighted controllability gramian (pre-computed to speed up computation)
        tau: optimal connection time
        dt: time step
    '''
    sbar = expm(A*tau)@s0
    d = Ginv@(s1 - sbar)
    
    block_mat = bmat([[A,BRB.T],[None,-A.T]]).toarray()
    block_vec = np.hstack([s1,d])
    
    def xydot(t,xy):
        return block_mat@xy

    waypoints = solve_ivp(xydot,[tau,0],block_vec,t_eval = np.arange(tau, 0, -dt)).y
    x_waypoints = waypoints[0:4,:].T
    u_waypoints = (np.linalg.inv(R)@B.T@waypoints[4:,:]).T

    return x_waypoints, u_waypoints

def show_trajectory(ax, s0, s1, Ginv, tau, dt, c_='gray', linewidth_=0.5):
    x_waypoints,_ = gen_trajectory(s0, s1, Ginv, tau, dt)
    M = np.zeros((4, int(np.abs(np.ceil(tau/dt)))))
    for i in range(len(x_waypoints)):
        M[:, i] = x_waypoints[i]
    ax.plot(M[0, :], M[1, :], c=c_, linewidth=linewidth_)

def gen_path(s0, s1, dx):
    '''Generates straight path without dynamics'''
    dx, dy = s1[0] - s0[0], s1[1] - s0[1]
    yaw = np.arctan2(dy, dx)
    d = np.hypot(dx, dy)
    steps = np.arange(0, d, dx).reshape(-1, 1)
    pts = s0[0:2] + steps * np.array([np.cos(yaw), np.sin(yaw)])
    return np.vstack((pts, s1[0:2]))


# SUBGOAL FUNCTIONS
def ccw(A,B,C):
    # https://bryceboe.com/2006/10/23/line-segment-intersection-algorithm/
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

def intersect(A,B,C,D):
    # Return true if line segments AB and CD intersect

    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def line_intersection(line1, line2):
    # https://stackoverflow.com/questions/20677795/how-do-i-compute-the-intersection-point-of-two-lines

    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       return False

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return [x, y]

def find_frontier(xbar_now, world_box, start, FoV):
    '''
    Finds the unknown frontier
    
    Inputs:
        xbar_now: current estimate of occ_space
        world_box: world boundary
        start: current state
        FoV: field of view
    
    Output:
        frontier: dictionary of rays and corresponding segments
    '''
    ray_left = start[2]+FoV/2
    ray_right = start[2]-FoV/2
    box_vertices = {}
    vertex_rays = {'left':ray_left,'right':ray_right}
    rays = []
    frontier = {}
    scaling = np.linalg.norm(world_box[0]-world_box[1])
    
    world_vertices = [world_box[0],np.array([world_box[1][0],world_box[0][1]]),world_box[1],np.array([world_box[0][0],world_box[1][1]])]
    world_edges = np.array([[world_vertices[i],world_vertices[(i+1)%4]] for i in range(4)])
    
    # create {box: 4 vertices}
    # find all rays corresponding to all vertices
    for i in range(len(xbar_now)):
        obstacle = xbar_now[i]
        vertices = [obstacle[0],np.array([obstacle[1][0],obstacle[0][1]]),obstacle[1],np.array([obstacle[0][0],obstacle[1][1]])]
        box_vertices[i] = vertices
        for j in range(len(vertices)):
            vertex = vertices[j]
            ray = np.mod(np.arctan2(vertex[1]-start[1],vertex[0]-start[0]),2*np.pi)
            if ray_right <= ray <= ray_left: # vertex is in FoV
                vertex_rays[vertex.tobytes()] = ray
    
    # discard rays in the way
    for vert, ray in vertex_rays.items():
        show = True
        if type(vert) == bytes:
            vertex = np.frombuffer(vert)
        else:
            vertex = np.array(start[0:2])
        ray_line = np.array([start[0:2],start[0:2]+scaling*np.array([np.cos(ray),np.sin(ray)])])
        
        segments = []
        for world_edge in world_edges:
            if intersect(world_edge[0],world_edge[1],ray_line[0],ray_line[1]):
                intersection = line_intersection(world_edge, ray_line)
                segment = np.array([vertex, intersection])
        segments.append(segment)

        # for this ray, look at each box
        for box, vertices in box_vertices.items():
            edges =np.array([[vertices[i],vertices[(i+1)%4]] for i in range(4)])
            if np.any(np.all(vertices == vertex, axis = 1)): # same box
                # make sure ray doesn't intersect with its own box
                for edge in edges:
                    if np.any(np.all(edge == vertex, axis = 1)): # this edge is for this vertex
                        continue
                    else:
                        if intersect(edge[0],edge[1],ray_line[0],ray_line[1]):
                            show = False
            else: # different box
                # discard rays blocked by other boxes
                # get edges
                for edge in edges:
                    # find intersection
                    if intersect(edge[0],edge[1],ray_line[0],ray_line[1]):
                        intersection = line_intersection(edge, ray_line)
                        # compute distance
                        dist_to_edge = np.linalg.norm(np.array(intersection)-np.array(start[0:2]))
                        dist_to_vertex = np.linalg.norm(vertex-np.array(start[0:2]))
                        if dist_to_edge < dist_to_vertex:
                            # print('different box', edge, ray, intersection, dist_to_edge, dist_to_vertex)
                            show = False
                        else:
                            segments.append(np.array([vertex, intersection]))
                        
        if ray not in rays and show == True:
            rays.append(ray)
            segments_len = np.array([np.linalg.norm(segment[0]-segment[1]) for segment in segments])
            frontier[ray] = segments[np.argmin(segments_len)]
    
    return frontier

def find_candidates(frontier, radius):
    '''find subgoal candidates'''
    candidates = []
    for ray_angle, segment in frontier.items():
        # segment: [[x1,y1],[x2,y2]]
        segment_len = np.linalg.norm(segment[0]-segment[1])
        num_segments = int(np.floor(0.5*segment_len/radius))
        for i in range(num_segments):
            candidates.append(segment[0] + np.array([2*(i+1/2)*radius*np.cos(ray_angle),2*(i+1/2)*radius*np.sin(ray_angle)]))
    return candidates

def plot_frontier(occ_space, world_box, start, FoV):
    segments_dict = find_frontier(occ_space, world_box, start, FoV)
    candidates = find_candidates(segments_dict, 0.05)
    fig, ax = plt.subplots()
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    for i in range(occ_space.shape[0]):
        w = occ_space[i,1,0] - occ_space[i,0,0]
        h = occ_space[i,1,1] - occ_space[i,0,1]
        ax.add_patch(Rectangle(occ_space[i,0,:],w, h, edgecolor = 'k',fc=(0, 0.4470, 0.7410,0.5)))
    for segment in segments_dict.values():
        ax.plot([segment[0,0],segment[1,0]],[segment[0,1],segment[1,1]])
    ax.scatter(*zip(*candidates))
    plt.show()