import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import networkx as nx
from fmt import FMTPlanner
from fmt_dynamics import FMTPlanner_dynamics


def box_to_occ_space(*args):
    '''
    converts bounding box coordinates to occupied spaces
    use: box_to_occ_space(x_1, x_2, ..., x_n)
    args x_i: 2xd array of corner coords [[corners1_pred],[corners2_pred]] = [[x,y,z]_min,[x,y,z]_max] (for 3D)
    return x: nx2xd array of n sets of bounding box coordinates [x_1; x_2; ...; x_n]   
    '''  
    return np.array(args)

def visualize_2d(occ_space):
    '''
    Plots 2D occupied space
    '''
    fig, ax = plt.subplots()
    for i in range(occ_space.shape[0]):
        w = occ_space[i,1,0] - occ_space[i,0,0]
        h = occ_space[i,1,1] - occ_space[i,0,1]
        ax.add_patch(Rectangle(occ_space[i,0,:],w, h, edgecolor = 'k',fc=(0, 0.4470, 0.7410,0.5)))
        # plt.axis('off')
    plt.show()

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

def occ_to_map(occ_space, w, h, dx):
    '''
    converts rectangle coordinates to map
    inputs:
        occ_space: nx2xd array of bounding box coordinates
        w: width of map
        h: height of map
        dx: resolution of map
    outputs:
        map: wxh array of map, 0=obstacle, 1=free
    '''
    
    # initialize map
    map = np.ones((int(np.ceil(w/dx)),int(np.ceil(h/dx))))

    # iterate through bounding boxes
    for i in range(occ_space.shape[0]):
        # get bounding box
        box = occ_space[i,:,:]
        
        # get min and max indices
        x_min = int(np.floor(box[0,0]/dx))
        x_max = int(np.ceil(box[1,0]/dx))
        y_min = int(np.floor(box[0,1]/dx))
        y_max = int(np.ceil(box[1,1]/dx))
        
        # update map
        map[x_min:x_max,y_min:y_max] = 0

    return map

def ICS(occ_space, brake_dist):
    '''
    dependent on dynamics
    returns new occupancy space with ICS included
    '''
    
    # initialize new occupancy space
    occ_ics = np.zeros(occ_space.shape)
    # iterate through bounding boxes
    for i in range(occ_space.shape[0]):
        # get bounding box
        box = occ_space[i,:,:]
        # add brake distance
        box_ics =np.vstack((box[0,:] - brake_dist, box[1,:] + brake_dist))
        occ_ics[i,:,:] = box_ics
    return occ_ics

def visualize_result(map_design: np.ndarray, planner: FMTPlanner,
                     path_info: dict) -> None:
    plt.figure()
    plt.imshow(map_design, cmap="gray")
    nx.draw_networkx(planner.graph, [x[::-1] for x in planner.node_list],
            node_size=1,
            alpha=.5,
            with_labels=False)
    path = path_info["path"]
    plt.plot(path[:, 1], path[:, 0], 'r-', lw=2)

def plan(map_design, start, goal):
    '''
    Plan path to goal for time step $t$, ensuring that we do not enter the ICS at time step $t+1$.
    Inputs
        `occ_space`: nx2xd array of bounding boxes of obstacles
            n = # of boxes
            2 = corner_min, corner_max
            d = 2d or 3d (only 2d implementation for now)
        `start`: [x,y] index of start location
        `goal`: [x,y] index of goal location
    Outputs
        `path_info`: if exist, in order
            collision-free path to goal
            path to goal with no collision at next time step
            brake
    '''
    planner = FMTPlanner(map_design)
    path_info = planner.plan(start, goal) # plan collision-free path to goal
    if path_info['goal_flag'] == 0:
        path_temp = planner.plan(start, goal, optimistic = True) # plan path to goal with no collision at t+1
        if path_temp['goal_flag'] == 0:
            return 'brake'
        else:
            visualize_result(map_design, planner, path_temp)
            return path_temp

    else:
        visualize_result(map_design, planner, path_info)
        return path_info

# https://bryceboe.com/2006/10/23/line-segment-intersection-algorithm/
def ccw(A,B,C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

# Return true if line segments AB and CD intersect
def intersect(A,B,C,D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

# https://stackoverflow.com/questions/20677795/how-do-i-compute-the-intersection-point-of-two-lines
def line_intersection(line1, line2):
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
    # occ_space = 2D array
    # start = [x,y,psi]
    # FoV = total angle
    # returns frontier = list of np.array[[x1,y1],[x2,y2]]
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
                        # angle1 = np.mod(np.arctan((edge[0][1]-start[1])/(edge[0][0]-start[0])),np.pi)
                        # angle2 = np.mod(np.arctan((edge[1][1]-start[1])/(edge[1][0]-start[0])),np.pi)
                        # if angle1 <= ray <= angle2 or angle2 <= ray <= angle1:
                        if intersect(edge[0],edge[1],ray_line[0],ray_line[1]):
                            # print('same box', edge, ray, angle1, angle2)
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
    # print(ray_angles)
    # print(segments_dict)
    candidates = find_candidates(segments_dict, 0.05)
    fig, ax = plt.subplots()
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    for i in range(occ_space.shape[0]):
        w = occ_space[i,1,0] - occ_space[i,0,0]
        h = occ_space[i,1,1] - occ_space[i,0,1]
        ax.add_patch(Rectangle(occ_space[i,0,:],w, h, edgecolor = 'k',fc=(0, 0.4470, 0.7410,0.5)))
        # plt.axis('off')
    # for angle in ray_angles:
    #     ax.plot([start[0],start[0]+np.cos(angle)],[start[1],start[1]+np.sin(angle)],'r')
    for segment in segments_dict.values():
        ax.plot([segment[0,0],segment[1,0]],[segment[0,1],segment[1,1]])
    ax.scatter(*zip(*candidates))
    plt.show()

def goal_inter(occ_space, world_box, planner_dynamic, planner, start, goal, radius, FoV):
    """Returns best intermediate goal to explore"""
    
    frontier = find_frontier(occ_space, world_box, start, FoV) # list of [x_start, y_start, x_end, y_end]
    candidates = find_candidates(frontier, radius) # list of segment midpoints

    costs = []
    
    start_id = [int(np.floor(start[0]/planner.dx)), int(np.floor(start[1]/planner.dx)),start[2]]
    goal_id = [int(np.ceil(goal[0]/planner.dx)), int(np.ceil(goal[1]/planner.dx)),goal[2]]
    
    for subgoal in candidates:
        subgoal_id = [int(np.ceil(subgoal[0]/planner.dx)), int(np.ceil(subgoal[1]/planner.dx)),start[2]]
        print(subgoal_id)
        to_come = planner_dynamic.plan(start_id, subgoal_id, ICS=False)
        cost_to_come = to_come['cost']
        # speed_subgoal = to_come['speed'] # TODO: something like this
        # print(cost_to_come)
        to_go = planner.plan(subgoal_id[0:2], goal_id[0:2])
        dist_to_go = to_go['cost']
        #print(to_go['path'])

        costs.append(cost_to_come + dist_to_go) # constant speed
        # below is rough linear constant acceleration
        # TODO: somehow return speed, and fix cost for dynamics planner
        '''
        v_max = planner_dynamic.ux_max
        a_max = planner_dynamic.a_max
        t_to_max = (v_max - speed_subgoal)/a_max
        dist_to_max = speed_subgoal*t_to_max + 0.5*a_max*t_to_max**2
        if dist_to_go < dist_to_max:
            cost_to_go = np.sqrt(2*a_max*dist_to_go + speed_subgoal**2)/a_max #?
        else:
            cost_to_go = t_to_max + (dist_to_go - dist_to_max)/v_max
        costs.append(cost_to_come + cost_to_go)
        '''
    [x,y] = candidates[np.argmin(costs)]
    return [x,y,start[2]]