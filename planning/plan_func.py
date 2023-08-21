import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import networkx as nx
from fmt.fmt import FMTPlanner


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
    ab_all = []
    for a in xbar_prev:
        for b in xhat_now:
            # print(ab)
            ab = overlap(a, b)
            if (np.any(np.all(a == ab))) or (np.any(np.all(b == ab))) or (ab is None):
                continue
            else:
                ab_all.append(ab)
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
    planner = FMTPlanner(map_design, n_samples=1000, r_n=20, path_resolution=0.1, rr=1.0, max_search_iter=10000)
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