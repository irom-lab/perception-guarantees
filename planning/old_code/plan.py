from plan_func import *

def create_map(occ_prev,occ_est, brake_dist, w,h,dx, start,goal):
    # apply filter
    occ_now = non_det_filter(occ_prev, occ_est)
    # add ICS
    occ_now = ICS(occ_now, brake_dist)
    # convert to map
    map_design = occ_to_map(occ_now, w, h, dx)
    # convert start and goal to indices
    start_id = [int(np.floor(start[0]/dx)), int(np.floor(start[1]/dx))]
    goal_id = [int(np.ceil(goal[0]/dx)), int(np.ceil(goal[1]/dx))]
    return map_design, start_id, goal_id

if __name__ == "__main__":
    A1x = 0.1; A1y = 0.1; A2x=0.2; A2y=0.2
    B1x = 0.3; B1y = 0.3; B2x=0.4; B2y=0.8
    C1x = 0.35; C1y = 0.6; C2x=0.45; C2y=0.7
    D1x = 0.15; D1y = 0.15; D2x=0.7; D2y=0.5
    # bounding boxes t-1
    A = np.array([[A1x,A1y],[A2x,A2y]])
    B = np.array([[B1x,B1y],[B2x,B2y]])
    # bounding boxes t
    C = np.array([[C1x,C1y],[C2x,C2y]])
    D = np.array([[D1x,D1y],[D2x,D2y]])
    # brake distance
    brake_dist = 0.04
    # width, height, resolution
    w = 1; h = 1; dx = 0.01
    # start and goal
    start = [0,0]; goal = [1,1]

    '''CREATE MAP'''
    map_design, start_id, goal_id = create_map(box_to_occ_space(A,B),box_to_occ_space(C,D), brake_dist, w,h,dx, start,goal)

    '''FIND PATH'''
    path_plan = plan(map_design, start_id, goal_id)

