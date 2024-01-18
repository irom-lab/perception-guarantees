from planning.Safe_Planner import *
import rospy
from geometry_msgs.msg import TransformStamped
from scipy.spatial.transform import Rotation
import numpy as np
from utils.go1_move import *
from utils.plotting import *
import time


def state_to_planner(state, sp):
    # convert robot state to planner coordinates
    return np.array([[[0,-1,0,0],[1,0,0,0],[0,0,0,-1],[0,0,1,0]]])@np.array(state) + np.array([sp.world.w/2,0,0,0])

def state_to_go1(state, sp):
    x, y, vx, vy = state
    return np.array([y, -x+sp.world.w/2, vy, -vx])

def boxes_to_planner(boxes, sp):
    boxes_new = np.zeros_like(boxes)
    for i in range(len(boxes)):
        boxes_new[i,:,:] = np.reshape(np.array([[[0,0,0,-1],[1,0,0,0],[0,-1,0,0],[0,0,1,0]]])@np.reshape(boxes[0],(4,1)),(2,2)) + np.array([sp.world.w/2,0])
    return boxes_new

def boxes_to_planner_frame(boxes, sp):
    boxes_new = np.zeros_like(boxes)
    for i in range(len(boxes)):
        #boxes_new[i,:,:] = np.reshape(np.array([[[0,0,0,-1],[1,0,0,0],[0,-1,0,0],[0,0,1,0]]])@np.reshape(boxes[0],(4,1)),(2,2)) + np.array([sp.world.w/2,0])
        boxes_new[i,0,0] = -boxes[i,1,1] + sp.world.w/2
        boxes_new[i,1,0] = -boxes[i,0,1] + sp.world.w/2
        boxes_new[i,:,1] =  boxes[i,:,0]
    return boxes_new

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


def plan_loop():
    # ****************************************************
    # SET DESIRED FLAGS
    vicon = True # set if want vicon state active; If true, make sure ./vicon.sh is running from pg_ws (and zed box connected to gas dynamics network)
    state_type = 'zed' # 'vicon' if want to set the state used as the vicon state, 'zed' 
    replan = True # set if want to just follow open loop plan
    save_traj = True  # set if want to save trajectory and compare against plan
    plot_traj = True  # set if want to visualize trajectory at the end of execution
    result_dir = 'results/debug_trial1/' # set to unique trial identifier if saving results
    goal_forrestal = [7.0, -2.0, 0.0, 0.0] # goal in forrestal coordinates
    reachable_file = 'planning/pre_compute/reachable_10Hz.pkl'
    pset_file = 'planning/pre_compute/Pset_10Hz.pkl'
    num_samples = 2000  # number of samples used for the precomputed files
    dt = 0.1 #   planner dt
    radius = 0.1 # distance between intermediate goals on the frontier
    chairs = [2, 3]  # list of chair labels to be used to get ground truth bounding boxes
    num_detect = 10  # number of boxes for 3DETR to detect
    cp = 1.0
    sensor_dt = 1 # time in seconds to replan
    # ****************************************************
    if vicon:
        rospy.init_node('listener', anonymous=True)
        chair_states = GroundTruthBB(chairs)
        time.sleep(3)
        
 
    vicon_traj = []
    state_traj = []
    plan_traj = []

    # load pre-computed: need to recompute for actual gains
    f = open(reachable_file, 'rb')
    reachable = pickle.load(f)
    f = open(pset_file, 'rb')
    Pset = pickle.load(f)

    # initialize planner
    sp = Safe_Planner(goal_f=goal_forrestal, sensor_dt=sensor_dt,dt=dt, n_samples=num_samples, radius=radius)
    print("goal (planner coords): ", sp.goal)
    
    # *** Alternate commenting of two lines below if goal changes
    # sp.find_goal_reachable(reachable)
    sp.load_reachable(Pset, reachable)

    go1 = Go1_move(sp, vicon=vicon, state_type=state_type)
    go1.get_state()
    # print(go1.state)
    # time.sleep(2)
    time.sleep(dt)
    chair_states_bb = chair_states.get_true_bb()
    ground_truth = boxes_to_planner_frame(chair_states_bb, sp)
    print("ground truth bb (planner coords): ", ground_truth, ground_truth.shape)
    
    # ****************************************************
    # QUICK MOTION/STATE DEBUG CODE. Comment out planning
    # for t in range(100):
    #     if vicon:
    #         go1.get_true_state()
    #         print("true: ", go1.true_state)
    #         go1.move([0.5, 0.0])
                
    #     go1.get_state()
    #     print("state: ", go1.state)

    #     time.sleep(0.2)

    
    # ****************************************************
    # GET INITIAL PLAN
    
    t = 0
    cp = 0.0

    # perception + cp
    # boxes = get_boxes(sp)
    # boxes = np.array([[[2.0,4.0],[3.0,6.0]]])
    # boxes[:,0,:] -= cp
    # boxes[:,1,:] += cp
    boxes = go1.camera.get_boxes(cp, num_detect)
    boxes = boxes[:,:,0:2]
    # print("Boxes before planner transform ",  boxes)
    boxes = boxes_to_planner_frame(boxes, sp)
    # print("Boxes after planner transform ",  boxes)
    # plan
    state = state_to_planner(go1.state, sp)
    start_idx = np.argmin(cdist(np.array(sp.Pset),state))

    # print(start_idx,Pset[start_idx],state)
    res = sp.plan(state, boxes)
    prev_policy = []
    idx_prev = 0
    if not replan:
        plan_traj.append(res)

    # fig, ax = sp.world.show()
    # plt.show()

    # ****************************************************
    # EXECUTION LOOP
    while True:
        # perception + cp
        # boxes = get_boxes(sp)
        # boxes = np.array([[[0,0],[0.01,0.01]]])
        # boxes = np.array([[[2.0,4.0],[3.0,6.0]]])
        # boxes[:,0,:] -= cp
        # boxes[:,1,:] += cp


        if replan:
            # plan
            st = time.time()
            gs, _, yaw = go1.get_state()
            state = state_to_planner(gs, sp)
            start_idx = np.argmin(cdist(np.array(sp.Pset),state))

            # print(start_idx,Pset[start_idx],state)
            boxes = go1.camera.get_boxes(cp, num_detect)
            boxes = boxes[:,:,0:2]
            # print("Boxes before planner transform ",  boxes)
            boxes = boxes_to_planner_frame(boxes, sp)
            # print("Boxes after planner transform ",  boxes)
            res = sp.plan(state, boxes)
            plan_traj.append(res)
            et = time.time()
            t += (et-st)
            if plot_traj:
                plot_trajectories(plan_traj, sp, vicon_traj, state_traj, ground_truth=ground_truth, replan=replan, save_fig=save_traj, filename=result_dir+str(t))

            # sp.show(res[0], true_boxes=ground_truth)
            # plt.show()

        # execute
        if len(res[0]) > 1:
            # print(res[0])
            # print("res 2", res[2])
            # fig, ax = sp.show_connection(res[0])
            # plt.show()
            # fig, ax = sp.show(res[0])
            # plt.show()
            # fig, ax = plot_trajectories(res[0], sp)
            # plt.show()
            policy_before_trans = np.vstack(res[2])
            # print("action shpae", policy_before_trans.shape)
            policy = (np.array([[0,1],[-1,0]])@policy_before_trans.T).T
            prev_policy = policy
            # print("policy: ", policy)
            if replan:
                end_idx = min(int(sp.sensor_dt/sp.dt),len(policy))
            else:
                end_idx = len(policy)
            time_adjust = 0
            for step in range(end_idx):
                st = time.time()
                idx_prev = step
                # print("step: ", step)
                action = policy[step]
                # print("action: ", action)
                go1.move(action)
                # update go1 state 
                if vicon:
                    vicon_state, vicon_yaw, vicon_ts = go1.get_true_state()
                    vicon_traj.append(vicon_state)
                    # print("VICON state: ", vicon_state, " yaw", vicon_yaw)
                if time_adjust==0:
                    state, ts,yaw = go1.get_state()
                    state_traj.append(state)
                    # print("ZED state: ", state, " yaw", yaw)
                et = time.time()
                # print("Time taken ", et-st )
                if (sp.dt-et+st+time_adjust) >0:
                    time.sleep((sp.dt-et+st+time_adjust))
                    t += sp.dt+time_adjust
                    time_adjust =0
                else:
                    time_adjust = time_adjust+sp.dt-et+st
                    t += (et-st)
                # print("t: ", t)
            #state, ts,yaw = go1.get_state()
            #state_traj.append(state)
            # print("state: ", go1.state)
            if go1.check_goal():
                break
        else:
            plan_traj.pop()
            print("BREAK 1: FAILED TO FIND PLAN")
            # for step in range(int(sp.sensor_dt/sp.dt)):
            # if len(prev_policy) > idx_prev+1:
            #     idx_prev += 1
            #     action = policy[step]
            # else:
            action = [0,0]
            go1.move(action)
            time.sleep(sp.dt)
            t += sp.dt
        if t > 40:
            # time safety break
            print("BREAK 2: RAN OUT OF TIME")
            break

    # print("res 2", res[2])
    if save_traj:
        check_dir(result_dir)
        np.save(result_dir + 'plan.npy', res[0]) # initial open loop plan atm
        np.save(result_dir + 'state_traj.npy', state_traj)
        if vicon:
            np.save(result_dir + 'vicon_traj.npy', vicon_traj)
        # TODO: add anything else to save for a trial

    if plot_traj:
        # currently only set up for open loop init plan
        # TODO: handle plotting with replan
       plot_trajectories(plan_traj, sp, vicon_traj, state_traj, ground_truth=ground_truth, replan=replan, save_fig=save_traj, filename=result_dir)

    if vicon:    
        rospy.spin()

if __name__ == '__main__':
    plan_loop()
