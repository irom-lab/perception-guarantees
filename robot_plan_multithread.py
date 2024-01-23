from planning.Safe_Planner import *
import rospy
from geometry_msgs.msg import TransformStamped
from scipy.spatial.transform import Rotation
import numpy as np
from utils.go1_move import *
from utils.plotting import *
import time
import multiprocessing
import threading
from queue import Queue
import concurrent.futures
from numpy.linalg import inv, pinv

replan_finished = threading.Event()
plan_queue = Queue()

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

def plan_generator(sp, state, boxes):
    print("planning in parallel")

    res = sp.plan(state, boxes)
    # plan_queue.put(res)
    # replan_finished.set()
    return res

def plan_executor(sp, go1, policy, t, vicon=True):
    time_adjust=0
    state_traj = []
    vicon_traj=[]
    for action in policy:
        st = time.time()
        go1.move(action)
        # update go1 state 
        if vicon:
            vicon_state, vicon_yaw, vicon_ts = go1.get_true_state()
            vicon_traj.append(vicon_state)
            # print("VICON state: ", vicon_state, " yaw", vicon_yaw)
        if time_adjust==0:
            gs, _, yaw = go1.get_state()
            state = state_to_planner(gs, sp)
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
    return state_traj, vicon_traj, state

def pred_state_planner(state_, actions,Ad, Bd):
    state = state_.T
    new_state = np.copy(state)
    print("Current state", state, state.shape)
    print("Ad", Ad.shape, "Bd", Bd.shape)
    for action_ in actions:
        action_ = action_[:, np.newaxis]
        action = np.copy(action_)
        action[0] = -action_[1]
        action[1] = action_[0]
        print("action", action.shape)
        print("Ad@state", (Ad@state).shape)
        print("Ad.dot(state)", (np.dot(Ad, state)).shape)
        print("Bd@action", (Bd@action).shape)
        new_state = (Ad@state)+(Bd@action)
        state = np.copy(new_state)
        print("New state prediction", new_state, new_state.shape)
    return (new_state.T)

def pred_state(state_, actions,Ad, Bd):
    state = state_[:, np.newaxis]
    new_state = np.copy(state)
    print("Current state", state, state.shape)
    print("Ad", Ad.shape, "Bd", Bd.shape)
    for action_ in actions:
        action_ = action_[:, np.newaxis]
        action = np.copy(action_)
        # action[0] = -action_[1]
        # action[1] = action_[0]
        print("action", action.shape)
        print("Ad@state", (Ad@state).shape)
        print("Ad.dot(state)", (np.dot(Ad, state)).shape)
        print("Bd@action", (Bd@action).shape)
        new_state = (Ad@state)+(Bd@action)
        state = np.copy(new_state)
        print("New state prediction", new_state, new_state.shape)
    new_state_list = (new_state.T).tolist()
    print("list", new_state_list, new_state_list[0])
    return new_state_list[0]

def plan_loop():
    # ****************************************************
    # SET DESIRED FLAGS
    vicon = True # set if want vicon state active; If true, make sure ./vicon.sh is running from pg_ws (and zed box connected to gas dynamics network)
    state_type =  'zed' #'vicon' # if want to set the state used as the vicon state, 'zed' 
    replan = True # set if want to just follow open loop plan
    save_traj = True  # set if want to save trajectory and compare against plan
    plot_traj = True  # set if want to visualize trajectory at the end of execution
    result_dir = 'results/debug_trial2/' # set to unique trial identifier if saving results
    goal_forrestal = [7.0, -2.0, 0.0, 0.0] # goal in forrestal coordinates
    reachable_file = 'planning/pre_compute/reachable_10Hz.pkl'
    pset_file = 'planning/pre_compute/Pset_10Hz.pkl'
    num_samples = 2000  # number of samples used for the precomputed files
    dt = 0.1 #   planner dt
    radius = 0.5 # distance between intermediate goals on the frontier
    chairs = [2, 3]  # list of chair labels to be used to get ground truth bounding boxes
    num_detect = 10  # number of boxes for 3DETR to detect
    cp = 1.19
    sensor_dt = 1 # time in seconds to replan
    replan_start = 9 # index to start replanning at
    # ****************************************************
    if vicon:
        rospy.init_node('listener', anonymous=True)
        chair_states = GroundTruthBB(chairs)
        time.sleep(3)

    if save_traj:
        check_dir(result_dir)   
 
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

    # Dynamics
    # k1=3.968; k2=2.517; k3=0.1353; k4=-0.5197; k5 = 4.651; k6 = 2.335
    # A = np.array([[0,0,1,0],[0,0,0,1],[0,0,-k2,k3],[0,0,k4,-k1]])
    # Ad = np.expm1(A*sp.dt)
    # # Ad = np.identity(4) + A*
    # B = np.array([[0,0],[0,0],[k6,0],[0,k5]])
    # Bd =pinv(A)@(Ad-np.identity(4))@B

    k1=3.968; k2=2.517; k3=0.1353; k4=-0.5197; k5 = 4.651; k6 = 2.335
    A = np.array([[0,0,1,0],[0,0,0,1],[0,0,-k1,-k4],[0,0,-k3,-k2]])
    Ad = np.expm1(A*sp.dt)
    # Ad = np.identity(4) + A*
    B = np.array([[0,0],[0,0],[k5,0],[0,k6]])
    Bd =pinv(A)@(Ad-np.identity(4))@B
    
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
    # state = state_to_planner(go1.state, sp)
    # print("SHAPPPPPE, ", state.shape)

    # print(start_idx,Pset[start_idx],state)
    gs, _, yaw = go1.get_state()
    state = state_to_planner(gs, sp)
    print("SHAPPPPPE, ", state.shape)
    start_idx = np.argmin(cdist(np.array(sp.Pset),state))
    res = sp.plan(state, boxes)
    prev_policy = []
    idx_prev = 0
    # if not replan:
    plan_traj.append(res)


    # set up replan multiprocessing
    # plan_gen_process = multiprocessing.Process(target=plan_generator, args=(sp, state, boxes))
    

   
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

        # if replan:
        #     replan_finished.clear()
        #     replan_timer = threading.Timer(0.8, plan_generator, args=(sp, state, boxes))
        #     replan_timer.start()

        # if replan_finished.is_set():
        #     res = plan_queue.get()

        # add in if time % 0.8: # so we aren't creating all sorts of new threads
        # with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        #     # updated boxes
        #     boxes = go1.camera.get_boxes(cp, num_detect)
        #     boxes = boxes[:,:,0:2]
        #     # print("Boxes before planner transform ",  boxes)
        #     boxes = boxes_to_planner_frame(boxes, sp)
        #     args = (sp, state, boxes)
        #     future_plan = executor.submit(plan_generator, *args)

        # if future_plan.done(): # this needs to be fixed..error returned on second iteration; 
            #   trying to just update available plan if its been calculated from parallel thread running
        #     res = future_plan.result()

        # if replan:
        #     # plan
        #     st = time.time()
        #     gs, _, yaw = go1.get_state()
        #     state = state_to_planner(gs, sp)
        #     start_idx = np.argmin(cdist(np.array(sp.Pset),state))

        #     # print(start_idx,Pset[start_idx],state)
        #     boxes = go1.camera.get_boxes(cp, num_detect)
        #     boxes = boxes[:,:,0:2]
        #     # print("Boxes before planner transform ",  boxes)
        #     boxes = boxes_to_planner_frame(boxes, sp)
        #     # print("Boxes after planner transform ",  boxes)

        #     res = sp.plan(state, boxes)
        #     plan_traj.append(res)
        #     et = time.time()
        #     t += (et-st)
        #     if plot_traj and len(res[0]) > 1:
        #         t_str = str(round(t, 1))
        #         plot_trajectories(plan_traj, sp, vicon_traj, state_traj, ground_truth=ground_truth, replan=replan, save_fig=save_traj, filename=result_dir+t_str)

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
            for step in range(replan_start):
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
                    # state, ts,yaw = go1.get_state()
                    gs, _, yaw = go1.get_state()
                    state = state_to_planner(gs, sp)
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

            st = time.time()
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                policy_to_exec= policy[replan_start:end_idx]
                args_exec = (sp,go1, policy_to_exec,t, vicon)
                traj = executor.submit(plan_executor, *args_exec)
                s_hist, v_hist, state = traj.result()
                # updated boxes
                gs, _, yaw = go1.get_state()
                state = state_to_planner(gs, sp)
                boxes = go1.camera.get_boxes(cp, num_detect)
                boxes = boxes[:,:,0:2]
                # print("Boxes before planner transform ",  boxes)
                boxes = boxes_to_planner_frame(boxes, sp)
                print("state", state)
                state_go1 = state_to_go1(state[0], sp)
                new_state_go1 = pred_state(state_go1, policy_to_exec,Ad, Bd)
                # print("after pred state", new_state_go1, new_state_go1.shape)
                new_state = state_to_planner(new_state_go1, sp)
                print("after state to plabber", new_state, new_state.shape)
                # new_state = state_to_planner(ns, sp)
                args_replan = (sp, new_state, boxes)
                p = executor.submit(plan_generator, *args_replan)
                res = p.result()
                plan_traj.append(res)
                for (zs, vs) in zip(s_hist, v_hist):
                    state_traj.append(zs)
                    vicon_traj.append(vs)
                if plot_traj and len(res[0]) > 1:
                    t_str = str(round(t, 1))
                    plot_trajectories(plan_traj, sp, vicon_traj, state_traj, ground_truth=ground_truth, replan=replan, save_fig=save_traj, filename=result_dir+t_str)
            et = time.time()
            t += (et-st)
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
            action =[[0,0]]
            # go1.move(action)
            # time.sleep(sp.dt)
            # t += sp.dt
            st = time.time()
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                args_exec = (sp, go1, action, t, vicon)
                traj = executor.submit(plan_executor, *args_exec)
                s_hist, v_hist, state = traj.result()
                # updated boxes
                boxes = go1.camera.get_boxes(cp, num_detect)
                boxes = boxes[:,:,0:2]
                # print("Boxes before planner transform ",  boxes)
                state_go1 = state_to_go1(state[0], sp)
                new_state_go1 = pred_state(state_go1, policy_to_exec,Ad, Bd)
                new_state = state_to_planner(new_state_go1, sp)
                # print("after pred state", new_state_go1, new_state_go1.shape)
                boxes = boxes_to_planner_frame(boxes, sp)
                # new_state = state_to_planner(ns, sp)
                args_replan = (sp, new_state, boxes)
                p = executor.submit(plan_generator, *args_replan)
                res = p.result()
                plan_traj.append(res)
            et = time.time()
            t += (et-st)
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