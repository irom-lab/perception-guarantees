import matplotlib.pyplot as plt
import numpy as np
import os

def plot_trajectories(reses, sp, vicon_traj, state_traj, ground_truth=None, replan=False, save_fig=False, filename=None): #executed_traj):
    '''Plot solution'''
    fig, ax = sp.world.show(true_boxes=ground_truth)
    plt.gca().set_aspect('equal', adjustable='box')
    if replan:
        color=['r','y','g','k','m']
        for j in range(len(reses)):
            #idx_solution = reses[j]
            x_waypoints=np.vstack(reses[j][1])
            #for i in range(len(idx_solution)-1):
                #s0 = idx_solution[i] #idx
                #s1 = idx_solution[i+1] #idx
                #Ginv = self.reachable[s1][2][3][self.reachable[s1][2][0].index(s0)]
                #x_waypoints = sp.reachable[s0][1][3][sp.reachable[s0][1][0].index(s1)][0]
            ax.plot(x_waypoints[:,0], x_waypoints[:,1], c=color[j%len(color)], linewidth=1, label='plan')
    else:
        res = reses[0][0]
        for i in range(len(res)-1):
            s0 = res[i] #idx
            s1 = res[i+1] #idx
            # Ginv = self.reachable[s1][2][3][self.reachable[s1][2][0].index(s0)]
            x_waypoints = sp.reachable[s0][1][3][sp.reachable[s0][1][0].index(s1)][0]
            ax.plot(x_waypoints[:,0], x_waypoints[:,1], c='red', linewidth=1, label='plan')
    if len(vicon_traj) > 0:
        # print("vicon length", len(vicon_traj))
        vicon_arr = np.array(vicon_traj)
        vicon_tf = sp.state_to_planner(vicon_arr)
        print('vicon tf', vicon_tf.shape)
        ax.plot(vicon_tf[0, :], vicon_tf[1, :], c='blue', linewidth=1, label='vicon')
    if len(state_traj) >0:
        # state_arr = np.array(state_traj)
        state_tf = np.squeeze(np.array(state_traj)).T

        # state_tf = sp.state_to_planner(state_arr)
        print('state tf', state_tf.shape)
        ax.plot(state_tf[0, :], state_tf[1, :], c='c', linewidth=1, label='state')
        if replan:
            ax.plot(state_tf[0,range(0,len(state_traj),int(sp.sensor_dt/sp.dt))], state_tf[1,range(0,len(state_traj),int(sp.sensor_dt/sp.dt))], 'co',label='replan')
    # TODO: add in state_traj
        
    plt.legend()
    
    if save_fig:
        plt.savefig(filename + 'traj_plot.png')
    
    plt.show()

def check_dir(path):
    # check if directory exists, if not create
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory '{path}' created.")

    else:
        print(f"Directory '{path}' already exists.")