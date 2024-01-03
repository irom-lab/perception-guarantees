import matplotlib.pyplot as plt

def plot_trajectories(idx_solution, sp): #executed_traj):
    '''Plot solution'''
    fig, ax = sp.world.show()
    for i in range(len(idx_solution)-1):
        s0 = idx_solution[i] #idx
        s1 = idx_solution[i+1] #idx
        # Ginv = self.reachable[s1][2][3][self.reachable[s1][2][0].index(s0)]
        x_waypoints = sp.reachable[s0][1][3][sp.reachable[s0][1][0].index(s1)][0]
        ax.plot(x_waypoints[:,0], x_waypoints[:,1], c='red', linewidth=1)
        plt.show()