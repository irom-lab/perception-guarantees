from planning.Safe_Planner import Safe_Planner
import numpy as np
import pickle
import matplotlib.pyplot as plt

# sample states and compute reachability
sp = Safe_Planner(r=7, n_samples = 2000, speed = 1.5, goal_f = [7,-2,0.5,0], world_box=np.array([[0,0],[8,8]]))
sp.find_all_reachable()

# save pre-computed data
f = open('planning/pre_compute/reachable_15_7_2K_unfiltered.pkl', 'ab')
pickle.dump(sp.reachable, f)
f = open('planning/pre_compute/Pset_15_7_2K_unfiltered.pkl', 'ab')
pickle.dump(sp.Pset, f)



# load pre-computed data from file
f = open('planning/pre_compute/Pset_15_7_2K_unfiltered.pkl', 'rb')
Pset = pickle.load(f)
f = open('planning/pre_compute/reachable_15_7_2K_unfiltered.pkl', 'rb')
reachable = pickle.load(f)

# initialize planner
init_state = [4.8,0.8,0,0]
sp = Safe_Planner(init_state=init_state,radius = 1,n_samples=len(Pset)-1,world_box=np.array([[0,0],[8,8]]), max_search_iter=500)

# load pre-computed data to planner
sp.load_reachable(Pset, reachable)


boxes = np.array([[[1,4],[3.5,6]],
                  [[2,3],[2.5,3.5]],
                  [[5.3,2.5],[6,3]]])
state = np.array([init_state])
res = sp.plan(state, boxes)
# res[0] = idx of nodes in the path
# res[1] = [x_trajectory]
# res[2] = [u_trajectory]

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

plot_reachable(sp, "F")