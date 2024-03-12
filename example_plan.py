from planning.Safe_Planner import *

# load pre-computed data from file
f = open('planning/pre_compute/Pset.pkl', 'rb')
Pset = pickle.load(f)
f = open('planning/pre_compute/reachable.pkl', 'rb')
reachable = pickle.load(f)

# initialize planner
init_state = [4.8,0.8,0,0]
sp = Safe_Planner(init_state=init_state,radius = 1,n_samples=2000,world_box=np.array([[0,0],[8,8]]), max_search_iter=500)

# load pre-computed data to planner
sp.load_reachable(Pset, reachable)

# example planning
boxes = np.array([[[1,4],[3.5,6]],
                  [[2,3],[2.5,3.5]],
                  [[5.3,2.5],[6,3]]])
state = np.array([init_state])
res = sp.plan(state, boxes)
# res[0] = idx of nodes in the path
# res[1] = [x_trajectory]
# res[2] = [u_trajectory]

sp.show(res[0])