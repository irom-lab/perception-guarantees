from planning.Safe_Planner import *

# sample states and compute reachability
sp = Safe_Planner(r=5, n_samples = 2000, goal_f = [7,-2,0.5,0], world_box=np.array([[0,0],[8,8]]))
sp.find_all_reachable()

# save pre-computed data
f = open('planning/pre_compute/reachable.pkl', 'ab')
pickle.dump(sp.reachable, f)
f = open('planning/pre_compute/Pset.pkl', 'ab')
pickle.dump(sp.Pset, f)