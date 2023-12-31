import pickle
import numpy as np
from planning.Safe_Planner import *

# specify goal in forrestal coordinates
goal_f = [7.0, -2.0, 0.0, 0.0]

sp = Safe_Planner(goal_f=goal_f)

# convert to planner coordinates
print(sp.goal)

# sp.find_all_reachable()

# f = open('planning/precompute/sp_reachable_forrestal.pkl', 'rb')
# reachable = pickle.load(f)
# f = open('planning/precompute/sp_Pset_forrestal.pkl', 'rb')
# Pset = pickle.load(f)