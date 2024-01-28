from planning.Safe_Planner import *
import rospy
from geometry_msgs.msg import TransformStamped
from scipy.spatial.transform import Rotation
import numpy as np
from utils.go1_move import *
from utils.plotting import *
import time
import pickle


result_dir = 'results/debug_trial1/'

actions_applied = np.load(result_dir + 'action_applied.npy')
# print(actions_applied)
# print(actions_applied.shape)

with open(result_dir + 'ground_truth_bb.pkl', 'rb') as f:
    ground_truth_bb, chair_yaws = pickle.load(f)

# print(ground_truth_bb)
# print(chair_yaws)

with open(result_dir + 'plan.pkl', 'rb') as f:
    res = pickle.load(f)

# print(res[0])
# print(res[2])

replan_states = np.load(result_dir + 'replan_states.npy')
# print(replan_states)
# print(replan_states.shape)

replan_times = np.load(result_dir + 'replan_times.npy')
print(replan_times)
print(np.mean(replan_times))

with open(result_dir + 'safe_planners.pkl', 'rb') as f:
    SPs = pickle.load(f)

print(SPs[-1])

state_traj = np.load(result_dir + 'state_traj.npy')
# print(state_traj)
# print(state_traj.shape)

vicon_traj = np.load(result_dir + 'vicon_traj.npy')
# print(vicon_traj)
# print(vicon_traj.shape)

replan = True
save_traj = False
plot_trajectories(res, SPs[-1], vicon_traj, state_traj, ground_truth=[ground_truth_bb, chair_yaws], replan=replan, replan_state=replan_states, save_fig=save_traj, filename=result_dir)
plt.show()