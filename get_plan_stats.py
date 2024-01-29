import os
import random
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import math
import IPython as ipy
import time
import torch

num_envs = 80
# cp = 0.02
cp=0.73
# cp=0.61
foldername = "../data/perception-guarantees/rooms_multiple_chairs_with_occlusions/"
filename = "cp_" + str(cp) + "_10Hz.npz"
goal_loc_planner_frame = [6,7]
traj = {}
done= []
coll = []
dist_from_goal = 0

for i in range(num_envs):
    file_env = foldername + str(i) + "/" + filename
    data_ = np.load(file_env, allow_pickle=True)
    traj_info = data_["data"].item()
    # ipy.embed()
    # print(data[tra])
    # print("Environment ", i)
    traj[i] = traj_info['trajectory']
    done.append(int(traj_info['done']))
    coll.append(int(traj_info['collision']==False))

traj_length = 0
for i in range(num_envs):
    if done[i] == 1:
        traj_length+= np.sum(np.linalg.norm(np.array(traj[i][:-1,0,0:2]) - np.array(traj[i][1:, 0, 0:2]), axis=1))
    if done[i] == 0:
        dist_from_goal += np.linalg.norm(np.array(traj[i][-1,0,0:2]-np.array(goal_loc_planner_frame)))-1

print("Average trajectory length: ", traj_length/np.sum(done))
print("Successful task completion: ", np.mean(done))
print("Safety rate: ", np.mean(coll))
print("Failed in environments: ", np.where(np.array(done)<1))
print("Collisions in environments: ", np.where(np.array(coll)<1))
print("Average distance from goal if failed: ", dist_from_goal/(np.sum(1-np.array(done))) )

