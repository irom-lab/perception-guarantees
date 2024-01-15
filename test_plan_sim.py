# %%
import os
import sys
from omegaconf import OmegaConf
import importlib
import json
from mpl_toolkits.mplot3d import Axes3D
# %matplotlib tk

import torch
import torch.nn as nn

from itertools import product, combinations

from planning.Safe_Planner import *
from nav_sim.env.go1_env import Go1Env

from models import build_model
from datasets import build_dataset

# sys.path.append('../utils')

from nav_sim.test.clustering import cluster
from utils.pc_util import preprocess_point_cloud, read_ply, pc_to_axis_aligned_rep, pc_cam_to_3detr, is_inside_camera_fov
from utils.box_util import box2d_iou
from utils.make_args import make_args_parser

from datasets.sunrgbd import SunrgbdDatasetConfig as dataset_config
import time

import trimesh


# %% [markdown]
# ### Sim Code

# %%
# task

# get root repository path
nav_sim_path = '/home/anushri/Documents/Projects/perception-guarantees/nav_sim'

# Initialize task
task = OmegaConf.create()
task.init_state = [5, 0.2, 0.0, 0.0]  # x, y, vx, vy
task.goal_loc = [6, 7]
task.goal_radius = 0.7

# obstacles
ground_truth = []
task.furniture = {}
task.furniture.piece_1 = {
    'path':
        os.path.join(
            nav_sim_path,
            'asset/sample_furniture/00a91a81-fc73-4625-8298-06ecd55b6aaa/raw_model.obj'
        ),
    'position': [6, 4.5, 0.0],
    'yaw': 0
}
piece1 = trimesh.load(
                os.path.join(nav_sim_path,'asset/sample_furniture/00a91a81-fc73-4625-8298-06ecd55b6aaa/raw_model.obj')
            )
piece1.apply_transform([
            [1, 0, 0, 0],
            [0, 0, -1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ])
gt1 = piece1.bounds
ground_truth.append([[task.furniture.piece_1.position[0]+gt1[0,0],  task.furniture.piece_1.position[1]+gt1[0,1]], [task.furniture.piece_1.position[0]+gt1[1,0],  task.furniture.piece_1.position[1]+gt1[1,1]]])
task.furniture.piece_2 = {
    'path':
        os.path.join(
            nav_sim_path,
            'asset/sample_furniture/59e52283-361c-4b98-93e9-0abf42686924/raw_model.obj'
        ),
    'position': [4, 5, 0.0],
    'yaw': -np.pi / 2
}
piece2 = trimesh.load(
                os.path.join(nav_sim_path,'asset/sample_furniture/59e52283-361c-4b98-93e9-0abf42686924/raw_model.obj')
            )
piece2.apply_transform([
            [1, 0, 0, 0],
            [0, 0, -1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
        ])
gt2 = piece2.bounds
ground_truth.append([[task.furniture.piece_2.position[0]+gt2[0,0],  task.furniture.piece_2.position[1]+gt2[0,1]], [task.furniture.piece_2.position[0]+gt2[1,0],  task.furniture.piece_2.position[1]+gt2[1,1]]])
print(ground_truth)
#
task.observation = {}
task.observation.type = 'rgb'  # 'rgb' or 'lidar'
task.observation.rgb = {}
task.observation.depth = {}
task.observation.lidar = {}
task.observation.rgb.x_offset_from_robot_front = 0.05  # no y offset
task.observation.rgb.z_offset_from_robot_top = 0.05
task.observation.rgb.tilt = 0  # degrees of tilting down towards the floor
task.observation.rgb.img_w = 662
task.observation.rgb.img_h = 376
task.observation.rgb.aspect = 1.57
task.observation.rgb.fov = 70  # in PyBullet, this is vertical field of view in degrees
task.observation.depth.img_w = task.observation.rgb.img_w  # needs to be the same now - assume coming from the same camera
task.observation.depth.img_h = task.observation.rgb.img_h
task.observation.lidar.z_offset_from_robot_top = 0.01  # no x/y offset
task.observation.lidar.horizontal_res = 1  # resolution, in degree
task.observation.lidar.vertical_res = 1  # resolution, in degree
task.observation.lidar.vertical_fov = 30  # half in one direction, in degree
task.observation.lidar.max_range = 5  # in meter


# %%
# planner
# load pre-computed
f = open('planning/reachable_10Hz.pkl', 'rb')
reachable = pickle.load(f)
f = open('planning/Pset_10Hz.pkl', 'rb')
Pset = pickle.load(f)

# initialize planner
sp = Safe_Planner(init_state=task.init_state, FoV=60*np.pi/180, n_samples=2000,dt=0.1)
sp.load_reachable(Pset, reachable)

# %%
# camera + 3DETR
num_pc_points = 40000

parser = make_args_parser()
args = parser.parse_args(args=[])

# Dataset config: use SUNRGB-D
dataset_config = dataset_config()
# Build model
model, _ = build_model(args, dataset_config)

# Load pre-trained weights
sd = torch.load(args.test_ckpt, map_location=torch.device("cpu")) 
model.load_state_dict(sd["model"]) 

model = model.cuda()
model.eval()

device = torch.device("cuda")

visualize = False

# Load params from json file
with open("env_params.json", "r") as read_file:
    params = json.load(read_file)


# %%
env = Go1Env(render=True)
env.dt = sp.dt

# %%
env.reset(task)

# %%
def get_box(observation):
    if task.observation.type == 'lidar' or task.observation.type == 'rgb':
        # Filter points with z < 0.01 and abs(y) > 3.5 and x> 0.01 and within a 1m distance of the robot
        # axis transformed, so filter x,y same way
        observation = observation[:, observation[2, :] < 2.9]
        # observation = observation[:, observation[0, :] > 0.05]
        # observation = observation[:, observation[0, :] < 7.95]
        # observation = observation[:, observation[1, :] > 0.05]
        # observation = observation[:, observation[1, :] < 7.95]

        visualize = True

        if (len(observation[0])>0):
            # Preprocess point cloud (random sampling of points), if there are any LIDAR returns
            points_new = np.transpose(np.array(observation))
            points = np.zeros((1,num_pc_points, 3),dtype='float32')
            points = preprocess_point_cloud(np.array(points_new), num_pc_points)
        else:
            # There are no returns from the LIDAR, object is not visible
            points = np.zeros((1,num_pc_points, 3),dtype='float32')
        
        # Convert from camera frame to world frame
        point_clouds = []
        point_clouds.append(points)
    
    batch_size = 1
    pc = np.array(point_clouds).astype('float32')
    pc = pc.reshape((batch_size, num_pc_points, 3))

    pc_all = torch.from_numpy(pc).to(device)
    pc_min_all = pc_all.min(1).values
    pc_max_all = pc_all.max(1).values
    inputs = {'point_clouds': pc_all, 'point_cloud_dims_min': pc_min_all, 'point_cloud_dims_max': pc_max_all}

    # start = tm.time()
    outputs = model(inputs)
    # end = tm.time()
    # print("Time taken for inference: ", end-start)
    
    bbox_pred_points = outputs['outputs']['box_corners'].detach().cpu()
    obj_prob = outputs["outputs"]["objectness_prob"].clone().detach().cpu()
    cls_prob = outputs["outputs"]["sem_cls_prob"].clone().detach().cpu()

    chair_prob = cls_prob[:,:,3]
    sort_box = torch.sort(obj_prob,1,descending=True)

    # Visualize
    if visualize:
        pc_plot = pc[:, pc[0,:,2] > 0.0,:]
        plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter3D(
            pc_plot[0,:,0], pc_plot[0,:,1],pc_plot[0,:,2]
        )
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_aspect('auto')

    num_probs = 0
    num_boxes = 10
    corners = []
    if np.any(np.isnan(np.array(bbox_pred_points))):
            return get_room_size_box(pc_all)
    
    for (sorted_idx,prob) in zip(list(sort_box[1][0,:]), list(sort_box[0][0,:])):
        if (num_probs < num_boxes):
            prob = prob.numpy()
            bbox = bbox_pred_points[range(batch_size), sorted_idx, :, :]
            cc = pc_to_axis_aligned_rep(bbox.numpy())
            flag = False
            if num_probs == 0:
                corners.append(cc)
                num_probs +=1
            else:
                for cc_keep in corners:
                    bb1 = (cc_keep[0,0,0],cc_keep[0,0,1],cc_keep[0,1,0],cc_keep[0,1,1])
                    bb2 = (cc[0,0,0],cc[0,0,1],cc[0,1,0],cc[0,1,1])
                    # Non-maximal supression, check if IoU more than some threshold to keep box
                    if(box2d_iou(bb1,bb2) > 0.1):
                        flag = True
                if not flag:    
                    corners.append(cc)
                    num_probs +=1

            if visualize:
                r0 = [cc[0,0, 0], cc[0,1, 0]]
                r1 = [cc[0,0, 1], cc[0,1, 1]]
                r2 = [cc[0,0, 2], cc[0,1, 2]]

                for s, e in combinations(np.array(list(product(r0, r1, r2))), 2):
                    if (np.sum(np.abs(s-e)) == r0[1]-r0[0] or 
                        np.sum(np.abs(s-e)) == r1[1]-r1[0] or 
                        np.sum(np.abs(s-e)) == r2[1]-r2[0]):
                        if (visualize and not flag):
                            ax.plot3D(*zip(s, e), color=(0.5+0.5*prob, 0.1,0.1))
    
    boxes = np.zeros((len(corners),2,2))
    for i in range(len(corners)):
        boxes[i,:,:] = corners[i][0,:,0:2]

    return boxes

def get_room_size_box(self, pc_all):
    room_size = 8
    num_chairs =5
    boxes = np.zeros((num_chairs, 2,3))
    # box = torch.tensor(pc_to_axis_aligned_rep(bbox_pred_points.numpy()))
    boxes[:,0,1] = 0*np.ones_like(boxes[:,0,0])
    boxes[:,0,0] = (-room_size/2)*np.ones_like(boxes[:,0,1])
    boxes[:,0,2] = 0*np.ones_like(boxes[:,0,2])
    boxes[:,1,1] = room_size*np.ones_like(boxes[:,1,0])
    boxes[:,1,0] = (room_size/2)*np.ones_like(boxes[:,1,1])
    boxes[:,1,2] = room_size*np.ones_like(boxes[:,1,2])

    pc_min_all = pc_all.min(1).values
    pc_max_all = pc_all.max(1).values
    inputs = {'point_clouds': pc_all, 'point_cloud_dims_min': pc_min_all, 'point_cloud_dims_max': pc_max_all}

    outputs = self.model(inputs)
    box_features = outputs["box_features"].detach().cpu()
    return boxes

# %%
t = 0
cp = 0.4
observation = env.step([0,0])[0] # initial observation
steps_taken = 0
while True:
    state = np.array([env._state])
    boxes = get_box(observation)
    print(boxes)
    boxes[:,0,:] -= cp
    boxes[:,1,:] += cp

    res = sp.plan(state, boxes)
    if (steps_taken % 10) == 0 :
        # sp.show_connection(res[0]) 
        sp.world.free_space
        sp.show(res[0], true_boxes=np.array(ground_truth))
    steps_taken+=1
    if len(res[0]) > 1:
        policy = np.vstack(res[2])
        for step in range(min(int(sp.sensor_dt/sp.dt), len(policy))):
            action = policy[step]
            # print("Policy : ", policy)
            # print("Step ", step, " Action ", action, " dt", sp.dt)
            observation, reward, done, info = env.step(action)
            # summarize the step in one line
            # print('\nStep:{}, Action:{}, Done:{}, {}'.format(
            #         step, action, done, info))
            t += sp.dt
            # time.sleep(sp.dt)
            if done:
                break
            # if (step % 10) == 0 :
            #     # sp.show_connection(res[0]) 
            #     sp.world.free_space
            #     sp.show_connection(res[0]) 
            #     sp.show(res[0])
    else:
        for step in range(int(sp.sensor_dt/sp.dt)):
            action = [0,0]
            observation, reward, done, info = env.step(action)
            t += sp.sensor_dt
    if t >100:
        break

# %%
sp.world.isValid([5,0.2])

# %%
sp.world.free_space

# %%



