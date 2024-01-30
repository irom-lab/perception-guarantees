"""
Test the navigation simulation with the task dataset.

Please contact the author(s) of this library if you have any questions.
Authors: Allen Z. Ren (allen.ren@princeton.edu)
"""

import os
import random
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
# from multiprocessing import Pool
import math
import json

from itertools import product, combinations
import torch
from torch.multiprocessing import Pool, Process, set_start_method
try:
    set_start_method('spawn')
except RuntimeError:
    pass

from clustering import is_box_visible
from nav_sim.env.task_env import TaskEnv
import sys
sys.path.append('../utils')
sys.path.append('../datasets')
from utils.pc_util import preprocess_point_cloud, pc_cam_to_3detr, random_sampling
import warnings
warnings.filterwarnings("error")
import IPython as ipy
from models import build_model
from datasets.sunrgbd import SunrgbdDatasetConfig as dataset_config

from nav_sim.test.clustering import cluster
from utils.pc_util import preprocess_point_cloud, read_ply, pc_to_axis_aligned_rep, pc_cam_to_3detr, is_inside_camera_fov
from utils.box_util import box2d_iou
from utils.make_args import make_args_parser

np.random.seed(44)
torch.manual_seed(44)

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
# device = torch.device("cpu")
model.to(device)

# Load the x,y points to sample
with open('planning/Pset_10Hz.pkl', 'rb') as f:
    samples = pickle.load(f)
    # Remove goal
    samples = samples[:-1][:]
# Remove duplicates
sample_proj = [[sample[0], sample[1]] for sample in samples]
s = []
s = [x for x in sample_proj if x not in s and not s.append(x)]
# Transform from planner frame
x = [sample[1] for sample in s]
y = [sample[0]-4 for sample in s]

num_steps = len(x)
# print("Num steps: ", num_steps)

# Load params from json file
with open("env_params.json", "r") as read_file:
    params = json.load(read_file)

def run_env(task):
    visualize = True
    verbose = 0
    init_state = [0,-3.5,0]
    goal_loc = [7.5,3.5]
    task.init_state = [float(v) for v in init_state]
    task.goal_loc = [float(v) for v in goal_loc]
    env = TaskEnv(render=visualize)
    env.reset(task)
    camera_pos = []
    cam_not_inside_obs = []
    is_visible = []
    # point_clouds = []
    bbs = []
    outputs = []
    matched_gts =[]
    # Change seed if you want
    # np.random.default_rng(12345)

    # num_steps = 1000 # 500 #1000
    # x = np.random.uniform(0.05,7.95,num_steps)
    # y = np.random.uniform(-3.95,3.95,num_steps)

    # Press any key to start
    # print("\n=========================================")
    # input("Press any key to start")
    # print("=========================================\n")

    # Run
    # for step in range(16*4):
    gt_obs = [[[-obs[4], obs[0], obs[2]],[-obs[1], obs[3], obs[5]]] for obs in task.piece_bounds_all]
    for step in range(num_steps):
        # print(step)
        # Execute action
        task, bb, box_features, output, matched_gt = run_step(env, task, x, y, step, visualize)
        camera_pos.append(task.observation.camera_pos[step])
        cam_not_inside_obs.append(task.observation.cam_not_inside_obs[step])
        bbs.append(bb)
        outputs.append(output)
        matched_gts.append(matched_gt)
        # Append box features
        if step == 0:
            all_box_features = torch.clone(box_features)
            # print(all_box_features.shape)
        else:
            all_box_features = torch.cat((all_box_features, box_features), 0)
            # print(all_box_features.shape)

        # Convert from camera frame to world frame
        is_visible.append(task.observation.is_visible[step])

    return {"box_axis_aligned": np.array(bbs), 
            "box_features": all_box_features,
            "bbox_labels": np.array(gt_obs), 
            "loss": np.array(is_visible),
            "box_finetune": np.array(outputs),
            "matched_gt": np.array(matched_gts)}
            

def run_step(env, task, x, y, step, visualize=False):
    action  = [x[step],y[step],0]
    # t_s = time.time()
    observation, reward, done, info = env.step(action)
    # t_e_1 = time.time()
    # print("Time to take step and get observation: ", t_e_1 - t_s)
    num_chairs = len(task.piece_bounds_all) # Number of chairs in the current environment
    num_boxes = 15 # Number of boxes we want to predict using 3DETR

    task.observation.camera_pos[step] = [float(env.lidar_pos[0]), float(env.lidar_pos[1]), float(env.lidar_pos[2])]
    pos = task.observation.camera_pos[step]
    not_inside_xyz = [[(0 if (pos[i]>obs[i]-0.1 and pos[i]<obs[3+i]+0.1) else 1) for i in range(3)] for obs in task.piece_bounds_all]
    gt_obs = [[[-obs[4], obs[0], obs[2]],[-obs[1], obs[3], obs[5]]] for obs in task.piece_bounds_all]
    task.observation.cam_not_inside_obs[step] = all([True if any(obs) == 1 else False for obs in not_inside_xyz])
    is_vis = [False]*num_chairs
    
    # Filter points with z < 0.01 and abs(y) > 3.5 and x> 0.01 and within a 1m distance of the robot
    observation = observation[:, observation[2, :] < 2.9]
    # observation = observation[:, np.abs(observation[1, :]) < 3.9]
    # observation = observation[:, observation[0, :] > 0.05]
    # observation = observation[:, observation[0, :] < 7.95]

    X = observation[:, (observation[2, :] >0.1)]
    X = X[:, np.abs(X[1,:]) < 3.9]
    X = X[:, X[0,:]>0.05]
    X = X[:, X[0,:] < 7.95]
    # X = observation[:, ((observation[0,:]-pos[0])**2 + (observation[1,:]-pos[1])**2)**0.5 > 1]
    # X = X[:, ((X[0,:]-pos[0])**2 + (X[1,:]-pos[1])**2)**0.5 < 8]
    
    X = np.transpose(np.array(X))
    if(len(X) > 0):
        # print(len(X))
        # X = random_sampling(X, 5000)
    #     cluster_centers = cluster(X, visualize)
    #     for obs_idx, obs in enumerate(task.piece_bounds_all):
    #         # even "just outside" the box is acceptable as a cluster center, we'd rather be more conservative (-+)
    #         is_vis[obs_idx] = (any([all([cc[i]>obs[i]+0.2 and cc[i]<obs[3+i]-0.2 for i in range(2)]) for cc in cluster_centers]) and task.observation.cam_not_inside_obs[step])
        is_vis = is_box_visible(X, task.piece_bounds_all, visualize)
        for obs_idx, obs in enumerate(task.piece_bounds_all):
            if obs_idx < task.num_objects:
                is_vis[obs_idx] = (is_vis[obs_idx] and task.observation.cam_not_inside_obs[step])
                # print("obs idx: (might be visible) ", obs_idx, is_vis[obs_idx])
            else:
                is_vis[obs_idx] = False
                # print("obs idx: ", obs_idx, "False")
    task.observation.is_visible[step] = is_vis
    # print("Is visible: ", is_vis)

    if (len(observation[0])>0):
        # Preprocess point cloud (random sampling of points), if there are any LIDAR returns
        points_ = np.zeros_like(np.transpose(np.array(observation)).shape)
        points_ = pc_cam_to_3detr(np.transpose(np.array(observation)))
        points = np.zeros((1,num_pc_points, 3),dtype='float32')
        points = preprocess_point_cloud(np.array(points_), num_pc_points)
        # print(points.shape)
        # Get bounding boxes
        pc = np.array(points).astype('float32')
        pc = pc.reshape((1, num_pc_points, 3))
        pc_all = torch.from_numpy(pc).to(device)
        # t_s = time.time()
        output, box_features = get_box(pc_all, num_chairs, num_boxes)
        # t_e_1 = time.time()
        bb = match_gt_output_boxes(output, np.array(gt_obs), is_vis)
        matched_gt = match_output_gt_boxes(output, np.array(gt_obs), is_vis) 
        # t_e_2 = time.time()
        # print("Time to generate boxes ", t_e_1 - t_s)
        # print("Time to match boxes to ground truth ", t_e_2 - t_e_1)
        bbnp = np.array(bb)
        gtnp = np.array(gt_obs)
        visnp = np.array(is_vis)
        if (any(visnp) and ((bbnp[visnp!=0,0,:]-gtnp[visnp!=0,0,:]>1).any() or (gtnp[visnp!=0,1,:]-bbnp[visnp!=0,1,:]>1).any())):
            print(step, len(observation[0]))
            # plot_box_pc(pc, bbnp, gtnp, is_vis)
            # plot_box_pc(pc, np.array(output), gtnp, is_vis)
    else:
        # There are no returns from the LIDAR, object is not visible
        # print("Trying again... ")
        points = np.zeros((1,num_pc_points, 3),dtype='float32')
        task.observation.is_visible[step] = [False]*num_chairs
        is_vis = [False]*num_chairs
        # print(points.shape)
        pc_all = torch.from_numpy(points).to(device)
        bb, _ = get_room_size_box(pc_all, num_chairs)
        output, box_features = get_room_size_box(pc_all, num_boxes)
        matched_gt = match_output_gt_boxes(output, np.array(gt_obs), is_vis) 
        # observation = tuple(map(tuple, observation))
        # observation = [[float(observation[i][j]) for j in range(len(observation[i]))] for i in range(3)]
        # task.observation.lidar[step] = observation
    return task, bb, box_features, output, matched_gt

def get_room_size_box(pc_all, num_boxes):
    room_size = 8
    # num_chairs =5
    boxes = np.zeros((num_boxes, 2,3))
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

    outputs = model(inputs)
    box_features = torch.zeros_like(outputs["box_features"]).detach().cpu()
    return boxes, box_features
    
def get_box(pc_all, num_chairs, num_boxes):

    pc_min_all = pc_all.min(1).values
    pc_max_all = pc_all.max(1).values
    inputs = {'point_clouds': pc_all, 'point_cloud_dims_min': pc_min_all, 'point_cloud_dims_max': pc_max_all}

    outputs = model(inputs)
    bbox_pred_points = outputs['outputs']['box_corners'].detach().cpu()
    # print(bbox_pred_points.shape)
    cls_prob = outputs["outputs"]["sem_cls_prob"].clone().detach().cpu()

    # model_outputs_all["box_corners"][env,batch_inds,:,:] = outputs["outputs"]["box_corners"].detach().cpu()
    # model_outputs_all["box_features"][env,batch_inds,:,:] = outputs["box_features"].detach().cpu()

    box_features = outputs["box_features"].detach().cpu()

    chair_prob = cls_prob[:,:,3]
    obj_prob = outputs["outputs"]["objectness_prob"].clone().detach().cpu()
    sort_box = torch.sort(obj_prob,1,descending=True)

    num_probs = 0
    # num_boxes = 15
    corners = np.zeros((num_boxes, 2,3))
    if np.any(np.isnan(np.array(bbox_pred_points))):
        return get_room_size_box(pc_all, num_boxes)
    
    for (sorted_idx,prob) in zip(list(sort_box[1][0,:]), list(sort_box[0][0,:])):
        if (num_probs < num_boxes):
            prob = prob.numpy()
            bbox = bbox_pred_points[0, sorted_idx, :, :]
            cc = pc_to_axis_aligned_rep(bbox.numpy())
            # print(cc)
            flag = False
            if num_probs == 0:
                corners[num_probs,:,:] = cc
                num_probs +=1
            else:
                for cc_keep in corners:
                    bb1 = (cc_keep[0,0],cc_keep[0,1],cc_keep[1,0],cc_keep[1,1])
                    bb2 = (cc[0,0],cc[0,1],cc[1,0],cc[1,1])
                    # Non-maximal supression, check if IoU more than some threshold to keep box
                    if(box2d_iou(bb1,bb2) > 0.1):
                        flag = True
                if not flag:    
                    corners[num_probs,:,:] = cc
                    num_probs +=1
    return corners, box_features

def match_gt_output_boxes(output_boxes, ground_truth, is_visible):
    max_iou = torch.zeros(ground_truth.shape[0])
    center_diff = 100*torch.ones(ground_truth.shape[0])
    sorted_pred = np.copy(ground_truth)
    for j, val in enumerate(is_visible):
        if val:
            gt = (ground_truth[j,0,0], ground_truth[j,0,1], ground_truth[j,1,0], ground_truth[j,1,1])
            for kk in range(output_boxes.shape[0]):
                pred_ = output_boxes[kk,:,:]
                pred = (pred_[0,0], pred_[0,1], pred_[1,0], pred_[1,1])
                iou = box2d_iou(pred, gt)
                diff = ((((gt[2]+gt[0]-pred[2]-pred[0])**2) + (gt[3]+gt[1]-pred[3]-pred[1])**2)**0.5)/2
                # print("Prediction: ", pred_)
                if iou > max_iou[j]:
                    # print("IoU: ", iou, " Max IoU: ", max_iou[j])
                    max_iou[j] = iou
                    sorted_pred[j,:,:] = pred_
                    center_diff[j] = diff
                elif iou == 0 and max_iou[j] == 0 and (center_diff[j] > diff):
                    # Centers of the predicted box are closer than before
                    # print("Center diff: ", diff, " Diff so far: ", center_diff[j])
                    center_diff[j] = diff
                    sorted_pred[j,:,:] = pred_
    return sorted_pred

def match_output_gt_boxes(output_boxes, ground_truth, is_visible):
    max_iou = torch.zeros(output_boxes.shape[0])
    center_diff = 100*torch.ones(output_boxes.shape[0])
    gt_matched_outputs = match_gt_output_boxes(output_boxes, ground_truth, is_visible)
    sorted_pred = np.copy(output_boxes)
    sorted_pred[:,0,:] = np.mean(output_boxes, axis=1)-0.01
    sorted_pred[:,1,:] = np.mean(output_boxes, axis=1)+0.01
    for kk in range(output_boxes.shape[0]):
        pred_ = output_boxes[kk,:,:]
        pred = (pred_[0,0], pred_[0,1], pred_[1,0], pred_[1,1])
        for j, val in enumerate(is_visible):
            if val and np.all(gt_matched_outputs[j,:,:] == output_boxes[kk,:,:]):
                sorted_pred[kk,:,:] = ground_truth[j,:,:]
                # ipy.embed()
            # if val:
                # gt = (ground_truth[j,0,0], ground_truth[j,0,1], ground_truth[j,1,0], ground_truth[j,1,1])
                # iou = box2d_iou(pred, gt)
                # diff = ((((gt[2]+gt[0]-pred[2]-pred[0])**2) + (gt[3]+gt[1]-pred[3]-pred[1])**2)**0.5)/2
                # # print("Prediction: ", pred_)
                # if iou > max_iou[kk]:
                #     # print("IoU: ", iou, " Max IoU: ", max_iou[j])
                #     max_iou[kk] = iou
                #     sorted_pred[kk,:,:] = ground_truth[j,:,:]
                #     center_diff[kk] = diff
                # elif iou == 0 and max_iou[kk] == 0 and (center_diff[kk] > diff):
                #     # Centers of the predicted box are closer than before
                #     # print("Center diff: ", diff, " Diff so far: ", center_diff[j])
                #     center_diff[kk] = diff
                #     sorted_pred[kk,:,:] = ground_truth[j,:,:]
    return sorted_pred

def plot_box_pc(pc, output_boxes, gt_boxes, is_vis):
    # Visualize
    pc_plot = pc[:, pc[0,:,2] > 0.0,:]
    pc_plot = pc_plot[:, pc_plot[0,:,1]<7.95,:]
    pc_plot = pc_plot[:, pc_plot[0,:,1]>0.05,:]
    pc_plot = pc_plot[:, abs(pc_plot[0,:,0]) < 3.9,:]
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(
        pc_plot[0,:,0], pc_plot[0,:,1],pc_plot[0,:,2]
    )
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_aspect('equal')

    for jj, cc in enumerate(output_boxes):
        # if is_vis[jj]:
        r0 = [cc[0, 0], cc[1, 0]]
        r1 = [cc[0, 1], cc[1, 1]]
        r2 = [cc[0, 2], cc[1, 2]]

        for s, e in combinations(np.array(list(product(r0, r1, r2))), 2):
            if (np.sum(np.abs(s-e)) == r0[1]-r0[0] or 
                np.sum(np.abs(s-e)) == r1[1]-r1[0] or 
                np.sum(np.abs(s-e)) == r2[1]-r2[0]):
                ax.plot3D(*zip(s, e), color=(0.5, 0.1,0.1))

    for jj, cc in enumerate(gt_boxes):
        # if is_vis[jj]:
        r0 = [cc[0, 0], cc[1, 0]]
        r1 = [cc[0, 1], cc[1, 1]]
        r2 = [cc[0, 2], cc[1, 2]]

        for s, e in combinations(np.array(list(product(r0, r1, r2))), 2):
            if (np.sum(np.abs(s-e)) == r0[1]-r0[0] or 
                np.sum(np.abs(s-e)) == r1[1]-r1[0] or 
                np.sum(np.abs(s-e)) == r2[1]-r2[0]):
                ax.plot3D(*zip(s, e), color=(0.1, 0.5,0.1))
    plt.show()

def format_results(results):
    num_envs = len(results)
    num_cam_positions = results[0]["loss"].shape[0]
    num_chairs = results[0]["loss"].shape[1]
    nqueries = results[0]["box_features"].shape[1]
    dec_dim = results[0]["box_features"].shape[2]
    num_pred = results[0]["box_finetune"].shape[1]
    print("Number of predicted boxes for finetuning: ", num_pred)
    loss_mask = torch.zeros(num_envs, num_cam_positions, num_chairs)
    model_outputs_all = {
        "box_features": torch.zeros(num_envs, num_cam_positions, nqueries, dec_dim),
        "box_axis_aligned": torch.zeros(num_envs, num_cam_positions, num_chairs, 2, 3),
    }
    match_outputs_gt = {
        "output": torch.zeros(num_envs, num_cam_positions, num_pred, 2, 3),
        "gt": torch.zeros(num_envs, num_cam_positions, num_pred, 2, 3),
    }
    bboxes_ground_truth_aligned = torch.zeros(num_envs, num_chairs, 2,3)
    for env, result in enumerate(results):
        loss_mask[env,:,:] = torch.from_numpy(results[env]["loss"]).float()
        model_outputs_all["box_features"][env,:,:,:] = results[env]["box_features"]
        model_outputs_all["box_axis_aligned"][env,:,:,:] = torch.from_numpy(results[env]["box_axis_aligned"])
        bboxes_ground_truth_aligned[env,:,:,:] = torch.from_numpy(results[env]["bbox_labels"])
        match_outputs_gt["output"][env,:,:,:] = torch.from_numpy(results[env]["box_finetune"])
        match_outputs_gt["gt"][env,:,:,:] = torch.from_numpy(results[env]["matched_gt"])
    return model_outputs_all, bboxes_ground_truth_aligned, loss_mask, match_outputs_gt

def combine_old_files(filenames, num_files, num_envs_per_file):
    model_outputs_all = {"box_features": torch.tensor([]), "box_axis_aligned": torch.tensor([])}
    match_outputs_gt = {"output": torch.tensor([]), "gt": torch.tensor([])}
    bboxes_ground_truth_aligned = torch.tensor([])
    loss_mask = torch.tensor([])
    for i in range(num_files):
        features = torch.load(filenames[0]+str(i+1) + ".pt")
        bboxes = torch.load(filenames[1]+str(i+1) + ".pt")
        loss = torch.load(filenames[2]+str(i+1) + ".pt")
        finetune = torch.load(filenames[3]+str(i+1) + ".pt")
        model_outputs_all["box_features"] = torch.cat((model_outputs_all["box_features"], features["box_features"]))
        model_outputs_all["box_axis_aligned"] = torch.cat((model_outputs_all["box_axis_aligned"], features["box_axis_aligned"]))
        match_outputs_gt["output"] = torch.cat((match_outputs_gt["output"], finetune["output"]))
        match_outputs_gt["gt"] = torch.cat((match_outputs_gt["gt"], finetune["gt"]))
        bboxes_ground_truth_aligned= torch.cat((bboxes_ground_truth_aligned, bboxes))
        loss_mask = torch.cat((loss_mask, loss))
    return model_outputs_all, bboxes_ground_truth_aligned, loss_mask, match_outputs_gt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--task_dataset', default='/home/anushri/Documents/Projects/data/perception-guarantees/task.pkl',
        nargs='?', help='path to save the task files'
    )
    parser.add_argument(
        '--save_dataset', default='/home/anushri/Documents/Projects/data/perception-guarantees/task.npz',
        nargs='?', help='path to save the task files'
    )
    args = parser.parse_args()

    # Load task dataset
    with open(args.task_dataset, 'rb') as f:
        task_dataset = pickle.load(f)

    # get root repository path
    nav_sim_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


    # Sample random task
    save_tasks = []
    # ipy.embed()
    for task in task_dataset:

        # task = random.choice(task_dataset)

        # Initialize task
        task.goal_radius = 0.5
        #
        task.observation = {}
        task.observation.type = 'rgb'  # 'rgb' or 'lidar'
        task.observation.rgb = {}
        task.observation.depth = {}
        task.observation.lidar = {}
        task.observation.camera_pos = {}
        task.observation.cam_not_inside_obs = {}
        task.observation.is_visible = {}
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
        task.observation.lidar.horizontal_res = 1  # resolution, in degree,1
        task.observation.lidar.vertical_res = 1  # resolution, in degree , 1
        task.observation.lidar.vertical_fov = 30  # half in one direction, in degree
        task.observation.lidar.max_range = 5 # in meter Anushri changed from 5 to 8
        object_phantom = task.piece_bounds_all[0]
        task.num_objects = len(task.piece_bounds_all)
        for i in range(5-len(task.piece_bounds_all)):
            task.piece_bounds_all.append(object_phantom)

        # Run environment
        # run_env(task)
        # save_tasks += [task]
        # print(len(save_tasks))

    ##################################################################
    # Number of environments
    num_envs = 100

    # Number of parallel threads
    num_parallel = 1
    ##################################################################

    # _, _, _ = render_env(seed=0)

    ##################################################################
    env = 0
    batch_size = num_parallel
    save_file = args.save_dataset
    save_res = []
    ##################################################################

    for task in task_dataset:
        # save_tasks += [task]
        env += 1 
        if env%batch_size == 0:
            if env==391: #410: # In case code stops running, change starting environment to last batch saved

                batch = math.floor(env/batch_size)
                print("Saving batch", str(batch))
                t_start = time.time()
                pool = Pool(num_parallel) # Number of parallel processes
                # seeds = range(batch) # Seeds to use for the different processes
                # print(task.piece_id_all)
                results = pool.map_async(run_env, task_dataset[env-batch_size:env]) # Compute results
                pool.close()
                pool.join()
                results = results.get()
                save_res = save_res + results
                # run_env(task)
                t_end = time.time()
                print("Time to generate results: ", t_end - t_start)
                ##########################################################################
                model_outputs_all, bboxes_ground_truth_aligned, loss_mask, match_outputs_gt = format_results(results)
                torch.save(model_outputs_all, "data/dataset_intermediate/features15_cal_variable_chairs"+str(batch) + ".pt")
                torch.save(bboxes_ground_truth_aligned, "data/dataset_intermediate/bbox_labels15_cal_variable_chairs"+str(batch) + ".pt")
                torch.save(loss_mask, "data/dataset_intermediate/loss_mask15_cal_variable_chairs"+str(batch) + ".pt")
                torch.save(match_outputs_gt, "data/dataset_intermediate/finetune15_cal_variable_chairs"+str(batch) + ".pt")

            #     ii = 0
            #     for result in results.get():
            #         # Save data
            #         # file_batch = save_file[:-4] + str(batch) + ".npz"
            #         file_batch = save_file[:-4] + str(env+ii) + ".npz"
            #         print(file_batch)
            #         np.savez_compressed(file_batch, data=result)
            #         ii=+1
            #         print("Time to generate results: ", t_end - t_start)
            # save_tasks = []
    #################################################################

    # model_outputs_all, bboxes_ground_treuth_aligned, loss_mask, match_outputs_gt = format_results(save_res)
    filenames = ["data/dataset_intermediate/features15_cal_variable_chairs", "data/dataset_intermediate/bbox_labels15_cal_variable_chairs", "data/dataset_intermediate/loss_mask15_cal_variable_chairs", "data/dataset_intermediate/finetune15_cal_variable_chairs"]
    model_outputs_all, bboxes_ground_truth_aligned, loss_mask, match_outputs_gt = combine_old_files(filenames, 40, 10)
    ###########################################################################
    # # Save processed feature data
    torch.save(model_outputs_all, "data/features15_cal_variable_chairs.pt")
    # # Save ground truth bounding boxes
    torch.save(bboxes_ground_truth_aligned, "data/bbox_labels15_cal_variable_chairs.pt")
    # # Save loss mask
    torch.save(loss_mask, "data/loss_mask15_cal_variable_chairs.pt")
    # # Save all box outputs for finetuning
    torch.save(match_outputs_gt, "data/finetune15_cal_variable_chairs.pt")
    ###########################################################################
