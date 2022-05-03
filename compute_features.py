""" 
Compute feature representation from point clouds using pre-trained 3DETR. 
"""

import os
import sys
import numpy as np
import argparse
import importlib
import time
import IPython as ipy

import torch
import torch.nn as nn
import torch.optim as optim

from models import build_model
from datasets import build_dataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'pretrained'))

from pc_util import preprocess_point_cloud, read_ply, write_bbox_ply_from_outputs
from make_args import make_args_parser


if __name__=='__main__':


    ###########################################################################
    # Parse arguments
    parser = make_args_parser()
    args = parser.parse_args()

    # Dataset config: use SUNRGB-D
    from datasets.sunrgbd import SunrgbdDatasetConfig as dataset_config
    dataset_config = dataset_config()

    # Build model
    model, _ = build_model(args, dataset_config)

    # Load pre-trained weights
    sd = torch.load(args.test_ckpt, map_location=torch.device("cpu")) 
    model.load_state_dict(sd["model"]) 

    model = model.cuda()
    model.eval()

    device = torch.device("cuda")
    ###########################################################################


    ###########################################################################
    # Get data
    data = np.load("training_data_raw.npz", allow_pickle=True)
    data = data["data"]

    num_envs = len(data) # Number of environments where we collected data
    num_cam_positions = len(data[0]["cam_positions"]) # Number of camera locations in this environment

    num_pc_points = data[0]["point_clouds"][0].shape[1] # Number of points in each point cloud

    # Batch size for processing inputs on GPU
    batch_size = 2 # Cannot be too large since GPU runs out of memory

    num_batches = int(num_cam_positions / batch_size)
    assert (num_cam_positions % batch_size) == 0, "batch_size must divide num_cam_positions."
    ###########################################################################

    ###########################################################################


    ###########################################################################
    # Initialize data structure for ground truth bounding boxes for each environment
    bboxes_ground_truth = torch.zeros(num_envs, 8, 3)


    # Initialize outputs
    model_outputs_all = {
        "sem_cls_logits": torch.zeros(num_envs, num_cam_positions, args.nqueries, dataset_config.num_semcls+1),
        "center_normalized": torch.zeros(num_envs, num_cam_positions, args.nqueries, 3),
        "center_unnormalized": torch.zeros(num_envs, num_cam_positions, args.nqueries, 3),
        "size_normalized": torch.zeros(num_envs, num_cam_positions, args.nqueries, 3),
        "size_unnormalized": torch.zeros(num_envs, num_cam_positions, args.nqueries, 3),
        "angle_logits": torch.zeros(num_envs, num_cam_positions, args.nqueries, dataset_config.num_angle_bin),
        "angle_residual": torch.zeros(num_envs, num_cam_positions, args.nqueries, dataset_config.num_angle_bin),
        "angle_residual_normalized": torch.zeros(num_envs, num_cam_positions, args.nqueries, dataset_config.num_angle_bin),
        "angle_continuous": torch.zeros(num_envs, num_cam_positions, args.nqueries),
        "objectness_prob": torch.zeros(num_envs, num_cam_positions, args.nqueries),
        "sem_cls_prob": torch.zeros(num_envs, num_cam_positions, args.nqueries, dataset_config.num_semcls),
        "box_corners": torch.zeros(num_envs, num_cam_positions, args.nqueries, 8, 3),
        "box_features": torch.zeros(num_envs, num_cam_positions, args.nqueries, args.dec_dim)
    }
    ###########################################################################

    ###########################################################################

    t_start = time.time()
    for env in range(num_envs):

        print("Env: ", env)

        #####################################################
        # Save ground truth bounding box for this environment
        bboxes_ground_truth[env,:,:] = torch.tensor(data[env]["bbox_world_frame_vertices"])
        #####################################################

        for i in range(num_batches):

            # Read point clouds    
            batch_inds = slice(i*batch_size, (i+1)*batch_size)
            point_clouds = data[env]["point_clouds"][batch_inds]
            pc = np.array(point_clouds)
            pc = pc.reshape((batch_size, num_pc_points, 3))
            pc_all = torch.from_numpy(pc).to(device)
            pc_min_all = pc_all.min(1).values
            pc_max_all = pc_all.max(1).values
            inputs = {'point_clouds': pc_all, 'point_cloud_dims_min': pc_min_all, 'point_cloud_dims_max': pc_max_all}

            # Run through pre-trained 3DETR model
            outputs = model(inputs)


            #####################################
            # Save outputs from model
            model_outputs_all["sem_cls_logits"][env,batch_inds,:,:] = outputs["outputs"]["sem_cls_logits"].detach().cpu()
            model_outputs_all["center_normalized"][env,batch_inds,:,:] = outputs["outputs"]["center_normalized"].detach().cpu()
            model_outputs_all["center_unnormalized"][env,batch_inds,:,:] = outputs["outputs"]["center_unnormalized"].detach().cpu()
            model_outputs_all["size_normalized"][env,batch_inds,:,:] = outputs["outputs"]["size_normalized"].detach().cpu()
            model_outputs_all["size_unnormalized"][env,batch_inds,:,:] = outputs["outputs"]["size_unnormalized"].detach().cpu()
            model_outputs_all["angle_logits"][env,batch_inds,:,:] = outputs["outputs"]["angle_logits"].detach().cpu()
            model_outputs_all["angle_residual"][env,batch_inds,:,:] = outputs["outputs"]["angle_residual"].detach().cpu()
            model_outputs_all["angle_residual_normalized"][env,batch_inds,:,:] = outputs["outputs"]["angle_residual_normalized"].detach().cpu()
            model_outputs_all["angle_continuous"][env,batch_inds,:] = outputs["outputs"]["angle_continuous"].detach().cpu()
            model_outputs_all["objectness_prob"][env,batch_inds,:] = outputs["outputs"]["objectness_prob"].detach().cpu()
            model_outputs_all["sem_cls_prob"][env,batch_inds,:,:] = outputs["outputs"]["sem_cls_prob"].detach().cpu()
            model_outputs_all["box_corners"][env,batch_inds,:,:] = outputs["outputs"]["box_corners"].detach().cpu()
            model_outputs_all["box_features"][env,batch_inds,:,:] = outputs["box_features"].detach().cpu()
            #####################################


    t_end = time.time()
    print("Time: ", t_end - t_start)
    ###########################################################################



    ###########################################################################
    # Save processed feature data
    torch.save(model_outputs_all, "features.pt")


    # Save ground truth bounding boxes
    torch.save(bboxes_ground_truth, "bbox_labels.pt")
    ###########################################################################


