""" 
Visualize point cloud and bounding boxes.
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

from pc_util import write_bbox, point_cloud_to_bbox, preprocess_point_cloud, write_ply, read_ply, write_bbox_ply_from_outputs
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
    ###########################################################################

    ###########################################################################
    # Read point cloud    
    env = 5
    loc_ind = 0 # -1
    point_cloud = data[env]["point_clouds"][loc_ind]

    pc_ply = tuple(map(tuple, point_cloud.reshape(num_pc_points,3)))
    pc_ply = list(pc_ply)
    write_ply(pc_ply, "gibson.ply")
    # write_ply_rgb(points, colors, "gibson.obj")


    pc = torch.from_numpy(point_cloud).to(device)
    pc_min = pc.min(1).values
    pc_max = pc.max(1).values
    inputs = {'point_clouds': pc, 'point_cloud_dims_min': pc_min, 'point_cloud_dims_max': pc_max}

    # Run through pre-trained 3DETR model
    outputs = model(inputs)

    ipy.embed()

    # Write bbox from outputs
    num_objects = write_bbox_ply_from_outputs(outputs, "output_bboxes.ply", prob_threshold=0.1)
    print(" ")
    print("Number of objects detected: ", num_objects)
    print(" ")

    # Write ground truth bbox
    bbox_world_frame = data[env]["bbox_world_frame_vertices"]
    scene_bbox = point_cloud_to_bbox(bbox_world_frame)
    scene_bbox = scene_bbox.reshape((1,6))
    write_bbox(scene_bbox, "bbox_ground_truth.ply")

#     scene_bbox = np.concatenate((bbox_center, bbox_bf_extent, np.array([euler_from_quat(bbox_orn)[2]])))
#     scene_bbox = scene_bbox.reshape((1,7))
#     write_oriented_bbox(scene_bbox, "gibson_bbox.ply")








