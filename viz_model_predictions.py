import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import IPython as ipy
import argparse
from pc_dataset import PointCloudDataset
from models.model_perception import MLPModel
from utils.pc_util import write_ply, point_cloud_to_bbox, write_bbox

def main(raw_args=None):

	###################################################################
	# Set environment and camera location for which we should generate ply files
	env = 0 # Environment index
	loc = 45 # Camera location index
	###################################################################

	###################################################################
	# Parse arguments
	parser = argparse.ArgumentParser()
	parser.add_argument("--verbose", type=int, default=1, help="print more (default: 1")

	args = parser.parse_args(raw_args)
	verbose = args.verbose
	###################################################################

	###################################################################
	# Initialize dataset and dataloader
	dataset = PointCloudDataset("data/features.pt", "data/bbox_labels.pt", "data/loss_mask.pt")

	# Get point cloud data
	data_raw = np.load("data/training_data_raw.npz", allow_pickle=True)
	data_raw = data_raw["data"]
	num_pc_points = data_raw[0]["point_clouds"][0].shape[1]  # Number of points in each point cloud
	###################################################################

	###################################################################
	# Device
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	# device = torch.device('cpu')
	###################################################################

	###################################################################
	# Load trained perception model
	num_in = dataset.feature_dims[0] * dataset.feature_dims[1]
	num_out = (2, 3)
	model = MLPModel(num_in, num_out)

	model.load_state_dict(torch.load("trained_models/perception_model"))
	model.to(device)
	model.eval()

	###################################################################

	###################################################################

	# Get point cloud and save as ply
	point_cloud = data_raw[env]["point_clouds"][loc]
	pc_ply = tuple(map(tuple, point_cloud.reshape(num_pc_points, 3)))
	pc_ply = list(pc_ply)
	write_ply(pc_ply, "viz_pcs/gibson_camera.ply")

	# Get ground truth box and save as ply
	bbox_gt = dataset.bbox_labels[env, loc, :, :].numpy()
	bbox_gt = point_cloud_to_bbox(bbox_gt)
	bbox_gt = bbox_gt.reshape((1, 6))
	write_bbox(bbox_gt, "viz_pcs/bbox_gt.ply")

	# Get 3DETR prediction and save as ply
	bbox_3detr_orig = dataset.bbox_3detr[env, loc, :, :].numpy()
	bbox_3detr = point_cloud_to_bbox(bbox_3detr_orig)
	bbox_3detr = bbox_3detr.reshape((1, 6))
	write_bbox(bbox_3detr, "viz_pcs/bbox_3detr.ply")

	# Run trained perception model and save as ply
	features = dataset.features["box_features"][env,loc,:,:][None,None,:,:]
	features = features.to(device)
	outputs = model(features)
	bbox_pred = outputs[0,0,:,:].cpu().detach() + bbox_3detr_orig
	bbox_pred = bbox_pred.numpy()

	bbox_pred = point_cloud_to_bbox(bbox_pred)
	bbox_pred = bbox_pred.reshape((1, 6))
	write_bbox(bbox_pred, "viz_pcs/bbox_pred.ply")


#################################################################
# Run with command line arguments precisely when called directly
# (rather than when imported)
if __name__ == '__main__':
    main()