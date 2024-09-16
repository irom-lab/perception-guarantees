import numpy as np
import torch
from torch.utils.data import DataLoader
import IPython as ipy
import argparse 
from pc_dataset import PointCloudDataset
from loss_fn import *
from copy import deepcopy
from matplotlib import pyplot as plt

import sys
sys.path.append('../utils')

def main(raw_args=None):


	###################################################################
	# Parse arguments
	parser = argparse.ArgumentParser()
	parser.add_argument("--verbose", type=int, default=1, help="print more (default: 1)")

	args = parser.parse_args(raw_args)
	verbose = args.verbose
	###################################################################

	###################################################################
	# Initialize dataset and dataloader
	dataset = PointCloudDataset("/media/zm2074/Data Drive/data/perception-guarantees/calibrate_1.5k/data/features.pt", "/media/zm2074/Data Drive/data/perception-guarantees/calibrate_1.5k/data/bbox_labels.pt", "/media/zm2074/Data Drive/data/perception-guarantees/calibrate_1.5k/data/loss_mask.pt")
	dataloader_cp = DataLoader(dataset, batch_size=len(dataset))
	###################################################################

	###################################################################
	# Device
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	torch.cuda.set_device(0)
	# device = torch.device('cpu')
	###################################################################

	#################################################################
	# Without finetuning
	for i, data in enumerate(dataloader_cp, 0):
		print(i)
		inputs, targets, loss_mask = data
		boxes_3detr = targets["bboxes_3detr"].to(device)
		boxes_gt = targets["bboxes_gt"].to(device)
		loss_mask = loss_mask.to(device)


		corners_pred = boxes_3detr
		corners_gt = boxes_gt
		tol = 0.887

		B, K = corners_gt.shape[0], corners_gt.shape[1]

		# 2D projection
		corners_gt = corners_gt[:,:,:,:,0:2]
		corners_pred = corners_pred[:,:,:,:,0:2]

		# Ensure that corners of predicted bboxes satisfy basic constraints
		corners1_pred = torch.min(corners_pred[:, :, :, 0, :][:,:,None,:], corners_pred[:, :, :, 1, :][:,:,None,:])
		corners2_pred = torch.max(corners_pred[:, :, :, 0, :][:,:,None,:], corners_pred[:, :, :, 1, :][:,:,None,:])

		# Compute the mean position of the ground truth and predicted bounding boxes
		pred_center = torch.div(corners_pred[:, :, 0, :][:,:,None,:] + corners_pred[:, :, 1, :][:,:,None,:],2)
		gt_center = torch.div(corners_gt[:, :, 0, :][:,:,None,:] + corners_gt[:, :, 1, :][:,:,None,:],2)

		# Calculate the scaling between predicted and ground truth boxes
		corners1_diff = (corners1_pred - corners_gt[:,:,:,0,:][:,:,None,:])
		corners2_diff = (corners_gt[:,:,:,1,:][:,:,None,:] - corners2_pred)
		corners1_diff = torch.squeeze(corners1_diff,2)
		corners2_diff = torch.squeeze(corners2_diff,2)
		corners1_diff_mask = torch.mul(loss_mask,corners1_diff.amax(dim=3))
		corners2_diff_mask = torch.mul(loss_mask, corners2_diff.amax(dim=3))
		corners1_diff_mask[loss_mask == 0] = -np.inf
		corners2_diff_mask[loss_mask == 0] = -np.inf
		# ipy.embed()
		corners1_diff_mask = corners1_diff_mask.amax(dim=2)
		corners2_diff_mask = corners2_diff_mask.amax(dim=2)
		delta_all = torch.maximum(corners1_diff_mask, corners2_diff_mask)

		delta = delta_all.amax(dim=1)
		delta, indices = torch.sort(delta, dim=0, descending=False)
		idx = math.ceil((B+1)*(tol))-1
		
		idx = math.ceil((B+1)*(tol))-1



		ipy.embed()
		scaling_cp = scale_prediction(boxes_3detr, boxes_gt, loss_mask, 0.887) #for coverage of 0.85 w.p. 0.99 
		average_cp = scale_prediction_average(boxes_3detr, boxes_gt, loss_mask, 0.887)
		print('CP quantile prediction', scaling_cp)
		print('CP quantile prediction (for baseline CP-avg.)', average_cp)
	#################################################################

# Run with command line arguments precisely when called directly
# (rather than when imported)
if __name__ == '__main__':
	main() 

