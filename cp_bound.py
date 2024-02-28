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
	dataset = PointCloudDataset("data/features15_cal_variable_chairs.pt", "data/bbox_labels15_cal_variable_chairs.pt", "data/loss_mask15_cal_variable_chairs.pt")
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
		inputs, targets, loss_mask = data
		boxes_3detr = targets["bboxes_3detr"].to(device)
		boxes_gt = targets["bboxes_gt"].to(device)
		loss_mask = loss_mask.to(device)
		scaling_cp = scale_prediction(boxes_3detr, boxes_gt, loss_mask, 0.887) #for coverage of 0.85 w.p. 0.99 
		average_cp = scale_prediction_average(boxes_3detr, boxes_gt, loss_mask, 0.887)
		print('CP quantile prediction', scaling_cp)
		print('CP quantile prediction (for baseline CP-avg.)', average_cp)
	#################################################################

# Run with command line arguments precisely when called directly
# (rather than when imported)
if __name__ == '__main__':
	main() 

