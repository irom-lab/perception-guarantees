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
from e3nn.util.jit import script, trace
from loss_fn import *
from copy import deepcopy

import sys
sys.path.append('../utils')
from utils.pac_util import PAC_Bayes_regularizer, kl_inv_l

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
	dataset = PointCloudDataset("data/features.pt", "data/bbox_labels.pt", "data/loss_mask.pt")
	batch_size = 1 #100
	N=500
	delta = 0.009
	deltap = 0.001
	num_evaluations = 1000

	params = {'batch_size': batch_size,
				'shuffle': False}
	           # 'num_workers': 12}
	dataloader = DataLoader(dataset, **params)
	###################################################################
	###################################################################
	# Device
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	torch.cuda.set_device(0)
	# device = torch.device('cpu')
	###################################################################

	###################################################################
	# Initialize NN model
	num_in = dataset.feature_dims[0]*dataset.feature_dims[1]
	num_out = (2,3) # bbox corner representation
	model = MLPModel(num_in, num_out)
	model.to(device)
	# model.init_xi()
	prior = MLPModel(num_in,num_out)
	prior.init_xi()
	prior.init_logvar(-4,-4)
	prior.to(device)
	model.load_state_dict(deepcopy(prior.state_dict()))
	mean = torch.zeros(1,1,2,3)
	variance = torch.zeros_like(mean)
	n = 1

	for i, data in enumerate(dataloader, 0):

		if (i < 15):
			###################################################################
			# Get inputs, targets, loss mask
			inputs, targets, loss_mask = data
			inputs = inputs.to(device)
			boxes_3detr = targets["bboxes_3detr"].to(device)
			boxes_gt = targets["bboxes_gt"].to(device)
			loss_mask = loss_mask.to(device)
			idx_loss = np.nonzero(np.array(loss_mask.cpu()))
			idx_box = np.nonzero(np.array((boxes_3detr[0,0,:,:] - boxes_3detr[0,:,:,:]).cpu()))
			# print(boxes_3detr[0,0,:,:], idx_loss[1], idx_box[0])
			# ipy.embed()

			k = 0
			for (i,j) in enumerate(idx_loss[1]):
				while (j>=idx_box[0][k] and k < len(idx_loss[1])):
					# ipy.embed()
					if(j==idx_box[0][k]):
						prior.init_xi()
						data = prior(inputs[0,j,:,:].reshape((1,1,128,256)).to(device)).cpu()
						mean = (n*mean + data)/(n+1)
						variance = n*variance/(n+1) + torch.square(mean - data)/(n)
						n = n+1
						k = k +6
					elif (j>idx_box[0][k]):
						k=k+1

	print(mean, variance)
###################################################################

# Run with command line arguments precisely when called directly
# (rather than when imported)
if __name__ == '__main__':
    main() 
