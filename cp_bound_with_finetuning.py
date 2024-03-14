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
from models.model_perception import MLPModel, MLPModelDet
from e3nn.util.jit import script, trace
from loss_fn import *
from copy import deepcopy
from matplotlib import pyplot as plt

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
	dataset = PointCloudDataset("data/features15_cal_variable_chairs.pt", "data/bbox_labels15_cal_variable_chairs.pt", "data/loss_mask15_cal_variable_chairs.pt", "data/finetune15_cal_variable_chairs.pt")
	batch_size = 100 #100
	N=len(dataset)
	N_obj = dataset.bbox_labels.shape[2]
	split_cp_size = 100

	params = {'batch_size': batch_size,
				'shuffle': False}
	dataloader_test = DataLoader(dataset, batch_size=1)
	dataloader_cp = DataLoader(dataset, batch_size=(len(dataset)-split_cp_size))
	###################################################################

	###################################################################
	# Device
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	torch.cuda.set_device(0)
	# device = torch.device('cpu')
	###################################################################

	###################################################################
	# Conformal Prediction

	# Fine-tuning
	# Initialize NN model
	num_in = dataset.feature_dims[0]*dataset.feature_dims[1]
	num_out = (15, 2,3) # bbox corner representation
	model_cp = MLPModelDet(num_in, num_out)
	print(num_in,num_out)
	model_cp.to(device)
	# Define optimizer
	optimizer_cp = torch.optim.Adam(model_cp.parameters(), lr=1e-3) 
	###################################################################
	# Choose loss weights
	w1 = torch.tensor(1.0).to(device) #1
	w2 = torch.tensor(0.5).to(device) #0.1
	w3 = torch.tensor(0.1).to(device) #1

	# Run the finetuning  loop
	print("Finetuning")
	num_epochs = 200 # 1000
	for epoch in range(0, num_epochs):

		# Initialize running losses for this epoch
		current_loss = 0.0
		current_loss_true = 0.0
		num_batches = 0
		for i, data in enumerate(dataloader_test, 0):
			if i < len(dataset) - split_cp_size:
				continue

			# Get inputs, targets, loss mask
			inputs, targets, loss_mask, finetune = data
			inputs = inputs.to(device)
			boxes_3detr = finetune["bboxes_3detr"].to(device)
			boxes_gt = finetune["bboxes_gt"].to(device)
			loss_mask = torch.ones(loss_mask.shape[0], loss_mask.shape[1],  boxes_gt.shape[2]).to(device)

			# Perform forward pass
			outputs = model_cp(inputs).to(device)
			if  torch.any(torch.isnan(outputs)):
				idx = torch.where(torch.isnan(outputs))
				outputs[idx] = 0.0
			loss = box_loss_diff_jit(outputs + boxes_3detr, boxes_gt, w1, w2, w3, loss_mask)

			# Compute true (boolean) version of loss for this batch
			loss_true, not_enclosed = box_loss_true(outputs + boxes_3detr, boxes_gt, loss_mask, 0.01)

			# Zero the gradients
			optimizer_cp.zero_grad()

			# Perform backward pass
			loss.backward()

			# Perform optimization
			optimizer_cp.step()
			# Update current loss for this epoch (summing across batches)
			current_loss += loss.item()
			current_loss_true += loss_true.item()
			num_batches += 1
		# Print losses (averaged across batches in this epoch)
		print_interval = 1
		if verbose and (epoch % print_interval == 0):
			print("epoch: ", epoch, "; loss: ", '{:02.6f}'.format(current_loss/num_batches),
				  "; loss true: ", '{:02.6f}'.format(current_loss_true / num_batches), end='\r')
		###################################################################

	torch.save(model_cp.state_dict(), "trained_models/perception_model")
	#################################################################
			
	# With finetuning
	print("Calculating CP with finetuned model...")
	for i, data in enumerate(dataloader_cp, 0):
		if i > 0:
			continue
		inputs, targets, loss_mask, finetune = data
		# print(len(dataset), len(inputs))
		inputs = inputs.to(device)
		boxes_3detr = targets["bboxes_3detr"].to(device)
		boxes_gt = targets["bboxes_gt"].to(device)
		boxes_3detr_all = finetune["bboxes_3detr"].to(device)
		ipy.embed()
		loss_mask = loss_mask.to(device)
		outputs = model_cp(inputs)
		outputs_sorted = torch.clone(outputs)
		for ii, jj in np.ndindex((boxes_3detr.shape[0], boxes_3detr.shape[1])):
			for kk in range(boxes_3detr.shape[2]):
				if loss_mask[ii,jj,kk] == 1:
					idx = (boxes_3detr_all[ii,jj,:,:,:] == boxes_3detr[ii,jj,kk,:,:]).nonzero()
					if len(idx) >= 6 and all(idx[0:6,0] == idx[0,0]):
						outputs_sorted[ii,jj,kk,:,:] = outputs[ii,jj,idx[0,0],:,:]
					else:
						print("This should not happen")
						ipy.embed()
		finetuned_boxes = outputs_sorted[:,:,0:boxes_3detr.shape[2],:,:]
		scaling_cp = scale_prediction(boxes_3detr+finetuned_boxes, boxes_gt, loss_mask, 0.887) #for coverage of 0.85 w.p. 0.99 
		average_cp = scale_prediction_average(boxes_3detr, boxes_gt, loss_mask, 0.887)
		print('CP quantile prediction', scaling_cp)
		print('CP quantile prediction (for baseline CP-avg.)', average_cp)

# Run with command line arguments precisely when called directly
# (rather than when imported)
if __name__ == '__main__':
	main() 

