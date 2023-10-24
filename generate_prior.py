import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, RandomSampler
import IPython as ipy
import argparse 
from pc_dataset import PointCloudDataset
from models.model_perception import MLPModel, MLPModelDet
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
	test_dataset = PointCloudDataset("data/features_prior.pt", "data/bbox_labels_prior.pt", "data/loss_mask_prior.pt")

	sampler = RandomSampler(test_dataset, replacement=True, num_samples=50)
	dataloader_test = DataLoader(test_dataset, sampler=sampler, batch_size=1)
	###################################################################

	###################################################################
	# Device
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	torch.cuda.set_device(0)
	# device = torch.device('cpu')
	###################################################################

	###################################################################
	# Initialize NN model
	num_in = test_dataset.feature_dims[0]*test_dataset.feature_dims[1]
	num_out = (2,3) # bbox corner representation
	prior =  MLPModel(num_in,num_out)
	prior.init_xi()
	prior.init_logvar(-5,-5)
	prior.to(device)
	priors = []
	for i in range(4):
		priors.append( MLPModelDet(num_in,num_out))
		# priors[i].init_xi()
		# priors[i].init_logvar(-5,-5)
		priors[i].to(device)
	###################################################################

	###################################################################
	# Define optimizer
	optim_prior = []
	for i in range(4): 
		optim_prior.append(torch.optim.Adam(priors[i].parameters(), lr=1e-4) )

	###################################################################

	###################################################################
	# Choose loss weights
	w1 = torch.tensor(1.0).to(device) #1
	w2 = torch.tensor(0.3).to(device)
	w3 = torch.tensor(0.8).to(device) #1

	###################################################################

	# Run the training PRIOR loop
	print("Training prior")
	current_loss = []
	current_loss_true = []
	num_epochs = 200 # 1000
	for batch in range(4):
		sampler = RandomSampler(test_dataset, replacement=True, num_samples=100)
		dataloader_test = DataLoader(test_dataset, sampler=sampler, batch_size=100)
		current_loss.append(0.0)
		current_loss_true.append(0.0)
		for epoch in range(0, num_epochs):

			###################################################################
			# Initialize running losses for this epoch
			current_loss[batch] = 0.0
			current_loss_true[batch] = 0.0
			num_batches = 0
			###################################################################

			###################################################################
			# Iterate over the DataLoader fodr training data
			for i, data in enumerate(dataloader_test, 0):

				# if not i == batch:
				# 	continue

				###################################################################
				# Get inputs, targets, loss mask
				inputs, targets, loss_mask = data
				inputs = inputs.to(device)
				boxes_3detr = targets["bboxes_3detr"].to(device)
				boxes_gt = targets["bboxes_gt"].to(device)
				loss_mask = loss_mask.to(device)
				###################################################################
				# print('Ground truth BB: ', boxes_gt[:,:,:,:][:,:,None,:])
				# print('3DETR prediction BB: ',boxes_3detr[:, :, :, :][:,:,None,:])
				# print('Mask: ', loss_mask)
				###################################################################

				# Perform forward pass
				# prior.init_xi()
				outputs = priors[batch](inputs)
				###################################################################

				###################################################################
				# Compute loss
				loss = box_loss_diff_jit(outputs + boxes_3detr, boxes_gt, w1, w2, w3, loss_mask)

				# Compute true (boolean) version of loss for this batch
				loss_true, not_enclosed = box_loss_true(outputs + boxes_3detr, boxes_gt, loss_mask, 0.01)

				###################################################################

				###################################################################
				# Zero the gradients
				optim_prior[batch].zero_grad()

				# Perform backward pass
				loss.backward()

				# torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

				# Perform optimization
				optim_prior[batch].step()
				###################################################################

				###################################################################
				# Update current loss for this epoch (summing across batches)
				current_loss[batch] += loss.item()
				current_loss_true[batch] += loss_true.item()
				num_batches += 1
			###################################################################
			###################################################################
			# Print losses (averaged across batches in this epoch)
			print_interval = 1
			if verbose and (epoch % print_interval == 0) and (num_batches > 0):
				print("batch: ", batch, "; epoch: ", epoch, "; loss: ", '{:02.6f}'.format(current_loss[batch]/num_batches),
						"; loss true: ", '{:02.6f}'.format(current_loss_true [batch]/ num_batches), end='\r')
			###################################################################

	# Save model
	torch.save(priors[0].state_dict(), "trained_models/perception_prior_nav_sim_e0")
	torch.save(priors[1].state_dict(), "trained_models/perception_prior_nav_sim_e1")
	torch.save(priors[2].state_dict(), "trained_models/perception_prior_nav_sim_e2")
	torch.save(priors[3].state_dict(), "trained_models/perception_prior_nav_sim_e3")

	###################################################################
	# Load ensembles
	for i in range(4):
		priors[i].load_state_dict(torch.load(("trained_models/perception_prior_nav_sim_e"+str(i))))
		priors[i].eval()
	# Calculate ensemble variance
	mean_linear1 = torch.zeros(200, 32768)
	mean_linear2 = torch.zeros(6, 200)
	bias_linear1 = torch.zeros(200)
	bias_linear2 = torch.zeros(6)
	var_linear1 = torch.zeros_like(mean_linear1)
	var_linear2 = torch.zeros_like(mean_linear2)
	bias_var_linear1 = torch.zeros_like(bias_linear1)
	bias_var_linear2 = torch.zeros_like(bias_linear2)
	data1 = torch.zeros_like(mean_linear1)
	data2 = torch.zeros_like(mean_linear2)
	data1b = torch.zeros_like(bias_linear1)
	data2b = torch.zeros_like(bias_linear2)
	for i in range(4):
		n = i+1
		data1 = priors[i].linear1.weight.cpu()
		data2 = priors[i].linear2.weight.cpu()
		data1b = priors[i].linear1.bias.cpu()
		data2b = priors[i].linear2.bias.cpu()
		mean_linear1 = ((n-1)*mean_linear1 + data1)/(n)
		mean_linear2 = ((n-1)*mean_linear2 + data2)/(n)
		bias_linear1 = ((n-1)*bias_linear1 + data1b)/(n)
		bias_linear2 = ((n-1)*bias_linear2 + data2b)/(n)
		if n ==1:
			continue
		var_linear1 = (n-1)*var_linear1/(n) + torch.square(mean_linear1 - data1)/(n-1)
		var_linear2 = (n-1)*var_linear2/(n) + torch.square(mean_linear2 - data2)/(n-1)
		bias_var_linear1 = (n-1)*bias_var_linear1/(n) + torch.square(bias_linear1 - data1b)/(n-1)
		bias_var_linear2 = (n-1)*bias_var_linear2/(n) + torch.square(bias_linear2 - data2b)/(n-1)
	logvar_linear1 = torch.log(var_linear1)/2
	logvar_linear2 = torch.log(var_linear2)/2
	logvar_linear1b = torch.log(bias_var_linear1)/2
	logvar_linear2b = torch.log(bias_var_linear2)/2
	if verbose:
		# print('Saved trained model.')
		print("Model's state_dict:")
		print("Linear 1: Mean: ", mean_linear1, "; Log Variance: ", logvar_linear1)
		print("Linear 2: Mean: ", mean_linear2, "; Log Variance: ", logvar_linear2)
		print("Linear 1: Bias Mean: ", bias_linear1, "; Bias Log Variance: ", logvar_linear1b)
		print("Linear 2: Bias Mean: ", bias_linear2, "; Bias Log Variance: ", logvar_linear2b)
	###################################################################

	prior.linear1.init_mu(mean_linear1.to(device), bias_linear1.to(device))
	prior.linear2.init_mu(mean_linear2.to(device), bias_linear2.to(device))
	prior.linear1.init_logvar(logvar_linear1.to(device), logvar_linear1b.to(device))
	prior.linear2.init_logvar(logvar_linear2.to(device), logvar_linear2b.to(device))


	# Save model
	torch.save(prior.state_dict(), "trained_models/perception_prior_nav_sim")


###################################################################

# Run with command line arguments precisely when called directly
# (rather than when imported)
if __name__ == '__main__':
	main() 

