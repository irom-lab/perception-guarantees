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
	# dataset = PointCloudDataset("data/features15_cal_variable_chairs.pt", "data/bbox_labels15_cal_variable_chairs.pt", "data/loss_mask15_cal_variable_chairs.pt")
	# dataset = PointCloudDataset("data/features15_cal.pt", "data/bbox_labels15_cal.pt", "data/loss_mask15_cal.pt")
	# prior_dataset = PointCloudDataset("data/features15_prior.pt", "data/bbox_labels15_prior.pt", "data/loss_mask15_prior.pt", "data/finetune15_prior.pt")
	# test_dataset = PointCloudDataset("data/features15_test_variable_chairs.pt", "data/bbox_labels15_test_variable_chairs.pt", "data/loss_mask15_test_variable_chairs.pt", "data/finetune15_test_variable_chairs.pt")
	# test_dataset = PointCloudDataset("data/features_test.pt", "data/bbox_labels_test.pt", "data/loss_mask_test.pt")
	batch_size = 100 #100
	N=len(dataset)
	N_obj = dataset.bbox_labels.shape[2]
	# print("Num environments: ", N, " Test: ", len(test_dataset), " Prior: ", len(prior_dataset))
	delta = 0.009
	deltap = 0.001
	num_evaluations = 10000
	split_cp_size = 100

	params = {'batch_size': batch_size,
				'shuffle': False}
			   # 'num_workers': 12}
	dataloader = DataLoader(dataset, **params)
	# dataloader_prior = DataLoader(prior_dataset, batch_size=1)
	dataloader_test = DataLoader(dataset, batch_size=1)
	dataloader_cp = DataLoader(dataset, batch_size=len(dataset)-split_cp_size)
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

	# model_cp.load_state_dict(torch.load("trained_models/perception_model_planner"))

	# Run the finetuning  loop
	print("Finetuning")
	num_epochs = 200 # 1000
	for epoch in range(0, num_epochs):

		# Initialize running losses for this epoch
		current_loss = 0.0
		current_loss_true = 0.0
		num_batches = 0
		for i, data in enumerate(dataloader_test, 0):
			if i >= split_cp_size:

				# Get inputs, targets, loss mask
				inputs, targets, loss_mask, finetune = data
				inputs = inputs.to(device)
				boxes_3detr = finetune["bboxes_3detr"].to(device)
				boxes_gt = finetune["bboxes_gt"].to(device)
				loss_mask = torch.ones(loss_mask.shape[0], loss_mask.shape[1],  boxes_gt.shape[2]).to(device)

				# Perform forward pass
				outputs = model_cp(inputs).to(device)
				if  torch.any(torch.isnan(outputs)):
					# ipy.embed()
					idx = torch.where(torch.isnan(outputs))
					outputs[idx] = 0.0
					# loss_mask[idx[0], idx[1], idx[2]] = 0
					# print("Skipping ", i)
					# continue
				# Compute loss
				# print("Outputs: ", outputs.shape, " Boxes 3DETR: ", boxes_3detr.shape, " boxes gt: ", boxes_gt.shape, " Loss mask: ", loss_mask.shape)
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
	# 	###################################################################

	torch.save(model_cp.state_dict(), "trained_models/perception_model100")
	#################################################################
	# # Without finetuning
	# for i, data in enumerate(dataloader_cp, 0):
	# 	inputs, targets, loss_mask = data
	# 	# print(len(dataset), len(inputs))
	# 	# inputs = inputs.to(device)
	# 	boxes_3detr = targets["bboxes_3detr"].to(device)
	# 	boxes_gt = targets["bboxes_gt"].to(device)
	# 	loss_mask = loss_mask.to(device)
	# 	scaling_cp = scale_prediction(boxes_3detr, boxes_gt, loss_mask, 0.887) #for coverage of 0.95 w.p. 0.99 
	# 	average_cp = scale_prediction_average(boxes_3detr, boxes_gt, loss_mask, 0.887)
	# 	print('CP quantile prediction', scaling_cp)
	# 	print('CP quantile prediction (average)', average_cp)
	# #################################################################
			
	# With finetuning
	print("Calculating CP with finetuned model...")
	for i, data in enumerate(dataloader_cp, 0):
		if i < 1:
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
			scaling_cp = scale_prediction(boxes_3detr+finetuned_boxes, boxes_gt, loss_mask, 0.887) #for coverage of 0.95 w.p. 0.99 
			average_cp = scale_prediction_average(boxes_3detr, boxes_gt, loss_mask, 0.887)
			print('CP quantile prediction', scaling_cp)
			print('CP quantile prediction (average)', average_cp)

	###################################################################

	scaling_cp = 0.6138
	scaling_cp = 0.735
	# torch.cuda.empty_cache()
	# ###################################################################
	# # Initialize NN model
	# num_in = dataset.feature_dims[0]*dataset.feature_dims[1]
	# num_out = (2,3) # bbox corner representation
	# model = MLPModel(num_in, num_out)
	# model.to(device)
	# # model.init_xi()
	# prior = MLPModel(num_in,num_out)
	# prior.init_xi()
	# prior.init_logvar(-5,-5)
	# # prior.load_state_dict(torch.load("trained_models/perception_prior_nav_sim"))
	# # prior.eval()
	# prior.to(device)
	# # prior.init_xi()
	# # prior.init_mu()
	# # model.load_state_dict(deepcopy(prior.state_dict()))

	# # model.load_state_dict(torch.load("trained_models/perception_model"))
	# # model.eval()
	# # prior.load_state_dict(torch.load("trained_models/perception_prior"))
	# # prior.eval()
	# ###################################################################

	# ###################################################################
	# # Define optimizer
	# optimizer = torch.optim.Adam(model.parameters(), lr=1e-4) 
	# optim_prior = torch.optim.Adam(prior.parameters(), lr=1e-4) 

	# ###################################################################

	# ###################################################################
	# # Choose loss weights
	# w1 = torch.tensor(1.0).to(device) #1
	# w2 = torch.tensor(0.3).to(device) #0.1
	# w3 = torch.tensor(0.1).to(device) #1

	# ###################################################################
	# ###################################################################
	# # Run the PRIOR training loop
	# print("Training prior")
	# num_epochs = 100 # 1000
	# for epoch in range(0, num_epochs):

	# 	###################################################################
	# 	# Initialize running losses for this epoch
	# 	current_loss = 0.0
	# 	current_loss_true = 0.0
	# 	num_batches = 0
	# 	###################################################################

	# 	###################################################################
	# 	# Iterate over the DataLoader for training data
	# 	for i, data in enumerate(dataloader_prior, 0):

	# 		# if i < 1:
	# 		# 	continue

	# 		###################################################################
	# 		# Get inputs, targets, loss mask
	# 		inputs, targets, loss_mask = data
	# 		inputs = inputs.to(device)
	# 		boxes_3detr = targets["bboxes_3detr"].to(device)
	# 		boxes_gt = targets["bboxes_gt"].to(device)
	# 		loss_mask = loss_mask.to(device)
	# 		###################################################################
	# 		# print('Ground truth BB: ', boxes_gt[:,:,:,:][:,:,None,:])
	# 		# print('3DETR prediction BB: ',boxes_3detr[:, :, :, :][:,:,None,:])
	# 		# print('Mask: ', loss_mask)
	# 		###################################################################

	# 		# Perform forward pass
	# 		prior.init_xi()
	# 		outputs = prior(inputs)
	# 		###################################################################

	# 		###################################################################
	# 		# Compute loss
	# 		# reg = PAC_Bayes_regularizer(model, prior, N, delta, device)
	# 		loss = box_loss_diff_jit(outputs + boxes_3detr, boxes_gt, w1, w2, w3, loss_mask)
	# 		# loss = loss + torch.sqrt(reg/2)

	# 		# Compute true (boolean) version of loss for this batch
	# 		loss_true, not_enclosed = box_loss_true(outputs + boxes_3detr, boxes_gt, loss_mask, 0.01)

	# 		###################################################################

	# 		###################################################################
	# 		# Zero the gradients
	# 		optim_prior.zero_grad()

	# 		# Perform backward pass
	# 		loss.backward()

	# 		# torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

	# 		# Perform optimization
	# 		optim_prior.step()
	# 		###################################################################

	# 		###################################################################
	# 		# Update current loss for this epoch (summing across batches)
	# 		current_loss += loss.item()
	# 		current_loss_true += loss_true.item()
	# 		num_batches += 1
	# 	###################################################################

	# 	###################################################################
	# 	# Print losses (averaged across batches in this epoch)
	# 	print_interval = 1
	# 	if verbose and (epoch % print_interval == 0):
	# 		print("epoch: ", epoch, "; loss: ", '{:02.6f}'.format(current_loss/num_batches),
	# 			  "; loss true: ", '{:02.6f}'.format(current_loss_true / num_batches), end='\r')
	# 	###################################################################

	# ###################################################################
	# ###################################################################

	# # prior.init_logvar(-5,-10)

	# model.load_state_dict(deepcopy(prior.state_dict()))
	# # model.init_logvar(-5,-10)
	# # Choose loss weights
	# w1 = torch.tensor(1.0).to(device) #1
	# w2 = torch.tensor(1.0).to(device) #0.1
	# w3 = torch.tensor(0.1).to(device) #1

	# # Run the training loop
	# print("Training posterior")
	# num_epochs = 100 # 1000
	# for epoch in range(0, num_epochs):

	# 	###################################################################
	# 	# Initialize running losses for this epoch
	# 	current_loss = 0.0
	# 	current_loss_true = 0.0
	# 	num_batches = 0
	# 	###################################################################
			
	# 	###################################################################
	# 	# Iterate over the DataLoader for training data
	# 	for i, data in enumerate(dataloader, 0):

	# 		# if i < 1:
	# 		# 	continue

	# 		###################################################################
	# 		# Get inputs, targets, loss mask
	# 		inputs, targets, loss_mask = data
	# 		inputs = inputs.to(device)
	# 		boxes_3detr = targets["bboxes_3detr"].to(device)
	# 		boxes_gt = targets["bboxes_gt"].to(device)
	# 		loss_mask = loss_mask.to(device)
	# 		###################################################################
	# 		# print('Ground truth BB: ', boxes_gt[:,:,:,:][:,:,None,:])
	# 		# print('3DETR prediction BB: ',boxes_3detr[:, :, :, :][:,:,None,:])
	# 		# print('Mask: ', loss_mask)
	# 		###################################################################

	# 		# Perform forward pass
	# 		model.init_xi()
	# 		outputs = model(inputs)
	# 		###################################################################

	# 		###################################################################
	# 		# Compute loss
	# 		reg = PAC_Bayes_regularizer(model, prior, N, delta, device)
	# 		loss = box_loss_diff_jit(outputs + boxes_3detr, boxes_gt, w1, w2, w3, loss_mask)
	# 		loss = loss + torch.sqrt(reg/2)

	# 		# Compute true (boolean) version of loss for this batch
	# 		loss_true, not_enclosed = box_loss_true(outputs + boxes_3detr, boxes_gt, loss_mask, 0.01)

	# 		###################################################################

	# 		###################################################################
	# 		# Zero the gradients
	# 		optimizer.zero_grad()

	# 		# Perform backward pass
	# 		loss.backward()

	# 		# torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

	# 		# Perform optimization
	# 		optimizer.step()
	# 		###################################################################

	# 		###################################################################
	# 		# Update current loss for this epoch (summing across batches)
	# 		current_loss += loss.item()
	# 		current_loss_true += loss_true.item()
	# 		num_batches += 1
	# 	###################################################################

	# 	###################################################################
	# 	# Print losses (averaged across batches in this epoch)
	# 	print_interval = 1
	# 	if verbose and (epoch % print_interval == 0):
	# 		print("epoch: ", epoch, "; loss: ", '{:02.6f}'.format(current_loss/num_batches),
	# 			  "; loss true: ", '{:02.6f}'.format(current_loss_true / num_batches), end='\r')
	# 	###################################################################

	# ###################################################################
	# # print(' Output BB: ', outputs[:,:,:,:][:,:,None,:])
	# # Process is complete.
	# if verbose:
	# 	print('Training complete.')

	# Save model
	# torch.save(model_cp.state_dict(), "trained_models/perception_model")
	# torch.save(prior.state_dict(), "trained_models/perception_prior")
	# if verbose:
	# 	print('Saved trained model.')
	# 	# print('KL divergence: ', model.calc_kl_div(prior,device))
	# 	# print("Model's state_dict:")
	# 	# for param_tensor in model.state_dict():
	# 	#	  print(param_tensor, "\t", model.state_dict()[param_tensor].size())

	# 	# print("Model weight mean:")	 
	# 	# print(model.linear1.mu, model.linear2.mu)

	# 	# print("Model weight variance:")	 
	# 	# print(model.linear1.logvar, model.linear2.logvar)

	# 	# print("---")
	# 	# print("Optimizer's state_dict:")
	# 	# for var_name in optimizer.state_dict():
	# 	#	  print(var_name, "\t", optimizer.state_dict()[var_name])
	# ###################################################################
	# # ###################################################################

	# # Load model
	# model.load_state_dict(torch.load("trained_models/perception_model"))
	# model.eval()
	# prior.load_state_dict(torch.load("trained_models/perception_prior"))
	# prior.eval()
	# if verbose:
	# 	print('Loaded trained model.')
	# 	# print('KL divergence: ', model.calc_kl_div(prior,device))
	# 	# print("Model's state_dict:")
	# 	# for param_tensor in model.state_dict():
	# 	#	  print(param_tensor, "\t", model.state_dict()[param_tensor].size())

	# 	# print("Model weight mean:")	 
	# 	# print(model.linear1.mu, model.linear2.mu)

	# 	# print("Model weight variance:")	 
	# 	# print(model.linear1.logvar, model.linear2.logvar)

	# 	# print("---")
	# 	# print("Optimizer's state_dict:")
	# 	# for var_name in optimizer.state_dict():
	# 	#	  print(var_name, "\t", optimizer.state_dict()[var_name])
	# ###################################################################

	# print("Evaluating Bound")
	# error_rates = []
	# with torch.no_grad():
	# 	for data in dataloader:
	# 		inputs, targets, loss_mask = data
	# 		inputs = inputs.to(device)
	# 		boxes_3detr = targets["bboxes_3detr"].to(device)
	# 		boxes_gt = targets["bboxes_gt"].to(device)
	# 		loss_mask = loss_mask.to(device)
	# 		for _ in range(num_evaluations):
	# 			model.init_xi()
	# 			outputs = model(inputs)
	# 			error_rate, not_enclosed = box_loss_true(outputs + boxes_3detr, boxes_gt, loss_mask, 0.01)
	# 			error_rates.append(float(error_rate))
	# # ipy.embed()

	# # Computing sample convergence bound
	# avg_error_rate = np.mean(error_rates)
	# print("Average error rate", avg_error_rate)
	# sample_convergence_reg = np.log(1/deltap)/(num_evaluations*N)
	# error_rate_bound = kl_inv_l(avg_error_rate, sample_convergence_reg) if avg_error_rate < 1 else 1
	# print("Bound on the expected error rate for networks sampled from posterior:", error_rate_bound)

	# # Computing kl-inverse PAC-Bayes bound
	# reg = float((model.calc_kl_div(prior, device) + np.log(2*np.sqrt(N)/delta)) / N)
	# pac_bayes_bound = kl_inv_l(error_rate_bound, reg) if error_rate_bound < 1 else 1
	# print("PAC-Bayes guarantee on error rate for new samples", pac_bayes_bound)

	###################################################################
	# ###################################################################
	# # Run test loop
	# # Iterate over the DataLoader for test data
	# mean_loss_pac = 0
	# mean_loss_cp = 0
	# mean_diff_pac = 0
	# mean_diff_cp = 0
	# diff_gt_cp = torch.zeros(len(dataloader_prior)*N_obj,2)
	# diff_gt_pac = torch.zeros(len(dataloader_prior)*N_obj,2)
	# len_gt = torch.zeros(len(dataloader_prior),N_obj, 3)
	# for i, data in enumerate(dataloader_prior, 0):

	# 	###################################################################
	# 	# Get inputs, targets, loss mask
	# 	inputs, targets, loss_mask, finetune = data
	# 	inputs = inputs.to(device)
	# 	boxes_3detr = targets["bboxes_3detr"].to(device)
	# 	boxes_gt = targets["bboxes_gt"].to(device)
	# 	loss_mask = loss_mask.to(device)
	# 	boxes_3detr_all = finetune["bboxes_3detr"].to(device)

	# 	###################################################################
	# 	# print('Ground truth BB: ', boxes_gt[:,:,:,:][:,:,None,:])
	# 	# print('3DETR prediction BB: ',boxes_3detr[:, :, :, :][:,:,None,:])
	# 	# print('Mask: ', loss_mask)
	# 	# ###################################################################
	# 	# # PAC-Bayes prediction
	# 	# model.init_xi()
	# 	outputs = model_cp(inputs)
	# 	# ###################################################################
	# 	# CP prediction
	# 	boxes_cp = torch.zeros_like(boxes_3detr)
	# 	# prior.init_xi()
	# 	# output_cp = model_cp(inputs)
	# 	# if  torch.any(torch.isnan(output_cp)):
	# 	# 		# ipy.embed()
	# 	# 		idx = torch.where(torch.isnan(output_cp))
	# 	# 		output_cp[idx] = 0.0
	# 	# outputs_sorted = torch.clone(output_cp)
	# 	# for ii, jj in np.ndindex((boxes_3detr.shape[0], boxes_3detr.shape[1])):
	# 	# 	for kk in range(boxes_3detr.shape[2]):
	# 	# 		if loss_mask[ii,jj,kk] == 1:
	# 	# 			idx = (boxes_3detr_all[ii,jj,:,:,:] == boxes_3detr[ii,jj,kk,:,:]).nonzero()
	# 	# 			if len(idx) == 6 and all(idx[:,0] == idx[0,0]):
	# 	# 				outputs_sorted[ii,jj,kk,:,:] = outputs[ii,jj,idx[0,0],:,:]
	# 	# 			else:
	# 	# 				print("This should not happen")
	# 	# 				ipy.embed()
	# 	# finetuned_boxes = outputs_sorted[:,:,0:boxes_3detr.shape[2],:,:]
	# 	# boxes_finetune = boxes_3detr + finetuned_boxes
	# 	# if  torch.any(torch.isnan(boxes_finetune)):
	# 	# 		print("This should not happen NANs")
	# 	# 		ipy.embed()
	# 	boxes_finetune = boxes_3detr
	# 	boxes_cp[:,:,:,0,:][:,:,None,:] = torch.min(boxes_finetune[:, :, :, 0, :][:,:,None,:], boxes_finetune[:, :, :, 1, :][:,:,None,:]) - scaling_cp
	# 	boxes_cp[:,:,:,1,:][:,:,None,:] = torch.max(boxes_finetune[:, :, :, 0, :][:,:,None,:], boxes_finetune[:, :, :, 1, :][:,:,None,:])+ scaling_cp
	# 	# print("CP: ", boxes_cp)
	# 	# print("3DETR: ", boxes_3detr)
	# 	# print("PAC: ", outputs + boxes_3detr)
	# 	# print("GT: ", boxes_gt)

	# 	# # Compute true (boolean) version of loss for this batch
	# 	# # print("PAC")
	# 	# loss_true_pac, not_enclosed = box_loss_true(outputs + boxes_3detr, boxes_gt, loss_mask, 0.01)
	# 	# loss_diff_pac = box_loss_diff(outputs + boxes_3detr, boxes_gt, 0.,1.,0., loss_mask)
	# 	# mean_loss_pac = (i*mean_loss_pac + loss_true_pac)/(i+1)
	# 	# mean_diff_pac = (i*mean_diff_pac + loss_diff_pac)/(i+1)
	# 	# print("CP")
	# 	loss_true_cp, not_enclosed = box_loss_true(boxes_cp, boxes_gt, loss_mask, 0.01)
	# 	loss_diff_cp = box_loss_diff(boxes_cp, boxes_gt, 0.,1.,0., loss_mask)
	# 	mean_loss_cp = (i*mean_loss_cp + loss_true_cp)/(i+1)
	# 	mean_diff_cp = (i*mean_diff_cp + loss_diff_cp)/(i+1)
	# 	# print(loss_diff_cp, mean_diff_cp)

	# 	for j in range(N_obj):
	# 		len_gt[i,j,0] = torch.max(boxes_gt[0,0,j,0,0], boxes_gt[0,0,j,1,0]) - torch.min(boxes_gt[0,0,j,0,0],boxes_gt[0,0,j,1,0])
	# 		len_gt[i,j,1] = torch.max(boxes_gt[0,0,j,0,1], boxes_gt[0,0,j,1,1]) - torch.min(boxes_gt[0,0,j,0,1],boxes_gt[0,0,j,1,1])
	# 		len_gt[i,j,2] = torch.max(boxes_gt[0,0,j,0,2], boxes_gt[0,0,j,1,2]) - torch.min(boxes_gt[0,0,j,0,2],boxes_gt[0,0,j,1,2])
		
	# 	# # ipy.embed()
	# 	# # boxes_pac = outputs + boxes_3detr
	# 	# boxes_pac = boxes_3detr
	# 	idx = torch.where(loss_mask[0,:]==1)

	# 	if len(idx[0]) > 0:
	# 		for j in range(N_obj):
	# 			vis = idx[1]==j
	# 			len_pred_x = boxes_cp[:,idx[0][vis],idx[1][vis],1,0] - boxes_cp[:,idx[0][vis],idx[1][vis],0,0]
	# 			# print("Predicted length x: ", len_pred_x, "Ground truth: ", len_gt[i,j,0])
	# 			if len_pred_x.shape[1] > 0:
	# 				diff_gt_cp[i*N_obj+j,0] = torch.mean(len_pred_x)-len_gt[i,j,0]
	# 			# if diff_gt_cp[i*N_obj+j,0] >8:
	# 			# 	diff_gt_cp[i*N_obj+j,0]=8
	# 			len_pred_y = boxes_cp[:,idx[0][vis],idx[1][vis],1,1] - boxes_cp[:,idx[0][vis],idx[1][vis],0,1]
	# 			# print("Predicted length y: ", len_pred_y, "Ground truth: ", len_gt[i,j,0])
	# 			if len_pred_y.shape[1] > 0:
	# 				diff_gt_cp[i*N_obj+j,1] = torch.mean(len_pred_y)-len_gt[i,j,1]
	# 			# if diff_gt_cp[i*N_obj+j,1] >8:
	# 			# 	diff_gt_cp[i*N_obj+j,1]=8

	# 			# len_pred_x = boxes_pac[:,idx[0][vis],idx[1][vis],1,0] - boxes_pac[:,idx[0][vis],idx[1][vis],0,0]
	# 			# diff_gt_pac[i*N_obj+j,0] = torch.mean(len_pred_x)-len_gt[i,j,0]
	# 			# len_pred_y = boxes_pac[:,idx[0][vis],idx[1][vis],1,0] - boxes_pac[:,idx[0][vis],idx[1][vis],0,0]
	# 			# diff_gt_pac[i*N_obj+j,1] = torch.mean(len_pred_y)-len_gt[i,j,1]
	# 			# # diff_gt_pac[i,0] = torch.mean(boxes_pac[:,idx[0],1,0] - boxes_pac[:,idx[0],0,0])-len_gt[i,0]
	# 			# # diff_gt_pac[i,1] = torch.mean(boxes_pac[:,idx[0],1,1] - boxes_pac[:,idx[0],0,1])-len_gt[i,1]
	# 	# else:
	# 	# 	diff_gt_cp[i,0] = 8
	# 	# 	diff_gt_cp[i,1] = 8
	# # Creating histogram


	# # print("Mean loss for PAC-Bayes: ", mean_loss_pac, mean_diff_pac)
	# print("Mean loss for CP: ", mean_loss_cp, mean_diff_cp)
	# # print("Mean loss for finetuned model: ", mean_loss_pac, mean_diff_pac)

	# ipy.embed()
	# # fig, ax = plt.subplots(4, 1, tight_layout=True, sharex=True, sharey=True,)
	# fig, ax = plt.subplots(2, 1, tight_layout=True, sharex=True, sharey=True,)
	# fig.suptitle("Difference between predicted and ground length of boxes")
	# ax[0].set_xlabel("along x-axis using CP+finetuning")
	# ax[0].hist(diff_gt_cp[:,0].detach().cpu().numpy(), bins = 25)
	# ax[1].hist(diff_gt_cp[:,1].detach().cpu().numpy(), bins = 25)
	# ax[1].set_xlabel("along y-axis using CP+finetuning")
	# ax[1].set_ylabel("Frequency (100 environments)")
	# # plt.xlabel("y-axis difference pred and GT length")
	# # Show plot
	# # plt.show()

	# # # fig, ax = plt.subplots(2, 1, tight_layout=True, sharex=True, sharey=True,)
	# # # plt.xlabel("Difference between predicted and ground length of boxes along x/y-axis (upper/lower plot) ")
	# # ax[1].hist(diff_gt_pac[:,0].detach().cpu().numpy(), bins = 25)
	# # ax[1].set_xlabel("along x-axis using 3DETR")
	# # ax[3].hist(diff_gt_pac[:,1].detach().cpu().numpy(), bins = 25)
	# # ax[3].set_xlabel("along y-axis using 3DETR")

	# # plt.ylabel("Frequency (100 environments)")
	# # plt.xlabel("y-axis difference pred and GT length")
	# # Show plot
	# plt.show()


###################################################################

# Run with command line arguments precisely when called directly
# (rather than when imported)
if __name__ == '__main__':
	main() 

