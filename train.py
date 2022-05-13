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
from loss_fn import *

def main(raw_args=None):


	###################################################################
	# Parse arguments
	parser = argparse.ArgumentParser()
	parser.add_argument("--verbose", type=int, default=1, help="print more (default: 1")

	args = parser.parse_args(raw_args)
	verbose = args.verbose
	###################################################################

	###################################################################
	# Initialize dataset and dataloader
	dataset = PointCloudDataset("features.pt", "bbox_labels.pt")
	batch_size = 10

	params = {'batch_size': batch_size,
				'shuffle': False}
	           # 'num_workers': 12}
	dataloader = DataLoader(dataset, **params)
	###################################################################

	###################################################################
	# Device
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	# device = torch.device('cpu')
	###################################################################

	###################################################################
	# Initialize NN model
	num_in = dataset.feature_dims[0]*dataset.feature_dims[1]
	num_out = (2,3) # bbox corner representation
	model = MLPModel(num_in, num_out)
	model.to(device)
	###################################################################

	###################################################################
	# Define optimizer
	optimizer = torch.optim.Adam(model.parameters(), lr=1e-4) # , weight_decay=1e-5)
	###################################################################

	###################################################################
	# Choose loss weights
	w1 = torch.tensor(1.0).to(device)
	w2 = torch.tensor(0.2).to(device)
	w3 = torch.tensor(1.0).to(device)

	# Load loss mask
	loss_mask = torch.load("loss_mask.pt")
	loss_mask = loss_mask.to(device)
	###################################################################

	# Run the training loop
	num_epochs = 1000
	for epoch in range(0, num_epochs):

		###################################################################
		# Initialize running losses for this epoch
		current_loss = 0.0
		current_loss_true = 0.0
		num_batches = 0
		###################################################################

		###################################################################
		# Iterate over the DataLoader for training data
		for i, data in enumerate(dataloader, 0):

			###################################################################
			# Get inputs
			inputs, targets = data
			inputs = inputs.to(device)
			boxes_3detr = targets["bboxes_3detr"].to(device)
			boxes_gt = targets["bboxes_gt"].to(device)
			###################################################################

			###################################################################
			# Zero the gradients
			optimizer.zero_grad()

			# Perform forward pass
			outputs = model(inputs)
			###################################################################

			###################################################################
			# Compute loss
			loss = box_loss_diff_jit(outputs + boxes_3detr, boxes_gt, w1, w2, w3, loss_mask)

			# Compute true (boolean) version of loss for this batch
			loss_true, not_enclosed = box_loss_true(outputs + boxes_3detr, boxes_gt, loss_mask, 0.01)
			###################################################################

			###################################################################
			# Perform backward pass
			loss.backward()

			# Perform optimization
			optimizer.step()
			###################################################################

			###################################################################
			# Update current loss for this epoch (summing across batches)
			current_loss += loss.item()
			current_loss_true += loss_true.item()
			num_batches += 1
		###################################################################

		###################################################################
	    # Print losses (averaged across batches in this epoch)
		print_interval = 1
		if verbose and (epoch % print_interval == 0):
			print("epoch: ", epoch, "; loss: ", '{:02.6f}'.format(current_loss/num_batches),
				  "; loss true: ", '{:02.6f}'.format(current_loss_true / num_batches), end='\r')
		###################################################################

	###################################################################
	# Process is complete.
	if verbose:
	    print('Training complete.')

	# Save model
	torch.save(model.state_dict(), "trained_models/perception_model")
	if verbose:
	    print('Saved trained model.')
	###################################################################

	ipy.embed()


#################################################################

# Run with command line arguments precisely when called directly
# (rather than when imported)
if __name__ == '__main__':
    main() 

