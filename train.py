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
from loss_fn import * # box_loss_tensor
# import loss_fn

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
	batch_size = 5

	params = {'batch_size': batch_size,
				'shuffle': True}
	           # 'num_workers': 12}
	dataloader = DataLoader(dataset, **params)
	###################################################################

	###################################################################
	# Device
	# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	device = torch.device('cuda') # cpu'
	###################################################################

	###################################################################
	# Initialize NN model
	num_in = dataset.feature_dims[0]*dataset.feature_dims[1]
	num_out = (2,3)
	model = MLPModel(num_in, num_out)
	model.to(device)
	###################################################################


	###################################################################
	# Define the loss function
	# loss_function = # TODO
	###################################################################

	###################################################################
	# Define optimizer
	optimizer = torch.optim.Adam(model.parameters(), lr=1e-3) # , weight_decay=1e-5)
	###################################################################

	# Run the training loop
	num_epochs = 5000 
	for epoch in range(0, num_epochs):

		current_loss = 0.0
		num_batches = 0

		# Iterate over the DataLoader for training data
		for i, data in enumerate(dataloader, 0):

			# Get inputs
			inputs, targets = data
			inputs = inputs.to(device)
			boxes_3detr = targets["bboxes_3detr"].to(device)
			boxes_gt = targets["bboxes_gt"].to(device)


			# Zero the gradients
			optimizer.zero_grad()


			# Perform forward pass
			outputs = model(inputs)

			# Compute loss
			# loss = box_loss_tensor_jit(outputs + boxes_3detr, boxes_gt, torch.tensor(1).to(device), torch.tensor(1).to(device), torch.tensor(1).to(device))

			ipy.embed() # ll = box_loss_tensor(boxes_gt+0.1, boxes_gt, 1, 1, 1)

			# Perform backward pass
			loss.backward()

			# Perform optimization
			optimizer.step()

			# Update current loss
			current_loss += loss.item()
			num_batches += 1

	    # Print 
		if verbose and (epoch % 10 == 0):
			print("epoch: ", epoch, "; loss: ", current_loss/num_batches)

	# Process is complete.
	if verbose:
	    print('Training complete.')

	# Save model
	torch.save(model.state_dict(), "trained_models/perception_model")
	if verbose:
	    print('Saved trained model.')


#################################################################

# Run with command line arguments precisely when called directly
# (rather than when imported)
if __name__ == '__main__':
    main() 

