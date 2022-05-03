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

	params = {'batch_size': 5,
				'shuffle': True}
	           # 'num_workers': 12}
	dataloader = DataLoader(dataset, **params)
	###################################################################

	###################################################################
	# Device
	# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	device = 'cpu'
	###################################################################

	###################################################################
	# Initialize NN model
	ipy.embed()
	num_in = num_rays
	num_out = num_primitives
	model = MLPModel(num_in, num_out)
	# model.to(device)
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
	        # inputs, targets = inputs.to(device), targets.to(device)

	        # Zero the gradients
	        optimizer.zero_grad()

	        ipy.embed()

	        # Perform forward pass
	        outputs = model(inputs)

	        # Compute loss
	        loss = loss_function(outputs, targets)

	        # Perform backward pass
	        loss.backward()

	        # Perform optimization
	        optimizer.step()

	        # Update current loss
	        current_loss += loss.item()
	        num_batches += 1

	    # Print 
	    if verbose and (epoch % 1000 == 0):
	        print("epoch: ", epoch, "; loss: ", current_loss/num_batches)

	# Process is complete.
	if verbose:
	    print('Training complete.')

	# Save model
	torch.save(model.state_dict(), "lbi/training/models/trained_model")
	if verbose:
	    print('Saved trained model.')


#################################################################

# Run with command line arguments precisely when called directly
# (rather than when imported)
if __name__ == '__main__':
    main() 
