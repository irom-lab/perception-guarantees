import torch
import torch.nn as nn
import torch.nn.functional as F
import IPython as ipy
import numpy as np


class MLPModel(nn.Module):
    def __init__(self, num_in, num_out):
        super(MLPModel, self).__init__()

        self.linear1 = nn.Linear(num_in, 50)
        self.linear2 = nn.Linear(50, np.prod(num_out))
        self.num_out = num_out

    def forward(self, x):
        '''
        input x: (batch_size, num_locations, feature_dims0, feature_dims1)
        return:  (batch_size, num_locations, num_out)
        '''

        # Reshape input
        x = x.view((x.shape[0], x.shape[1], x.shape[2]*x.shape[3]))

        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        # x = F.relu(x)

        x = x.view((x.shape[0], x.shape[1], self.num_out[0], self.num_out[1]))

        return x