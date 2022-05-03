import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPModel(nn.Module):
    def __init__(self, num_in, num_out):
        super(MLPSoftmax, self).__init__()

        self.linear1 = nn.Linear(num_in, 50) 
        self.linear2 = nn.Linear(50, num_out) # num_out

    def forward(self, x):

        # obs_softmax = F.softmax(x, dim=1)

        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x) # softmax(x, dim=1)

        return x