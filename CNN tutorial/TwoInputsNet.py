from torch.autograd import Variable
import torch.nn.functional as F
import torch
import numpy as np
from functions import *


class TwoInputsNet(torch.nn.Module):

    def __init__(self):
        super(TwoInputsNet, self).__init__()

        # Input map cutout, size (4, 900, 900)
        # Input channels = 4, output channels = 1
        self.conv_cutout = torch.nn.Conv2d(4, 1, kernel_size=3, stride=3, padding=1)  # output size (1, 300, 300) (?)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)   # output size (1, 150, 150) (?)
        self.conv_cutout2 = torch.nn.Conv2d(1, 1, kernel_size=3, stride=3, padding=1)  # output size (1, 50, 50) (?)

        # Input sweep, size (4, 600, 600)
        # Input channels = 4, output channels = 1
        self.conv_sweep = torch.nn.Conv2d(4, 1, kernel_size=3, stride=3, padding=1)  # output size (1, 200, 200) (?)
        self.conv_sweep2 = torch.nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1)  # output size (1, 50, 50) (?)

        # layers for concatenated sweep+cutout
        self.conv = torch.nn.Conv2d(2, 1, kernel_size=5, stride=2, padding=2)

        self.fc1 = torch.nn.Linear(1 * 25 * 25, 64)
        self.fc2 = torch.nn.Linear(64, 3)

    def forward(self, input_sweep, input_cutout):
        # propagate the map and the sweep trhough their networks (different networks since they have different sizes)
        cutout = F.relu(self.conv_cutout(input_cutout))  # output size (1, 300, 300) (?)
        cutout = self.pool(cutout)  # output size (1, 150, 150) (?)
        cutout = F.relu(self.conv_cutout2(cutout))  # output size (1, 50, 50) (?)

        sweep = F.relu(self.conv_sweep(input_sweep))  # output size (1, 200, 200) (?)
        sweep = self.pool(sweep)   # output size (1, 100, 100) (?)
        sweep = F.relu(self.conv_sweep2(sweep))  # output size (1, 50, 50) (?)

        # concatenate the outputs, make sure they have the same size

        sweep_and_map = torch.cat((sweep, cutout), dim=1)
        # propagate the concatenated inputs and get output with 3 neurons
        sweep_and_map = F.relu(self.conv(sweep_and_map))
        sweep_and_map = sweep_and_map.view(-1, 1 * 25 * 25)
        sweep_and_map = self.fc1(sweep_and_map)
        out = self.fc2(sweep_and_map)

        return out
