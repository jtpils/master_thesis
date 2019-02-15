from torch.autograd import Variable
import torch.nn.functional as F
import torch
import numpy as np


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
        #self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)   # output size (1, 100, 100) (?)
        self.conv_sweep2 = torch.nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1)  # output size (1, 50, 50) (?)

        # layers for concatenated sweep+cutout
        #self.conv = torch.nn.Conv2d(4, 1, kernel_size=3, stride=1, padding=1)
        #self.fc1 = torch.nn.Linear(18 * 16 * 16, 64)
        #self.fc2 = torch.nn.Linear(18 * 16 * 16, 64)


    def forward(self, input_sweep, input_cutout):
        # propagate the map and the sweep trhough their networks (different networks since they have different sizes)
        cutout = self.conv_cutout(input_cutout)  # output size (1, 300, 300) (?)
        cutout = self.pool(cutout)  # output size (1, 150, 150) (?)
        cutout = self.conv_cutout2(cutout)  # output size (1, 50, 50) (?)
        print('cutout shape: ', np.shape(cutout))

        sweep = self.conv_sweep(input_sweep)  # output size (1, 200, 200) (?)
        sweep = self.pool(sweep)   # output size (1, 100, 100) (?)
        sweep = self.conv_sweep2(sweep)  # output size (1, 50, 50) (?)
        print('sweep shape: ', np.sweep(cutout))

        # concatenate the outputs, make sure they have the same size
        sweep_and_map = torch.cat((sweep.view(sweep.size(0), -1), cutout.view(cutout.size(0), -1)), dim=1)

        # propagate the concatenated inputs and get output with 3 neurons
        #sweep_and_map = self.conv(sweep_and_map)
        #sweep_and_map = self.fc1(sweep_and_map)
        #out = self.fc2(sweep_and_map)


        # Computes the activation of the first convolution
        # Size changes from (3, 32, 32) to (18, 32, 32)
        # x = F.relu(self.conv1(x))

        # Size changes from (18, 32, 32) to (18, 16, 16)
        # x = self.pool(x)

        # Reshape data to input to the input layer of the neural net
        # Size changes from (18, 16, 16) to (1, 4608)
        # Recall that the -1 infers this dimension from the other given dimension
        # x = x.view(-1, 18 * 16 * 16)

        # Computes the activation of the first fully connected layer
        # Size changes from (1, 4608) to (1, 64)
        # x = F.relu(self.fc1(x))

        # Computes the second fully connected layer (activation applied later)
        # Size changes from (1, 64) to (1, 10)
        # x = self.fc2(x)
        #return (x)
