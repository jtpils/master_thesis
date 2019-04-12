#from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import time
import torch.nn.functional as F


def ScatterPseudoImage(coordinates, PFN_output,  batch_size, use_cuda):

    num_channels = 64
    height = 60
    width = 60
    pillar_size = 0.5
    range = 15
    batch_size = np.shape(coordinates)[0] ####### THE BATCH IS DIVIDED BETWEEN THE GPUS, so each gpu will not have a batchsize, but rather half the batch size.
    batch_canvas = []
    for batch in np.arange(batch_size):
        # Find all nonzero elements in the coordinate tensor
        print('coordinates: ', np.shape(coordinates))
        print('batch ', batch)
        print('batch size ', batch_size)
        pillar_list = np.nonzero(coordinates[batch, :, 0])

        x_coords = coordinates[batch, pillar_list, 0]
        xgrids = torch.floor((x_coords + range) / pillar_size)
        y_coords = coordinates[batch, pillar_list, 1]
        ygrids = torch.floor((y_coords + range) / pillar_size)

        #convert these 2D-indices to 1D indices by declaring canvas as:
        if use_cuda:
            canvas = torch.zeros((num_channels, height*width)).cuda()
        else:
            canvas = torch.zeros((num_channels, height*width))
        indices = xgrids*width + ygrids
        indices = torch.squeeze(indices).long()
        pillars_to_canvas = torch.squeeze(PFN_output[batch,:,pillar_list])

        canvas[:, indices] = pillars_to_canvas
        canvas = torch.reshape(canvas, (num_channels, height, width))

        batch_canvas.append(canvas)

    return batch_canvas


class PFNLayer(torch.nn.Module):

    def __init__(self):

        super(PFNLayer, self).__init__()

        # Linear Layer:  The linear layer can be formulated as a 1*1 convolution layer across the tensor
        # Changed the input to 3 channels
        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=1, stride=1, padding=0)
        # Batch Norm
        self.conv1_bn = torch.nn.BatchNorm2d(64)

    def forward(self, inputs):

        x = F.relu(self.conv1_bn(self.conv1(inputs))) # shape (1,64,1200,100)
        x = torch.max(x, dim=3, keepdim=True)[0]  # shape (1,64,1200,1)
        x = x.view(np.shape(x)[:3])
        return x


class Backbone(nn.Module):
    def __init__(self):
        """
        Backbone. outputs a rigid transformation.
        """
        super(Backbone, self).__init__()

        self.conv1 = torch.nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        self.conv4 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)

        self.bn1 = torch.nn.BatchNorm2d(64)
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.bn3 = torch.nn.BatchNorm2d(64)
        self.bn4 = torch.nn.BatchNorm2d(64)

        # Block 2:
        # relu + 6 conv + stride 2 + bn
        self.conv5 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=0)
        self.conv6 = torch.nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0)
        self.conv7 = torch.nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=0)
        self.conv8 = torch.nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=0)
        self.conv9 = torch.nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=0)
        self.conv10 = torch.nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=0)

        self.bn5 = torch.nn.BatchNorm2d(128)
        self.bn6 = torch.nn.BatchNorm2d(128)
        self.bn7 = torch.nn.BatchNorm2d(64)
        self.bn8 = torch.nn.BatchNorm2d(32)
        self.bn9 = torch.nn.BatchNorm2d(16)
        self.bn10 = torch.nn.BatchNorm2d(8)

        # Block 3:
        # 2 fully connected with drop out.

        self.fc1 = torch.nn.Linear( 8 * 59 * 59, 32)
        self.fc1_bn = torch.nn.BatchNorm1d(32)
        self.fc_out = torch.nn.Linear(32, 3)

    def forward(self, x):

        # Block 1:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))

        # Block 2:
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu(self.bn8(self.conv8(x)))
        x = F.relu(self.bn9(self.conv9(x)))
        x = F.relu(self.bn10(self.conv10(x)))

        # Block 3:
        x = x.view(-1, 8 * 59 * 59)
        x = torch.tanh(self.fc1_bn(self.fc1(x)))
        x = self.fc_out(x)

        return x


class OurPointPillars(torch.nn.Module):
    """
    The whole damn net.
    """

    def __init__(self, batch_size, use_cuda):
        super(OurPointPillars, self).__init__()
        self.batch_size = batch_size
        self.PFNlayer_sweep = PFNLayer()
        self.PFNlayer_map = PFNLayer()
        self.Backbone = Backbone()
        self.use_cuda = use_cuda

    def forward(self, sweep, map, sweep_coordinates, map_coordinates):
        print('shape sweep', np.shape(sweep))
        print('shape sweep coords', np.shape(sweep_coordinates))
        sweep_outputs = self.PFNlayer_sweep.forward(sweep)
        map_outputs = self.PFNlayer_map.forward(map)
        sweep_canvas = ScatterPseudoImage(sweep_coordinates, sweep_outputs, self.batch_size, self.use_cuda)
        map_canvas = ScatterPseudoImage(map_coordinates, map_outputs, self.batch_size, self.use_cuda)
        zipped_canvas = list(zip(sweep_canvas,map_canvas))
        concatenated_canvas = torch.zeros(self.batch_size, 128, 60, 60)

        batch_size = np.shape(sweep_coordinates)[0]
        for i in np.arange(self.batch_size):
            sweep_layers = zipped_canvas[i][0]
            map_layers = zipped_canvas[i][1]
            concatenated_layers = torch.cat((sweep_layers , map_layers ), 0)

            concatenated_canvas[i, :, :, :] = concatenated_layers
        if self.use_cuda:
            output = self.Backbone.forward(concatenated_canvas.cuda())
        else:
            output = self.Backbone.forward(concatenated_canvas)

        del sweep_canvas, map_canvas, zipped_canvas, concatenated_layers, sweep_outputs, map_outputs, sweep, map
        return output

    def name(self):
        return "OurPointPillars"


