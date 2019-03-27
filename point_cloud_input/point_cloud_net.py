#from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import time
import torch.nn.functional as F
from preprocessing_data_functions import get_grid


'''
def PointPillarsScatter(PFN_input, PFN_output, batch_size):
    
    :param PFN_input: <tensor> size: [number_of_batches, 8, max_number_of_pillars, max_number_of_points]. The feature
            tensor that used as an input to the PFN layer. This is used to find the original location of the pillars
    :param PFN_output: <tensor> size: [number_of_batches, 64, max_number_of_pillars]. The output from the PFN layer.
            Containing the pillars that should be scattered back.
    :param batch_size: <int> size of the batch
    :return: batch_canvas: <list> lsit of canvas from each batch.
    

    num_channels = 64
    height = 282
    width = 282
    pillar_size = 0.16
    x_edges = np.arange(-22, 22, pillar_size)
    y_edges = np.arange(-22, 22, pillar_size)

    batch_canvas = []
    for batch in np.arange(batch_size):

        pillar_list = np.nonzero(PFN_input[batch, 0, :, 0])
        for pillar in pillar_list:
            canvas = torch.zeros((num_channels, height, width))
            x, y = PFN_input[batch, 0, pillar, 0], PFN_input[batch, 1, pillar, 0]
            xgrid, ygrid = get_grid(x, y, x_edges, y_edges)
            pillar_vector = PFN_output[batch, :, pillar]
            pillar_vector = torch.squeeze(pillar_vector)
            canvas[:, ygrid, xgrid] = pillar_vector
        batch_canvas.append(canvas)

    return batch_canvas'''


def fasterScatter(PFN_input, PFN_output, batch_size):

    num_channels = 64
    height = 282
    width = 282
    pillar_size = 0.16
    range = 22

    batch_canvas = []
    for batch in np.arange(batch_size):

        pillar_list = np.nonzero(PFN_input[batch, 0, :, 0]) # ???? List or array??? collect all non-empty pillars
        #pillars = np.resize(PFN_output, (64, 12000))

        x_coords = PFN_input[batch,0,pillar_list,0]  # 1 xcoord from each pillar # should we use the pillar_list here instead of ":" ?
        xgrids = torch.floor((x_coords + range) / pillar_size)
        y_coords = PFN_input[batch,1,pillar_list,0]  # first xvalue in each pillar
        ygrids = torch.floor((y_coords + range) / pillar_size)

        #convert these 2D-indices to 1D indices by declaring canvas as:
        canvas = torch.zeros((num_channels, height*width))
        indices = xgrids*width + ygrids  # new indices along 1D-canvas. or maybe swap x and y here?
        #indices = (height-xgrids)*height -ygrids-1
        indices = torch.squeeze(indices).long()
        #pillar_vector = np.resize(pillars, (64, height*width)) # reshape to 1 row
        pillars_to_canvas = torch.squeeze(PFN_output[batch,:,pillar_list])

        canvas[:, indices] = pillars_to_canvas
        canvas = torch.reshape(canvas, (num_channels, height, width))

        batch_canvas.append(canvas)

    return batch_canvas


'''
class PointPillarsScatter(nn.Module):
    def __init__(self, batch_size, output_shape = [64,188,188]):
        """
        Point Pillar's Scatter.
        Converts learned features from dense tensor to sparse pseudo image.
        :param output_shape: <nd.array(int*4)> Required output shape of features. (Output shape [1,C,H,W]
        C = channels (64), H = height (y-values), W = width (x-values)
        """
        super(PointPillarsScatter, self).__init__()
        #self.name = 'PointPillarsScatter'
        self.batch_size = batch_size
        self.num_channels = output_shape[0]
        self.height = 282  #output_shape[1]
        self.width = 282  #output_shape[2]

    def forward(self, PFN_input, PFN_output):
        pillar_size = 0.16

        x_edges = np.arange(-22, 22, pillar_size)
        y_edges = np.arange(-22, 22, pillar_size)

        batch_canvas = []
        for batch in np.arange(self.batch_size):

            pillar_list = np.nonzero(PFN_input[batch,0,:,0])
            for pillar in pillar_list:
                canvas = torch.zeros((self.num_channels, self.height, self.width))
                x, y = PFN_input[batch,0,pillar,0], PFN_input[batch,1,pillar,0]
                xgrid, ygrid = get_grid(x, y, x_edges, y_edges)
                pillar_vector = PFN_output[batch,:,pillar]
                pillar_vector = torch.squeeze(pillar_vector)
                canvas[:, ygrid, xgrid] = pillar_vector
            batch_canvas.append(canvas)

        # TODO: should we use torch.stack() to get some useful dimensions

        return batch_canvas
'''


class PFNLayer(torch.nn.Module):
    """
    Pillar Feature Net Layer (PFN) (The PFN could be composed of a series of these layers)
    Ïn the PointPillar paper a single layer of PFN is used.
    For each point, a linear layer is applied followed by BatchNorm and ReLU. This generates a (C=8,P,N) sized tensor.
    Then a max operator is applied over the channels to create an output tensor of size (C=8,P).
    """

    def __init__(self):

        super(PFNLayer, self).__init__()

        # Linear Layer:  The linear layer can be formulated as a 1*1 convolution layer across the tensor
        self.conv1 = torch.nn.Conv2d(8, 64, kernel_size=1, stride=1, padding=0)
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
        Backbone. output a rigid transformation.
        """
        super(Backbone, self).__init__()
        # input : [128, 282, 282]
        self.conv1 = torch.nn.Conv2d(128, 64, kernel_size=5, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(64, 16, kernel_size=4, stride=1, padding=0)

        self.conv1_bn = torch.nn.BatchNorm2d(64)
        self.conv2_bn = torch.nn.BatchNorm2d(16)

        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.fc1 = torch.nn.Linear(16 * 68 * 68, 32)
        self.fc1_bn = torch.nn.BatchNorm1d(32)
        self.fc_out = torch.nn.Linear(32, 3)

    def forward(self, x):

        x = F.relu(self.conv1_bn(self.conv1(x)))  # 64, 278, 278
        x = self.pool2(x) # 64, 139, 139

        x = F.relu(self.conv2_bn(self.conv2(x)))  # 16, 136, 136
        x = self.pool2(x)  # 16, 68, 68

        x = x.view(-1, 16 * 68 * 68)
        x = torch.tanh(self.fc1_bn(self.fc1(x)))
        x = self.fc_out(x)

        return x


class PointPillars(torch.nn.Module):
    """
    The whole damn net.
    """

    def __init__(self, batch_size):
        super(PointPillars, self).__init__()
        self.batch_size = batch_size
        self.PFNlayer_sweep = PFNLayer()
        self.PFNlayer_map = PFNLayer()
        self.Backbone = Backbone()

    def forward(self, sweep, map):

        sweep_outputs = self.PFNlayer_sweep.forward(sweep)
        map_outputs = self.PFNlayer_map.forward(map)

        # sweep_canvas = PointPillarsScatter.forward(sweep, sweep_outputs)
        # map_canvas = PointPillarsScatter.forward(map, map_outputs)

        #t1 = time.time()
        #sweep_canvas = PointPillarsScatter(sweep, sweep_outputs, self.batch_size)
        #t2 = time.time()
        #map_canvas = PointPillarsScatter(map, map_outputs, self.batch_size)
        #t3 = time.time()
        t1 = time.time()
        sweep_canvas = fasterScatter(sweep, sweep_outputs, self.batch_size)
        t2 = time.time()
        map_canvas = fasterScatter(map, map_outputs, self.batch_size)
        t3 = time.time()

        zipped_canvas = list(zip(sweep_canvas,map_canvas))
        t4 = time.time()
        concatenated_canvas = torch.zeros(self.batch_size, 128, 282, 282)

        print('Time sweep_canvas:', t2 - t1)
        print('Time map_canvas:', t3 - t2)
        print('Time zipped_canvas:', t4 - t3)

        for i in np.arange(self.batch_size):
            sweep_layers = zipped_canvas[i][0]
            map_layers = zipped_canvas[i][1]
            concatenated_layers = torch.cat((sweep_layers , map_layers ), 0)

            concatenated_canvas[i, :, :, :] = concatenated_layers


        output = self.Backbone.forward(concatenated_canvas)

        del sweep_canvas, map_canvas, zipped_canvas, concatenated_layers, sweep_outputs, map_outputs
        return output


