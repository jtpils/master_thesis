#from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from preprocessing_data_functions import get_grid


class PFNLayer(torch.nn.Module):
    """
    Pillar Feature Net Layer (PFN) (The PFN could be composed of a series of these layers)
    Ïn the PointPillar paper a single layer of PFN is used.
    For each point, a linear layer is applied followed by BatchNorm and ReLU. This generates a (C=8,P,N) sized tensor.
    Then a max operator is applied over the channels to create an output tensor of size (C=8,P).
    """

    def __init__(self):

        super(PFNLayer, self).__init__()
        #self.name = "PFNLayer"

        # Linear Layer:  The linear layer can be formulated as a 1*1 convolution layer across the tensor
        self.conv1 = torch.nn.Conv2d(8, 64, kernel_size=1, stride=1, padding=0)
        # Batch Norm
        self.conv1_bn = torch.nn.BatchNorm2d(64)

    def forward(self, inputs):

        x = F.relu(self.conv1_bn(self.conv1(inputs))) # shape (1,64,1200,100)
        x = torch.max(x, dim=3, keepdim=True)[0]  # shape (1,64,1200,1)
        x = x.view(np.shape(x)[:3])
        return x


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


class Backbone(nn.Module):
    def __init__(self):
        """
        Backbone. output a rigid transformation.
        """
        super(Backbone, self).__init__()
        #self.name = 'Backbone'
        self.conv1 = torch.nn.Conv2d(8, 64, kernel_size=1, stride=1, padding=0)

    def forward(self, inputs):

        outputs = self.conv1(inputs)

        return outputs


class PointPillars(torch.nn.Module):
    """
    The whole damn net.
    """

    def __init__(self, batch_size):
        super(PointPillars, self).__init__()
        #self.name = "PointPillars"
        self.batch_size = batch_size
        self.PFNlayer_sweep = PFNLayer()
        self.PFNlayer_map = PFNLayer()
        #self.PointPillarsScatter = PointPillarsScatter(self.batch_size)
        self.Backbone = Backbone()

    def forward(self, sweep, map, PointPillarsScatter):

        sweep_outputs = self.PFNlayer_sweep.forward(sweep)
        map_outputs = self.PFNlayer_map.forward(map)

        sweep_canvas = PointPillarsScatter.forward(sweep, sweep_outputs)
        map_canvas = PointPillarsScatter.forward(map, map_outputs)

        # concatenate the canvases, one sample should be a sweep+map
        samples = 1  #concatenate...

        output = self.Backbone.forward(samples)

        return output


