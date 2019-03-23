import torch.nn.functional as F
import torch
import numpy as np


class PointCloudNet(torch.nn.Module):

    def __init__(self):
        super(PointCloudNet, self).__init__()

        self.conv1 = torch.nn.Conv2d(8, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 32, kernel_size=3, stride=1, padding=1)
        # self.conv3 = nn.Conv2d()

        # self.fc1 = nn.Linear()
        # self.fc2 = nn.Linear()
        # self.fc3 = nn.Linear()

    def forward(self,input1, input2):
        c = self.conv1(input1)
        f = self.conv2(input2)

        # reshape 'c' and 'f' to 2D and concat them
        combined = torch.cat((c.view(c.size(0), -1), f.view(f.size(0), -1)), dim=1)

        out = self.fc2(combined)


# TODO: THIS IS NOT TESTED!
class PFNLayer(torch.nn.Module):
    """
    Pillar Feature Net Layer (PFN) (The PFN could be composed of a series of these layers)
    Ïn the PointPillar paper a single layer of PFN is used.
    For each point, a linear layer is applied followed by BatchNorm and ReLU. This generates a (C,P,N) sized tensor.
    Then a max operator is applied over the channels to create an output tensor of size (C,P).
    :param in_channels: <int>. Number of input channels.
    :param out_channels: <int>. Number of output channels. (default 64)
    :param use_norm: <bool>. Whether to include BatchNorm.
    :param last_layer: <bool>. If last_layer, there is no concatenation of features.
    """

    def __init__(self):

        super(PFNLayer, self).__init__()

        # Linear Layer:  The linear layer can be formulated as a 1*1 convolution layer across the tensor
        self.conv1 = torch.nn.Conv2d(8, 64, kernel_size=1, stride=1, padding=0)
        # Batch Norm
        self.conv1_bn = torch.nn.BatchNorm2d(64)

    def forward(self, inputs):

        x = F.relu(self.conv1_bn(self.conv1(inputs)))
        x_max = torch.max(x, dim=3, keepdim=True)[0]

        return x_max

    def name(self):
        return "PFNLayer"