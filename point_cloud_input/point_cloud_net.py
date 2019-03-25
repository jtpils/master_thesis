#from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F

# This is from PointNets github, we should use this on each and every pillar.
# I guess we create 1 instance, and propagate all pillars through it. /A
class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)


    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1,9).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x



# This is the second part, where we use the pseudo-images? /A
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
    For each point, a linear layer is applied followed by BatchNorm and ReLU. This generates a (C=8,P,N) sized tensor.
    Then a max operator is applied over the channels to create an output tensor of size (C=8,P).
    """

    def __init__(self):

        super(PFNLayer, self).__init__()
        self.name = "PFNLayer"

        # Linear Layer:  The linear layer can be formulated as a 1*1 convolution layer across the tensor
        self.conv1 = torch.nn.Conv2d(8, 64, kernel_size=1, stride=1, padding=0)
        # Batch Norm
        self.conv1_bn = torch.nn.BatchNorm2d(64)

    def forward(self, inputs):

        x = F.relu(self.conv1_bn(self.conv1(inputs))) # shape (1,64,1200,100)
        x_max = torch.max(x, dim=3, keepdim=True)[0]  # shape (1,64,1200,1)
        x_max = torch.squeeze(x_max)  # shape (64,12000)

        return x_max


class PointPillarsScatter(nn.Module):
    def __init__(self,
                 output_shape = [188,188]):
        """
        Point Pillar's Scatter.
        Converts learned features from dense tensor to sparse pseudo image.
        :param output_shape: <nd.array(int*4)> Required output shape of features. (Output shape [1,C,H,W]
        C = channels (64), H = height (y-values), W = width (x-values)
        """

        super(PointPillarsScatter).__init__()
        self.name = 'PointPillarsScatter'
        self.output_shape = output_shape
        self.height = output_shape[2]
        self.width = output_shape[3]
        self.num_channels = output_shape[1]

    def forward(self, PFN_output, pillar_tensor, batch_size):

        # batch_canvas will be the final output.
        batch_canvas = []
        for batch_itt in range(batch_size):
            # Create the canvas for this sample
            canvas = torch.zeros(self.num_channels, self.height, self.width)#, dtype=voxel_features.dtype, device=voxel_features.device)

            # Check pillar one in pillar tensor and save the 64 channels in pillar one in PFN output.
            # i [:,0,0] ska alla värden för pillar one vara




            # Only include non-empty pillars
            batch_mask = coords[:, 0] == batch_itt
            this_coords = coords[batch_mask, :]
            indices = this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            voxels = voxel_features[batch_mask, :]
            voxels = voxels.t()

            # Now scatter the blob back to the canvas.
            canvas[:, indices] = voxels

            # Append to a list for later stacking.
            batch_canvas.append(canvas)

        # Stack to 3-dim tensor (batch-size, nchannels, nrows*ncols)
        batch_canvas = torch.stack(batch_canvas, 0)

        # Undo the column stacking to final 4-dim tensor
        batch_canvas = batch_canvas.view(batch_size, self.nchannels, self.ny, self.nx)

        return batch_canvas

