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

        self.conv1 = nn.Conv2d(8, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(8, 32, kernel_size=3, stride=1, padding=1)
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


#net = PointCloudNet()
#print(net)

