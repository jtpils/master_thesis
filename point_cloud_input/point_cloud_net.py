import torch.nn.functional as F
import torch
import numpy as np


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

net = PointCloudNet()
print(net)