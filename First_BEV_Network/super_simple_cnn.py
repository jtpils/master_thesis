from torch.autograd import Variable
import torch.nn.functional as F
import torch
import numpy as np


class SuperSimpleCNN(torch.nn.Module):

    def __init__(self):
        super(SuperSimpleCNN, self).__init__()

        # Input channels = 8, output channels = 8
        self.conv1 = torch.nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1)
        self.conv1_bn = torch.nn.BatchNorm2d(8)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv2 = torch.nn.Conv2d(8, 4, kernel_size=3, stride=3, padding=0)
        self.conv2_bn = torch.nn.BatchNorm2d(4)


        self.conv3 = torch.nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1)
        self.conv3_bn = torch.nn.BatchNorm2d(4)
        self.pool3 = torch.nn.MaxPool2d(kernel_size=3, stride=3, padding=0)

        # FC-layers
        # 4 * 50 * 50
        self.fc1 = torch.nn.Linear(4 * 50 * 50, 128)
        self.fc1_bn = torch.nn.BatchNorm1d(128)
        self.fc2 = torch.nn.Linear(128, 3)

    def forward(self, x):
        x = F.relu(self.conv1_bn(self.conv1(x)))  # Size changes from (8, 900, 900) to (8, 900, 900)
        x = self.pool1(x)  # Size changes from (8, 900, 900) to (8, 450, 450)

        x = F.relu(self.conv2_bn(self.conv2(x)))  # Size changes from  (8, 450, 450) to (4, 150, 150)
        x = F.relu(self.conv3_bn(self.conv3(x)))  # Size changes from  (4, 150, 150) to (4, 150, 150)
        x = self.pool3(x)  # Size changes from (4, 150, 150) to (4, 50, 50)

        # FC-CONNECTED
        # "Flatten layer"       
        x = x.view(-1, 4 * 50 * 50)  # change size from (4, 50, 50) to (1, 1000)
        x = torch.tanh(self.fc1_bn(self.fc1(x)))  # Change size from (1, 1000) to (1, 128)
        x = self.fc2(x)  # Change size from (1, 128) to (1, 3)

        return x
