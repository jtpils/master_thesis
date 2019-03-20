from torch.autograd import Variable
import torch.nn.functional as F
import torch
import numpy as np


class FirstBEVNet(torch.nn.Module):
    # The first network is inspired by Luca's article.

    # Our batch shape for input x is (8, 900, 900)

    def __init__(self):
        super(FirstBEVNet, self).__init__()

        # Encoder:
        # conv1, conv2, pool, conv3, pool
        # Input channels = 8, output channels = 32
        self.conv1 = torch.nn.Conv2d(8, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)

        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=3, stride=3, padding=0)


        # Context Module : conv4,conv5,conv6
        self.conv4 = torch.nn.Conv2d(32, 128, kernel_size=3, stride=1, padding=1)
        self.conv5 = torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv6 = torch.nn.Conv2d(128, 32, kernel_size=1, stride=1, padding=0)

        # FC-layers
        # 32 * 125 * 125
        self.fc1 = torch.nn.Linear(32 * 75 * 75, 1024)
        self.fc2 = torch.nn.Linear(1024, 512)
        self.fc3 = torch.nn.Linear(512, 3)

    def forward(self, x):

        # ENCODER
        # Size changes from (8, 900, 900) to (32, 900, 900)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # Size changes from (32, 900, 900) to (32, 450, 450)
        x = self.pool1(x)
        x = F.relu(self.conv3(x))
        # Size changes from (32, 450, 450) to (32, 225, 225)
        x = self.pool1(x)

        # CONTEXT MODULE
        # Size changes from (32, 225, 225) to (128, 225, 225)
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        # Size changes from (128, 225, 225) to (32, 225, 225)
        x = F.relu(self.conv6(x))

        # Size changes from (32, 225, 225) to (32, 75, 75)
        x = self.pool2(x)
        # "Flatten layer"
        # change size from (32, 75, 75) to (1, 180000)
        x = x.view(-1, 32 * 75 * 75)

        # FC-CONNECTED
        # Change size from (1, 180000) to (1, 1024)
        x = F.relu(self.fc1(x))
        # Change size from (1, 1024) to (1, 512)
        x = F.relu(self.fc2(x))
        # Change size from (1, 512)  to (1, 3)
        x = F.relu(self.fc3(x))

        return x


class SuperSimpleCNN(torch.nn.Module):

    def __init__(self):
        super(SuperSimpleCNN, self).__init__()

        # Input channels = 8, output channels = 8
        self.conv1 = torch.nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1)
        #self.conv1_bn = torch.nn.BatchNorm2d(8)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv2 = torch.nn.Conv2d(8, 4, kernel_size=3, stride=3, padding=0)
        #self.conv2_bn = torch.nn.BatchNorm2d(4)

        #self.conv3 = torch.nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1)
        #self.conv3_bn = torch.nn.BatchNorm2d(4)
        #self.pool3 = torch.nn.MaxPool2d(kernel_size=3, stride=3, padding=0)

        # FC-layers
        # 4 * 50 * 50

        #dim1 = 512
        #dim2 = 256
        #dim3 = 128
        self.fc1 = torch.nn.Linear(4 * 75 * 75, 64)
        #self.fc1_bn = torch.nn.BatchNorm1d(dim1)


        #self.fc2 = torch.nn.Linear(dim1, dim2)
        #self.fc2_bn = torch.nn.BatchNorm1d(dim2)

        #self.fc3 = torch.nn.Linear(dim2, dim3)
        #self.fc3_bn = torch.nn.BatchNorm1d(dim3)

        self.fc_out = torch.nn.Linear(64, 3)

    def forward(self, x):
        # x = F.relu(self.conv1_bn(self.conv1(x)))  # Size changes from (8, 900, 900) to (8, 900, 900)
        # x = self.pool1(x)  # Size changes from (8, 900, 900) to (8, 450, 450)

        #x = F.relu(self.conv2_bn(self.conv2(x)))  # Size changes from  (8, 450, 450) to (4, 150, 150)
        #x = self.pool1(x)  # Size changes from (8, 900, 900) to (8, 450, 450)

        x = F.relu(self.conv1(x))  # Size changes from (8, 900, 900) to (8, 900, 900)
        x = self.pool1(x)  # Size changes from (8, 900, 900) to (8, 450, 450)

        x = F.relu(self.conv2(x)) # Size changes from  (8, 450, 450) to (4, 150, 150)
        x = self.pool1(x)  # Size changes from (8, 900, 900) to (8, 450, 450)


        #x = F.relu(self.conv3_bn(self.conv3(x)))  # Size changes from  (4, 150, 150) to (4, 150, 150)
        #x = self.pool3(x)  # Size changes from (4, 150, 150) to (4, 50, 50)

        # FC-CONNECTED
        # "Flatten layer"
        x = x.view(-1, 4 * 75 * 75)  # change size from (4, 50, 50) to (1, 1000)
        x = torch.tanh(self.fc1(x))  # Change size from (1, 1000) to (1, 512)
        #x = torch.tanh(self.fc2_bn(self.fc2(x)))  # Change size from (1, 512) to (1, 256)
        #x = torch.tanh(self.fc3_bn(self.fc3(x)))  # Change size from (1, 256) to (1, 128)
        x = self.fc_out(x)  # Change size from (1, 128) to (1, 3)

        return x
