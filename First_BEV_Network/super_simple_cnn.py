from torch.autograd import Variable
import torch.nn.functional as F
import torch
import numpy as np


class SuperSimpleCNN(torch.nn.Module):

    def __init__(self):
        super(SuperSimpleCNN, self).__init__()

        self.conv1 = torch.nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv2 = torch.nn.Conv2d(8, 4, kernel_size=3, stride=3, padding=0)

        self.fc1 = torch.nn.Linear(4 * 75 * 75, 32)
        self.fc_out = torch.nn.Linear(32, 3)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # Size changes from (8, 900, 900) to (8, 900, 900)
        x = self.pool1(x)  # Size changes from (8, 900, 900) to (8, 450, 450)

        x = F.relu(self.conv2(x)) # Size changes from  (8, 450, 450) to (4, 150, 150)
        x = self.pool1(x)  # Size changes from (8, 900, 900) to (8, 450, 450)

        # FC-CONNECTED
        x = x.view(-1, 4 * 75 * 75)  # change size from (4, 50, 50) to (1, 1000)
        x = torch.tanh(self.fc1(x))  # Change size from (1, 1000) to (1, 512)
        x = self.fc_out(x)  # Change size from (1, 128) to (1, 3)

        return x


class MoreConv(torch.nn.Module):

    def __init__(self):
        super(MoreConv, self).__init__()
        self.conv1 = torch.nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(8, 4, kernel_size=3, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(4, 1, kernel_size=3, stride=1, padding=1)

        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.pool3 = torch.nn.MaxPool2d(kernel_size=3, stride=3, padding=0)

        self.fc1 = torch.nn.Linear(1*25*25, 16)
        self.fc_out = torch.nn.Linear(16, 3)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # Size changes from (8, 900, 900) to (16, 900, 900)
        x = self.pool1(x)  # Size changes from (16, 900, 900) to (16, 450, 450)

        x = F.relu(self.conv2(x)) # Size changes from  (16, 450, 450) to (16, 450, 450)
        x = self.pool3(x)  # Size changes from (16, 450, 450) to (16, 150, 150)

        x = F.relu(self.conv3(x))  # Size changes from  (16, 150, 150) to (4, 150, 150)
        x = self.pool3(x)  # Size changes from (4, 150, 150) to (4, 50, 50)

        x = F.relu(self.conv4(x))  # Size changes from  (4, 50, 50) to (1, 50, 50)
        x = self.pool1(x)  # Size changes from (1, 50, 50) to (1, 25, 25)


        # FC-CONNECTED
        x = x.view(-1, 1*25*25)
        x = torch.tanh(self.fc1(x))
        x = self.fc_out(x)

        return x


class MoreConvFC(torch.nn.Module):

    def __init__(self):
        super(MoreConvFC, self).__init__()
        self.conv1 = torch.nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1)
        self.conv3 = torch.nn.Conv2d(8, 4, kernel_size=3, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(4, 1, kernel_size=3, stride=1, padding=1)

        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.pool3 = torch.nn.MaxPool2d(kernel_size=3, stride=3, padding=0)

        self.fc1 = torch.nn.Linear(1*25*25, 1024)
        self.fc2 = torch.nn.Linear(1024, 512)
        self.fc3 = torch.nn.Linear(512, 128)
        self.fc_out = torch.nn.Linear(128, 3)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # Size changes from (8, 900, 900) to (16, 900, 900)
        x = self.pool1(x)  # Size changes from (16, 900, 900) to (16, 450, 450)

        x = F.relu(self.conv2(x)) # Size changes from  (16, 450, 450) to (16, 450, 450)
        x = self.pool3(x)  # Size changes from (16, 450, 450) to (16, 150, 150)

        x = F.relu(self.conv3(x))  # Size changes from  (16, 150, 150) to (4, 150, 150)
        x = self.pool3(x)  # Size changes from (4, 150, 150) to (4, 50, 50)

        x = F.relu(self.conv4(x))  # Size changes from  (4, 50, 50) to (1, 50, 50)
        x = self.pool1(x)  # Size changes from (1, 50, 50) to (1, 25, 25)


        # FC-CONNECTED
        x = x.view(-1, 1*25*25)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = self.fc_out(x)

        return x


class LargerFilters(torch.nn.Module):

    def __init__(self):
        super(LargerFilters, self).__init__()

        self.conv1 = torch.nn.Conv2d(8, 8, kernel_size=5, stride=1, padding=2)
        self.conv2 = torch.nn.Conv2d(8, 4, kernel_size=5, stride=1, padding=2)
        self.conv3 = torch.nn.Conv2d(4, 2, kernel_size=3, stride=1, padding=2)
        self.conv4 = torch.nn.Conv2d(2, 1, kernel_size=3, stride=1, padding=1)

        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.pool3 = torch.nn.MaxPool2d(kernel_size=3, stride=3, padding=0)

        self.fc1 = torch.nn.Linear(1*25*25, 16)
        self.fc_out = torch.nn.Linear(16, 3)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # Size changes from (8, 900, 900) to (8, 900, 900)
        x = self.pool1(x)  # Size changes to (8, 450, 450)

        x = F.relu(self.conv2(x)) # Size changes to (8, 450, 450)
        x = self.pool3(x)  # Size changes from (8, 150, 150)

        x = F.relu(self.conv3(x))  # Size changes to (4, 150, 150)
        x = self.pool3(x)  # Size changes to (4, 50, 50)

        x = F.relu(self.conv4(x))  # Size changes to (1, 50, 50)
        x = self.pool1(x)  # Size changes to (1, 25, 25)

        # FC-CONNECTED
        x = x.view(-1, 1*25*25)
        x = torch.tanh(self.fc1(x))
        x = self.fc_out(x)

        return x
