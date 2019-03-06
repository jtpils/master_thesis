import torch.nn.functional as F
import torch
import numpy as np


class SimpleNet3(torch.nn.Module):
    # Simple net
    def __init__(self):
        super(SimpleNet3, self).__init__()

        self.conv1 = torch.nn.Conv2d(8, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv2 = torch.nn.Conv2d(32, 32, kernel_size=3, stride=3, padding=0)

        self.conv3 = torch.nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1)

        self.conv4 = torch.nn.Conv2d(16, 4, kernel_size=3, stride=1, padding=1)

        self.fc1 = torch.nn.Linear(4 * 37 * 37, 32)
        self.fc_out = torch.nn.Linear(32, 3)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # Size changes to 32,900,900
        x = self.pool1(x)  # Size changes to 32,450,450

        x = F.relu(self.conv2(x)) # Size changes to  32,150,150
        x = self.pool1(x)  # Size changes to  32,75,75

        x = F.relu(self.conv3(x)) # Size changes to 16,75,75
        x = self.pool1(x)  # Size changes to 16,37,37

        x = F.relu(self.conv4(x)) # Size changes to 4,37,37
        # FC-CONNECTED
        x = x.view(-1, 4 * 37 * 37)
        x = torch.tanh(self.fc1(x))
        x = self.fc_out(x)

        return x


class SimpleNet3_single_channel(torch.nn.Module):
    # Simple net
    def __init__(self):
        super(SimpleNet3_single_channel, self).__init__()

        self.conv1 = torch.nn.Conv2d(8, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv2 = torch.nn.Conv2d(32, 32, kernel_size=3, stride=3, padding=0)

        self.conv3 = torch.nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1)

        self.conv4 = torch.nn.Conv2d(16, 4, kernel_size=3, stride=1, padding=1)

        self.conv5 = torch.nn.Conv2d(4, 1, kernel_size=3, stride=1, padding=1)


        self.fc1 = torch.nn.Linear(1 * 18 * 18, 32)
        self.fc_out = torch.nn.Linear(32, 3)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # Size changes to 32,900,900
        x = self.pool1(x)  # Size changes to 32,450,450

        x = F.relu(self.conv2(x)) # Size changes to  32,150,150
        x = self.pool1(x)  # Size changes to  32,75,75

        x = F.relu(self.conv3(x)) # Size changes to 16,75,75
        x = self.pool1(x)  # Size changes to 16,37,37

        x = F.relu(self.conv4(x)) # Size changes to 4,37,37

        x = F.relu(self.conv5(x)) # Size changes to 1,37,37
        x = self.pool1(x)  # 1,18,18

        # FC-CONNECTED
        x = x.view(-1, 1 * 18 * 18)
        x = torch.tanh(self.fc1(x))
        x = self.fc_out(x)

        return x


class SimpleNet3_fc(torch.nn.Module):
    # Simple net
    def __init__(self):
        super(SimpleNet3_fc, self).__init__()

        self.conv1 = torch.nn.Conv2d(8, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv2 = torch.nn.Conv2d(32, 32, kernel_size=3, stride=3, padding=0)

        self.conv3 = torch.nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1)

        self.conv4 = torch.nn.Conv2d(16, 4, kernel_size=3, stride=1, padding=1)

        self.fc1 = torch.nn.Linear(4 * 37 * 37, 64)
        self.fc2 = torch.nn.Linear(64, 32)
        self.fc_out = torch.nn.Linear(32, 3)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # Size changes to 32,900,900
        x = self.pool1(x)  # Size changes to 32,450,450

        x = F.relu(self.conv2(x)) # Size changes to  32,150,150
        x = self.pool1(x)  # Size changes to  32,75,75

        x = F.relu(self.conv3(x)) # Size changes to 16,75,75
        x = self.pool1(x)  # Size changes to 16,37,37

        x = F.relu(self.conv4(x)) # Size changes to 4,37,37
        # FC-CONNECTED
        x = x.view(-1, 4 * 37 * 37)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc_out(x)

        return x


class SimpleNet3_more_kernels(torch.nn.Module):
    # Simple net
    def __init__(self):
        super(SimpleNet3_more_kernels, self).__init__()

        self.conv1 = torch.nn.Conv2d(8, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=3, padding=0)

        self.conv3 = torch.nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)

        self.conv4 = torch.nn.Conv2d(32, 4, kernel_size=3, stride=1, padding=1)

        self.fc1 = torch.nn.Linear(4 * 37 * 37, 32)
        self.fc_out = torch.nn.Linear(32, 3)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # Size changes to 32,900,900
        x = self.pool1(x)  # Size changes to 32,450,450

        x = F.relu(self.conv2(x)) # Size changes to  32,150,150
        x = self.pool1(x)  # Size changes to  32,75,75

        x = F.relu(self.conv3(x)) # Size changes to 16,75,75
        x = self.pool1(x)  # Size changes to 16,37,37

        x = F.relu(self.conv4(x)) # Size changes to 4,37,37
        # FC-CONNECTED
        x = x.view(-1, 4 * 37 * 37)
        x = torch.tanh(self.fc1(x))
        x = self.fc_out(x)

        return x
