import torch.nn.functional as F
import torch
import numpy as np


class SimpleNet2(torch.nn.Module):
    # Simple net
    def __init__(self):
        super(SimpleNet2, self).__init__()

        self.conv1 = torch.nn.Conv2d(8, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv2 = torch.nn.Conv2d(32, 4, kernel_size=3, stride=3, padding=0)

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


class SimpleNet2_moreconv(torch.nn.Module):
    # Simple net
    def __init__(self):
        super(SimpleNet2_moreconv, self).__init__()

        self.conv1 = torch.nn.Conv2d(8, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv2 = torch.nn.Conv2d(32, 32, kernel_size=3, stride=3, padding=0)

        self.conv3 = torch.nn.Conv2d(32, 4, kernel_size=3, stride=1, padding=1)

        self.fc1 = torch.nn.Linear(4 * 75 * 75, 32)
        self.fc_out = torch.nn.Linear(32, 3)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # Size changes from (8, 900, 900) to (8, 900, 900)
        x = self.pool1(x)  # Size changes from (8, 900, 900) to (8, 450, 450)

        x = F.relu(self.conv2(x)) # Size changes from  (8, 450, 450) to (4, 150, 150)
        x = self.pool1(x)  # Size changes from (8, 900, 900) to (8, 450, 450)

        x = F.relu(self.conv3(x)) # Size changes from  (8, 450, 450) to (4, 150, 150)

        # FC-CONNECTED
        x = x.view(-1, 4 * 75 * 75)  # change size from (4, 50, 50) to (1, 1000)
        x = torch.tanh(self.fc1(x))  # Change size from (1, 1000) to (1, 512)
        x = self.fc_out(x)  # Change size from (1, 128) to (1, 3)

        return x


class SimpleNet_more_more_conv(torch.nn.Module):
    # Simple net
    def __init__(self):
        super(SimpleNet_more_more_conv, self).__init__()

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


class SimpleNet_more_conv_fc(torch.nn.Module):
    # Simple net
    def __init__(self):
        super(SimpleNet_more_conv_fc, self).__init__()

        self.conv1 = torch.nn.Conv2d(8, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv2 = torch.nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)

        self.conv3 = torch.nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1)

        self.conv4 = torch.nn.Conv2d(16, 4, kernel_size=1, stride=1, padding=0)

        self.fc1 = torch.nn.Linear(4 * 56 * 56, 32)
        self.fc_out = torch.nn.Linear(32, 3)

    def forward(self, x):
        x = F.relu(self.conv1(x))  # 32,900,900
        x = self.pool1(x)  # 32,450,450

        x = F.relu(self.conv2(x)) # 32,450,450
        x = self.pool1(x)  # 32,225,225

        x = F.relu(self.conv3(x)) # 16,225,225
        x = self.pool1(x) # 16,112,112

        x = F.relu(self.conv4(x)) # 4,112,112
        x = self.pool1(x) # 4,56,56

        # FC-CONNECTED
        x = x.view(-1, 4 * 56 * 56)  #
        x = torch.tanh(self.fc1(x))  #
        x = self.fc_out(x)  #

        return x
