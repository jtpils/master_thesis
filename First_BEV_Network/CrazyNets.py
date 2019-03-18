import torch.nn as nn
import torch.nn.functional as F



class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(8, 20, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5, stride=1, padding=0)
        self.fc1 = nn.Linear(19*19*50, 500)
        self.fc2 = nn.Linear(500, 3)

    def forward(self, x):
        x = F.max_pool2d(x, 10, 10)  # shape 8, 90, 90
        x = F.relu(self.conv1(x))  # shape 20, 86, 86
        x = F.max_pool2d(x, 2, 2) # 20, 43, 43
        x = F.relu(self.conv2(x)) # 50, 39, 39
        x = F.max_pool2d(x, 2, 2) # 50, 19, 19
        x = x.view(-1, 19*19*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def name(self):
        return "LeNet"


class LeNetMORE(nn.Module):
    def __init__(self):
        super(LeNetMORE, self).__init__()
        self.conv1 = nn.Conv2d(8, 100, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(100, 50, kernel_size=5, stride=1, padding=0)
        self.fc1 = nn.Linear(19*19*50, 500)
        self.fc2 = nn.Linear(500, 3)

    def forward(self, x):
        x = F.max_pool2d(x, 10, 10)  # shape 8, 90, 90
        x = F.relu(self.conv1(x))  # shape 100, 86, 86
        x = F.max_pool2d(x, 2, 2) # 100, 43, 43
        x = F.relu(self.conv2(x)) # 50, 39, 39
        x = F.max_pool2d(x, 2, 2) # 50, 19, 19
        x = x.view(-1, 19*19*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def name(self):
        return "LeNetMORE"


class LeNetCRAZY(nn.Module):
    def __init__(self):
        super(LeNetCRAZY, self).__init__()
        self.conv1 = nn.Conv2d(8, 100, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(100, 100, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(100, 100, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(100, 50, kernel_size=5, stride=1, padding=0)
        self.fc1 = nn.Linear(19*19*50, 500)
        self.fc2 = nn.Linear(500, 3)

    def forward(self, x):
        x = F.max_pool2d(x, 10, 10)  # shape 8, 90, 90
        x = F.relu(self.conv1(x))  # shape 100, 86, 86
        x = F.relu(self.conv2(x))  # shape 100, 86, 86
        x = F.relu(self.conv3(x))  # shape 100, 86, 86
        x = F.max_pool2d(x, 2, 2) # 100, 43, 43
        x = F.relu(self.conv4(x)) # 50, 39, 39
        x = F.max_pool2d(x, 2, 2) # 50, 19, 19
        x = x.view(-1, 19*19*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def name(self):
        return "LeNetCRAZY"


class LeNetWTF(nn.Module):
    def __init__(self):
        super(LeNetWTF, self).__init__()
        self.conv1 = nn.Conv2d(8, 500, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(500, 500, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(500, 500, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(500, 500, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(500, 500, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(500, 50, kernel_size=5, stride=1, padding=0)
        self.fc1 = nn.Linear(19*19*50, 500)
        self.fc2 = nn.Linear(500, 3)

    def forward(self, x):
        x = F.max_pool2d(x, 10, 10)  # shape 8, 90, 90
        x = F.relu(self.conv1(x))  # shape 100, 86, 86
        x = F.relu(self.conv2(x))  # shape 100, 86, 86
        x = F.relu(self.conv3(x))  # shape 100, 86, 86
        x = F.relu(self.conv4(x))  # shape 100, 86, 86
        x = F.relu(self.conv5(x))  # shape 100, 86, 86

        x = F.max_pool2d(x, 2, 2) # 100, 43, 43
        x = F.relu(self.conv6(x)) # 50, 39, 39
        x = F.max_pool2d(x, 2, 2) # 50, 19, 19
        x = x.view(-1, 19*19*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def name(self):
        return "LeNetWTF"
