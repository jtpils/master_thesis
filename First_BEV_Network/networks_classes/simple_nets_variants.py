import torch.nn.functional as F
import torch


class SimpleNet(torch.nn.Module):
    # Simple net
    def __init__(self):
        super(SimpleNet, self).__init__()

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


class SimpleNet_large_kernels(torch.nn.Module):
    # Simple net
    def __init__(self):
        super(SimpleNet_large_kernels, self).__init__()

        self.conv1 = torch.nn.Conv2d(8, 8, kernel_size=9, stride=1, padding=8)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv2 = torch.nn.Conv2d(8, 4, kernel_size=5, stride=3, padding=0)

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


class SimpleNet_many_kernels(torch.nn.Module):
    # Simple net
    def __init__(self):
        super(SimpleNet_many_kernels, self).__init__()

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


class SimpleNet_more_conv(torch.nn.Module):
    # Simple net
    def __init__(self):
        super(SimpleNet_more_conv, self).__init__()

        self.conv1 = torch.nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv2 = torch.nn.Conv2d(8, 4, kernel_size=3, stride=3, padding=0)

        self.conv3 = torch.nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1)

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
