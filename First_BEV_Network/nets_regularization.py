import torch.nn.functional as F
import torch


class MyBestNetwork(torch.nn.Module):
    # Simple net
    def __init__(self):
        super(MyBestNetwork, self).__init__()

        self.dropout_2d = torch.nn.Dropout2d(0.1)
        self.dropout_1d = torch.nn.Dropout(0.1)

        self.conv1 = torch.nn.Conv2d(8, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv1_bn = torch.nn.BatchNorm2d(32)

        self.conv2 = torch.nn.Conv2d(32, 32, kernel_size=3, stride=3, padding=0)
        self.conv2_bn = torch.nn.BatchNorm2d(32)

        self.conv3 = torch.nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1)
        self.conv3_bn = torch.nn.BatchNorm2d(16)

        self.conv4 = torch.nn.Conv2d(16, 4, kernel_size=3, stride=1, padding=1)
        self.conv4_bn = torch.nn.BatchNorm2d(4)

        self.conv5 = torch.nn.Conv2d(4, 1, kernel_size=3, stride=1, padding=1)
        self.conv5_bn = torch.nn.BatchNorm2d(1)

        self.fc1 = torch.nn.Linear(1 * 18 * 18, 32)
        self.fc1_bn = torch.nn.BatchNorm1d(32)

        self.fc_out = torch.nn.Linear(32, 3)

    def forward(self, x):
        x = F.relu(self.conv1_bn(self.conv1(x)))  # Size changes to 32,900,900
        x = self.pool1(x)  # Size changes to 32,450,450
        x = self.dropout_2d(x)

        x = F.relu(self.conv2_bn(self.conv2(x))) # Size changes to  32,150,150
        x = self.pool1(x)  # Size changes to  32,75,75
        x = self.dropout_2d(x)

        x = F.relu(self.conv3_bn(self.conv3(x))) # Size changes to 16,75,75
        x = self.pool1(x)  # Size changes to 16,37,37
        x = self.dropout_2d(x)

        x = F.relu(self.conv4_bn(self.conv4(x))) # Size changes to 4,37,37
        x = self.dropout_2d(x)

        x = F.relu(self.conv5_bn(self.conv5(x))) # Size changes to 1,37,37
        x = self.pool1(x)  # 1,18,18
        x = self.dropout_2d(x)

        # FC-CONNECTED
        x = x.view(-1, 1 * 18 * 18)
        x = torch.tanh(self.fc1_bn(self.fc1(x)))
        x = self.dropout_1d(x)
        x = self.fc_out(x)

        return x


class Network_March(torch.nn.Module):
    # Simple net
    def __init__(self):
        super(Network_March, self).__init__()

        self.dropout_2d = torch.nn.Dropout2d(0.1)
        self.dropout_1d = torch.nn.Dropout(0.1)

        self.conv1 = torch.nn.Conv2d(8, 4, kernel_size=3, stride=1, padding=1)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv1_bn = torch.nn.BatchNorm2d(4)

        self.conv3 = torch.nn.Conv2d(4, 4, kernel_size=3, stride=3, padding=0)
        self.conv3_bn = torch.nn.BatchNorm2d(4)

        self.conv4 = torch.nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1)
        self.conv4_bn = torch.nn.BatchNorm2d(4)

        self.conv5 = torch.nn.Conv2d(4, 1, kernel_size=3, stride=1, padding=1)
        self.conv5_bn = torch.nn.BatchNorm2d(1)

        self.fc1 = torch.nn.Linear(1 * 18 * 18, 32)
        self.fc1_bn = torch.nn.BatchNorm1d(32)

        self.fc_out = torch.nn.Linear(32, 3)

    def forward(self, x):
        x = F.relu(self.conv1_bn(self.conv1(x)))  # Size changes to 4,900,900
        x = self.pool1(x)  # Size changes to 4,450,450
        x = self.dropout_2d(x)

        x = F.relu(self.conv3_bn(self.conv3(x))) # Size changes to 4,150,150
        x = self.pool1(x)  # Size changes to 4,75,75
        x = self.dropout_2d(x)

        x = F.relu(self.conv4_bn(self.conv4(x))) # Size changes to 4,75,75
        x = self.pool1(x)  # Size changes to 4,37,37
        x = self.dropout_2d(x)

        x = F.relu(self.conv5_bn(self.conv5(x))) # Size changes to 1,37,37
        x = self.pool1(x)  # 1,18,18
        x = self.dropout_2d(x)

        # FC-CONNECTED
        x = x.view(-1, 1 * 18 * 18)
        x = torch.tanh(self.fc1_bn(self.fc1(x)))
        x = self.dropout_1d(x)
        x = self.fc_out(x)

        return x
