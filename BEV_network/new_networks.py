import torch.nn.functional as F
import torch


class LookAtThisNet(torch.nn.Module):
    # Our batch shape for input x is (8, 900, 900)

    def __init__(self):
        super(LookAtThisNet, self).__init__()

        self.conv1 = torch.nn.Conv2d(8, 64, kernel_size=5, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        self.conv4 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0)
        self.conv5 = torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0)
        self.conv6 = torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0)
        self.conv7 = torch.nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=0)

        self.conv1_bn = torch.nn.BatchNorm2d(64)
        self.conv2_bn = torch.nn.BatchNorm2d(64)
        self.conv3_bn = torch.nn.BatchNorm2d(64)
        self.conv4_bn = torch.nn.BatchNorm2d(128)
        self.conv5_bn = torch.nn.BatchNorm2d(128)
        self.conv6_bn = torch.nn.BatchNorm2d(128)
        self.conv7_bn = torch.nn.BatchNorm2d(128)

        self.pool4 = torch.nn.MaxPool2d(kernel_size=4, stride=4, padding=0)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout_2d = torch.nn.Dropout2d(0.1)
        self.dropout_1d = torch.nn.Dropout(0.1)

        self.fc1 = torch.nn.Linear(128 * 1 * 1, 32)
        self.fc1_bn = torch.nn.BatchNorm1d(32)
        self.fc_out = torch.nn.Linear(32, 3)

    def forward(self, x):

        x = F.relu(self.conv1_bn(self.conv1(x))) # 64, 896, 896
        x = self.pool4(x) # 64, 224, 224
        x = self.dropout_2d(x)

        x = F.relu(self.conv2_bn(self.conv2(x))) # 64, 222, 222
        x = self.pool2(x) # 64, 111, 111
        x = self.dropout_2d(x)

        x = F.relu(self.conv3_bn(self.conv3(x))) # 64, 109, 109
        x = self.pool2(x) # 64, 54, 54
        x = self.dropout_2d(x)

        x = F.relu(self.conv4_bn(self.conv4(x))) # 128, 52, 52
        x = self.pool2(x) # 128, 26, 26
        x = self.dropout_2d(x)

        x = F.relu(self.conv5_bn(self.conv5(x))) # 128, 24, 24
        x = self.pool2(x) # 128, 12, 12
        x = self.dropout_2d(x)

        x = F.relu(self.conv6_bn(self.conv6(x))) # 128, 10, 10
        x = self.pool2(x) # 128, 5, 5
        x = self.dropout_2d(x)

        x = F.relu(self.conv7_bn(self.conv7(x))) # 128, 1, 1
        x = self.dropout_2d(x)

        x = x.view(-1, 128 * 1 * 1)
        x = torch.tanh(self.fc1_bn(self.fc1(x)))
        x = self.dropout_1d(x)
        x = self.fc_out(x)

        return x

    def name(self):
        return "LookAtThisNet"


class LookAtThisNetAgain(torch.nn.Module):
    # Our batch shape for input x is (8, 900, 900)

    def __init__(self):
        super(LookAtThisNetAgain, self).__init__()

        self.conv1 = torch.nn.Conv2d(8, 128, kernel_size=5, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0)
        self.conv4 = torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0)
        self.conv5 = torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0)
        self.conv6 = torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0)
        self.conv7 = torch.nn.Conv2d(256, 128, kernel_size=5, stride=1, padding=0)

        self.conv1_bn = torch.nn.BatchNorm2d(128)
        self.conv2_bn = torch.nn.BatchNorm2d(128)
        self.conv3_bn = torch.nn.BatchNorm2d(128)
        self.conv4_bn = torch.nn.BatchNorm2d(128)
        self.conv5_bn = torch.nn.BatchNorm2d(256)
        self.conv6_bn = torch.nn.BatchNorm2d(256)
        self.conv7_bn = torch.nn.BatchNorm2d(128)

        self.pool4 = torch.nn.MaxPool2d(kernel_size=4, stride=4, padding=0)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout_2d = torch.nn.Dropout2d(0.1)
        self.dropout_1d = torch.nn.Dropout(0.1)

        self.fc1 = torch.nn.Linear(128 * 1 * 1, 32)
        self.fc1_bn = torch.nn.BatchNorm1d(32)
        self.fc_out = torch.nn.Linear(32, 3)

    def forward(self, x):

        x = F.relu(self.conv1_bn(self.conv1(x))) # 64, 896, 896
        x = self.pool4(x) # 64, 224, 224
        x = self.dropout_2d(x)

        x = F.relu(self.conv2_bn(self.conv2(x))) # 64, 222, 222
        x = self.pool2(x) # 64, 111, 111
        x = self.dropout_2d(x)

        x = F.relu(self.conv3_bn(self.conv3(x))) # 64, 109, 109
        x = self.pool2(x) # 64, 54, 54
        x = self.dropout_2d(x)

        x = F.relu(self.conv4_bn(self.conv4(x))) # 128, 52, 52
        x = self.pool2(x) # 128, 26, 26
        x = self.dropout_2d(x)

        x = F.relu(self.conv5_bn(self.conv5(x))) # 128, 24, 24
        x = self.pool2(x) # 128, 12, 12
        x = self.dropout_2d(x)

        x = F.relu(self.conv6_bn(self.conv6(x))) # 128, 10, 10
        x = self.pool2(x) # 128, 5, 5
        x = self.dropout_2d(x)

        x = F.relu(self.conv7_bn(self.conv7(x))) # 128, 1, 1
        x = self.dropout_2d(x)

        x = x.view(-1, 128 * 1 * 1)
        x = torch.tanh(self.fc1_bn(self.fc1(x)))
        x = self.dropout_1d(x)
        x = self.fc_out(x)

        return x


    def name(self):
        return "LookAtThisNetAgain"
        
        
        
class LookAtThisNet_downsampled(torch.nn.Module):
    # Our batch shape for input x is (8, 900, 900)

    def __init__(self):
        super(LookAtThisNet_downsampled, self).__init__()

        self.conv1 = torch.nn.Conv2d(8, 64, kernel_size=5, stride=1, padding=0)
        self.conv2 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0)
        self.conv3 = torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0)
        self.conv4 = torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0)
        self.conv5 = torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0)
        self.conv6 = torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0)
        self.conv7 = torch.nn.Conv2d(128, 128, kernel_size=5, stride=1, padding=0)

        self.conv1_bn = torch.nn.BatchNorm2d(64)
        self.conv2_bn = torch.nn.BatchNorm2d(128)
        self.conv3_bn = torch.nn.BatchNorm2d(128)
        self.conv4_bn = torch.nn.BatchNorm2d(128)
        self.conv5_bn = torch.nn.BatchNorm2d(128)
        self.conv6_bn = torch.nn.BatchNorm2d(128)
        self.conv7_bn = torch.nn.BatchNorm2d(128)

        self.pool4 = torch.nn.MaxPool2d(kernel_size=4, stride=4, padding=0)
        self.pool3 = torch.nn.MaxPool2d(kernel_size=3, stride=3, padding=0)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout_2d = torch.nn.Dropout2d(0.1)
        self.dropout_1d = torch.nn.Dropout(0.1)

        self.fc1 = torch.nn.Linear(128 * 1 * 1, 32)
        self.fc1_bn = torch.nn.BatchNorm1d(32)
        self.fc_out = torch.nn.Linear(32, 3)

    def forward(self, x):
        
        #x = self.pool3(x)  # 8, 300, 300
        
        x = F.relu(self.conv1_bn(self.conv1(x))) # 64, 296, 296
        x = self.pool2(x) # 64, 148, 148
        x = self.dropout_2d(x)

        x = F.relu(self.conv2_bn(self.conv2(x))) # 128, 146, 146
        x = self.pool2(x) # 128, 73, 73
        x = self.dropout_2d(x)

        x = F.relu(self.conv3_bn(self.conv3(x))) # 128, 71, 71
        x = self.pool2(x) # 128, 35, 35
        x = self.dropout_2d(x)

        x = F.relu(self.conv4_bn(self.conv4(x))) # 128, 33, 33
        x = self.pool2(x) # 128, 16, 16
        x = self.dropout_2d(x)

        x = F.relu(self.conv5_bn(self.conv5(x))) # 128, 14, 14
        x = self.pool2(x) # 128, 7, 7
        x = self.dropout_2d(x)

        x = F.relu(self.conv6_bn(self.conv6(x))) # 128, 5, 5
        x = self.dropout_2d(x)

        x = F.relu(self.conv7_bn(self.conv7(x))) # 128, 1, 1
        x = self.dropout_2d(x)

        x = x.view(-1, 128 * 1 * 1)
        x = torch.tanh(self.fc1_bn(self.fc1(x)))
        x = self.dropout_1d(x)
        x = self.fc_out(x)

        return x

    def name(self):
        return "LookAtThisNet_downsampled"
