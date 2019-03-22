import torch.nn.functional as F
import torch
import numpy as np
# These are our network classes, all named after beloved cats.

class Duchess(torch.nn.Module):
    # Duchess was Sabina's first cat way back when she was a kid. It has been so long the memory of her has faded, but
    # she is still with us through this master's thesis.

    # Inspired by Caltagirone, but with more maxpooling the context module to simplify stuff.

    def __init__(self):
        super(Duchess, self).__init__()
        # Our batch shape for input x is (batch_size, 8, 450 ,450)

        # ENCODER
        self.conv1 = torch.nn.Conv2d(8, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)

        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # CONTEXT
        self.dropout_2d = torch.nn.Dropout2d(p=0.2)
        self.conv3 = torch.nn.Conv2d(32, 128, kernel_size=3, stride=1, padding=0)
        self.conv4 = torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0)
        self.conv5 = torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0)
        self.conv6 = torch.nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=0)
        self.conv7 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        self.conv8 = torch.nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=0)
        self.conv9 = torch.nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=0)
        self.conv10 = torch.nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=0)

        # FC
        self.dropout_1d = torch.nn.Dropout2d(p=0.2)
        self.fc1 = torch.nn.Linear(8*23*23, 32)
        self.fc2 = torch.nn.Linear(32,16)
        self.fc_out = torch.nn.Linear(16, 3)

    def forward(self, x):
        # ENCODER
        x = F.relu(self.conv1(x))  # 32,450,450
        x = F.relu(self.conv2(x))  # 32,450,450
        x = self.pool(x)  # 32,225,225

        # CONTEXT
        x = F.relu(self.conv3(x))  # 128,223,223
        x = self.dropout_2d(x)
        x = F.relu(self.conv4(x))  # 128,221,221
        x = self.dropout_2d(x)
        x = F.relu(self.conv5(x))  # 128,219,219
        x = self.dropout_2d(x)

        x = self.pool(x)  # 128,109,109

        x = F.relu(self.conv6(x))  # 64,107,107
        x = self.dropout_2d(x)
        x = F.relu(self.conv7(x))  # 64,105,105
        x = self.dropout_2d(x)

        x = self.pool(x)  # 64,52,52

        x = F.relu(self.conv8(x))  # 32,50,50
        x = self.dropout_2d(x)
        x = F.relu(self.conv9(x))  # 16,48,48
        x = self.dropout_2d(x)
        x = self.conv10(x)  # 8,46,46

        x = self.pool(x) # 8,23,23

        # FC
        x = x.view(-1, 8*23*23)
        x = torch.tanh(self.fc1(x))
        x = self.dropout_1d(x)

        x = self.fc2(x)
        x = self.fc_out(x)

        return x

    def name(self):
        return "Duchess"


class Gustav(torch.nn.Module):
    # Gustav was a sweet orange kitten from Foot Island that lived with Annika's family during some colorful weeks back
    # in the early 2000's. He found a new home, but will forever be in our hearts.

    # Inspired by Caltagirone.

    def __init__(self):
        super(Gustav, self).__init__()
        # Our batch shape for input x is (batch_size, 8, 450 ,450)

        # ENCODER
        self.conv1 = torch.nn.Conv2d(8, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)

        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # CONTEXT
        self.dropout_2d = torch.nn.Dropout2d(p=0.2)
        self.conv3 = torch.nn.Conv2d(32, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv5 = torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv6 = torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv7 = torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv8 = torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv9 = torch.nn.Conv2d(128, 32, kernel_size=3, stride=1, padding=1)
        self.conv10 = torch.nn.Conv2d(32, 8, kernel_size=1, stride=1, padding=0)

        # FC
        self.dropout_1d = torch.nn.Dropout2d(p=0.2)
        self.fc1 = torch.nn.Linear(8*112*112, 256)
        self.fc2 = torch.nn.Linear(256,64)
        self.fc3 = torch.nn.Linear(64,16)
        self.fc_out = torch.nn.Linear(16, 3)

    def forward(self, x):
        # ENCODER
        x = F.relu(self.conv1(x))  # 32,450,450
        x = F.relu(self.conv2(x))  # 32,450,450
        x = self.pool(x)  # 32,225,225

        # CONTEXT
        x = F.relu(self.conv3(x))  # 128,225,255
        x = self.dropout_2d(x)
        x = F.relu(self.conv4(x))
        x = self.dropout_2d(x)
        x = F.relu(self.conv5(x))
        x = self.dropout_2d(x)
        x = F.relu(self.conv6(x))
        x = self.dropout_2d(x)
        x = F.relu(self.conv7(x))
        x = self.dropout_2d(x)
        x = F.relu(self.conv8(x))
        x = self.dropout_2d(x)
        x = F.relu(self.conv9(x))  # 32,225,225
        x = self.dropout_2d(x)
        x = self.conv10(x)  # 8,225,225

        x = self.pool(x) # 8,112,112

        # FC
        x = x.view(-1, 8*112*112)
        x = torch.tanh(self.fc1(x))
        x = self.dropout_1d(x)

        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc_out(x)

        return x

    def name(self):
        return "Gustav"
