import torch.nn.functional as F
import torch
import numpy as np
# These are our network classes, all named after beloved cats.


class Duchess(torch.nn.Module):
    # Duchess was Sabina's first cat way back when she was a kid. It has been so long the memory of her has faded, but
    # she is still with us through this master's thesis.

    # Inspired by Caltagirone, but with more maxpooling in the context module to simplify stuff.

    def __init__(self):
        super(Duchess, self).__init__()
        # Our batch shape for input x is (batch_size, 2, 300, 300)

        # ENCODER
        self.conv1 = torch.nn.Conv2d(2, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv1_bn = torch.nn.BatchNorm2d(32)
        self.conv2_bn = torch.nn.BatchNorm2d(32)

        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # CONTEXT
        self.dropout_2d = torch.nn.Dropout2d(p=0.2)
        self.conv3 = torch.nn.Conv2d(32, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv5 = torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv6 = torch.nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.conv7 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv8 = torch.nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv9 = torch.nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1)
        self.conv10 = torch.nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1)

        self.conv3_bn = torch.nn.BatchNorm2d(128)
        self.conv4_bn = torch.nn.BatchNorm2d(128)
        self.conv5_bn = torch.nn.BatchNorm2d(128)
        self.conv6_bn = torch.nn.BatchNorm2d(64)
        self.conv7_bn = torch.nn.BatchNorm2d(64)
        self.conv8_bn = torch.nn.BatchNorm2d(32)
        self.conv9_bn = torch.nn.BatchNorm2d(16)
        self.conv10_bn = torch.nn.BatchNorm2d(8)

        # FC
        self.dropout_1d = torch.nn.Dropout2d(p=0.2)
        self.fc1 = torch.nn.Linear(8*18*18, 256)
        self.fc2 = torch.nn.Linear(256, 64)
        self.fc_out = torch.nn.Linear(64, 3)

        self.fc1_bn = torch.nn.BatchNorm1d(256)
        self.fc2_bn = torch.nn.BatchNorm1d(64)

    def forward(self, x):
        # ENCODER INPUT: 32,300,300
        x = F.relu(self.conv1_bn(self.conv1(x)))  # 32,300,300
        x = F.relu(self.conv2_bn(self.conv2(x)))  # 32,300,300
        x = self.pool(x)  # 32,150,150

        # CONTEXT
        x = F.relu(self.conv3_bn(self.conv3(x)))  # 128,150,150
        x = self.dropout_2d(x)
        x = F.relu(self.conv4_bn(self.conv4(x)))  # 128,150,150
        x = self.dropout_2d(x)
        x = F.relu(self.conv5_bn(self.conv5(x)))  # 128,150,150
        x = self.dropout_2d(x)

        x = self.pool(x)  # 128,75,75

        x = F.relu(self.conv6_bn(self.conv6(x)))  # 64,75,75
        x = self.dropout_2d(x)
        x = F.relu(self.conv7_bn(self.conv7(x)))  # 64,75,75
        x = self.dropout_2d(x)

        x = self.pool(x)  # 64,37,37

        x = torch.tanh(self.conv8_bn(self.conv8(x)))  # 32,37,37
        x = self.dropout_2d(x)
        x = torch.tanh(self.conv9_bn(self.conv9(x)))  # 16,37,37
        x = self.dropout_2d(x)
        x = torch.tanh(self.conv10_bn(self.conv10(x)))  # 8,37,37

        x = self.pool(x) # 8,18,18

        # FC
        x = x.view(-1, 8*18*18)
        x = torch.tanh(self.fc1_bn(self.fc1(x)))
        x = self.dropout_1d(x)
        x = torch.tanh(self.fc2_bn(self.fc2(x)))

        x = self.fc_out(x)

        return x

    def name(self):
        return "Duchess"



class Gustav(torch.nn.Module):
    # Gustav was a sweet orange kitten from Foot Island that lived with Annika's family during some colorful weeks back
    # in the early 2000's. He found a new home, but will forever be in our hearts.

    # Use two inputs, eg sweep + cutout

    def __init__(self):
        super(Gustav, self).__init__()

        self.conv1_input1 = torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2_input1 = torch.nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv1_bn_input1 = torch.nn.BatchNorm2d(32)
        self.conv2_bn_input1 = torch.nn.BatchNorm2d(32)
        self.conv3_input1 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3_bn_input1 = torch.nn.BatchNorm2d(64)

        # OBS I FLOWNET HAR DE INTE OLIKA CONV-LAGER FOR 2 INPUTS

        self.conv1_input2 = torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2_input2 = torch.nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv1_bn_input2 = torch.nn.BatchNorm2d(32)
        self.conv2_bn_input2 = torch.nn.BatchNorm2d(32)
        self.conv3_input2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3_bn_input2 = torch.nn.BatchNorm2d(64)


        self.dropout_2d = torch.nn.Dropout2d(p=0.2)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # CONTEXT
        self.conv4 = torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv5 = torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv6 = torch.nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.conv7 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv8 = torch.nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.conv9 = torch.nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1)
        self.conv10 = torch.nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1)

        self.conv3_bn = torch.nn.BatchNorm2d(128)
        self.conv4_bn = torch.nn.BatchNorm2d(128)
        self.conv5_bn = torch.nn.BatchNorm2d(128)
        self.conv6_bn = torch.nn.BatchNorm2d(64)
        self.conv7_bn = torch.nn.BatchNorm2d(64)
        self.conv8_bn = torch.nn.BatchNorm2d(32)
        self.conv9_bn = torch.nn.BatchNorm2d(16)
        self.conv10_bn = torch.nn.BatchNorm2d(8)

        # FC
        self.dropout_1d = torch.nn.Dropout2d(p=0.2)
        self.fc1 = torch.nn.Linear(8*18*18, 256)
        self.fc2 = torch.nn.Linear(256, 64)
        self.fc_out = torch.nn.Linear(64, 3)

        self.fc1_bn = torch.nn.BatchNorm1d(256)
        self.fc2_bn = torch.nn.BatchNorm1d(64)


    def forward(self, x):
        use_cuda = torch.cuda.is_available()

        batch_size = np.shape(x)[0]
        sweep = x[:,0,:,:]
        if use_cuda:
            sweep = torch.reshape(sweep, (batch_size,1,300,300)).cuda()
        else:
            sweep = torch.reshape(sweep, (batch_size,1,300,300))

        cutout = x[:,1,:,:] # [batch, 1, 300, 300]
        if use_cuda:
            cutout = torch.reshape(cutout, (batch_size,1,300,300)).cuda()
        else:
            cutout = torch.reshape(cutout, (batch_size,1,300,300))

        sweep = F.relu(self.conv1_bn_input1(self.conv1_input1(sweep)))  # 32,300,300
        sweep = F.relu(self.conv2_bn_input1(self.conv2_input1(sweep)))  # 32,300,300
        sweep = self.pool(sweep)  # 32,150,150
        sweep = F.relu(self.conv3_bn_input1(self.conv3_input1(sweep)))  # 64,150,150

        '''# using the same network for both sweep and cutout
        cutout = F.relu(self.conv1_bn_input1(self.conv1_input1(cutout)))  # 32,300,300
        cutout = F.relu(self.conv2_bn_input1(self.conv2_input1(cutout)))  # 32,300,300
        cutout = self.pool(cutout)  # 32,150,150
        cutout = F.relu(self.conv3_bn_input1(self.conv3_input1(cutout)))  # 64,150,150'''


        # using different networks for sweep and cutout
        cutout = F.relu(self.conv1_bn_input2(self.conv1_input2(cutout)))  # 32,300,300
        cutout = F.relu(self.conv2_bn_input2(self.conv2_input2(cutout)))  # 32,300,300
        cutout = self.pool(cutout)  # 32,150,150
        cutout = F.relu(self.conv3_bn_input2(self.conv3_input2(cutout)))  # 64,150,150

        if use_cuda:
            x = torch.zeros((batch_size, 2*64, 150, 150)).cuda()
        else:
            x = torch.zeros((batch_size, 2*64, 150, 150))

        for i in np.arange(batch_size):
            y = torch.cat((sweep[i,:,:,:],cutout[i,:,:,:]), 0)
            x[i, :, :, :] = y

        x = F.relu(self.conv4_bn(self.conv4(x)))  # 128,150,150
        x = self.dropout_2d(x)
        x = F.relu(self.conv5_bn(self.conv5(x)))  # 128,150,150
        x = self.dropout_2d(x)

        x = self.pool(x)  # 128,75,75

        x = F.relu(self.conv6_bn(self.conv6(x)))  # 64,75,75
        x = self.dropout_2d(x)
        x = F.relu(self.conv7_bn(self.conv7(x)))  # 64,75,75
        x = self.dropout_2d(x)

        x = self.pool(x)  # 64,37,37

        x = torch.tanh(self.conv8_bn(self.conv8(x)))  # 32,37,37
        x = self.dropout_2d(x)
        x = torch.tanh(self.conv9_bn(self.conv9(x)))  # 16,37,37
        x = self.dropout_2d(x)
        x = torch.tanh(self.conv10_bn(self.conv10(x)))  # 8,37,37

        x = self.pool(x) # 8,18,18

        # FC
        x = x.view(-1, 8*18*18)
        x = torch.tanh(self.fc1_bn(self.fc1(x)))
        x = self.dropout_1d(x)
        x = torch.tanh(self.fc2_bn(self.fc2(x)))

        x = self.fc_out(x)

        return x

    def name(self):
        return "Gustav"


class Aslan(torch.nn.Module):
    # A big orange cat which went old an crazy. In loving memory.


    def __init__(self):
        super(Aslan, self).__init__()
        # Our batch shape for input x is (batch_size, 2, 300, 300)

        # ENCODER
        self.conv1 = torch.nn.Conv2d(2, 64, kernel_size=11, stride=1, padding=5)
        self.conv2 = torch.nn.Conv2d(64, 64, kernel_size=11, stride=1, padding=5)
        self.conv1_bn = torch.nn.BatchNorm2d(64)
        self.conv2_bn = torch.nn.BatchNorm2d(64)

        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # CONTEXT
        self.dropout_2d = torch.nn.Dropout2d(p=0.2)
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=7, stride=1, padding=3)
        self.conv4 = torch.nn.Conv2d(128, 128, kernel_size=7, stride=1, padding=3)
        self.conv5 = torch.nn.Conv2d(128, 128, kernel_size=7, stride=1, padding=3)
        self.conv6 = torch.nn.Conv2d(128, 128, kernel_size=7, stride=1, padding=3)
        self.conv7 = torch.nn.Conv2d(128, 64, kernel_size=7, stride=1, padding=3)
        self.conv8 = torch.nn.Conv2d(64, 32, kernel_size=7, stride=1, padding=3)
        self.conv9 = torch.nn.Conv2d(32, 16, kernel_size=7, stride=1, padding=3)
        self.conv10 = torch.nn.Conv2d(16, 8, kernel_size=7, stride=1, padding=3)

        self.conv3_bn = torch.nn.BatchNorm2d(128)
        self.conv4_bn = torch.nn.BatchNorm2d(128)
        self.conv5_bn = torch.nn.BatchNorm2d(128)
        self.conv6_bn = torch.nn.BatchNorm2d(128)
        self.conv7_bn = torch.nn.BatchNorm2d(64)
        self.conv8_bn = torch.nn.BatchNorm2d(32)
        self.conv9_bn = torch.nn.BatchNorm2d(16)
        self.conv10_bn = torch.nn.BatchNorm2d(8)

        # FC
        self.dropout_1d = torch.nn.Dropout2d(p=0.2)
        self.fc1 = torch.nn.Linear(8*18*18, 512)
        self.fc2 = torch.nn.Linear(512, 256)
        self.fc3 = torch.nn.Linear(256, 64)
        self.fc_out = torch.nn.Linear(64, 3)

        self.fc1_bn = torch.nn.BatchNorm1d(512)
        self.fc2_bn = torch.nn.BatchNorm1d(256)
        self.fc3_bn = torch.nn.BatchNorm1d(64)

    def forward(self, x):
        # ENCODER INPUT: 32,300,300
        x = F.elu(self.conv1_bn(self.conv1(x)))  # 32,300,300
        x = F.elu(self.conv2_bn(self.conv2(x)))  # 32,300,300
        x = self.pool(x)  # 32,150,150

        # CONTEXT
        x = F.elu(self.conv3_bn(self.conv3(x)))  # 128,150,150
        x = self.dropout_2d(x)
        x = F.elu(self.conv4_bn(self.conv4(x)))  # 128,150,150
        x = self.dropout_2d(x)
        x = F.elu(self.conv5_bn(self.conv5(x)))  # 128,150,150
        x = self.dropout_2d(x)

        x = self.pool(x)  # 128,75,75

        x = F.elu(self.conv6_bn(self.conv6(x)))  # 64,75,75
        x = self.dropout_2d(x)
        x = F.elu(self.conv7_bn(self.conv7(x)))  # 64,75,75
        x = self.dropout_2d(x)

        x = self.pool(x)  # 64,37,37

        x = torch.tanh(self.conv8_bn(self.conv8(x)))  # 32,37,37
        x = self.dropout_2d(x)
        x = torch.tanh(self.conv9_bn(self.conv9(x)))  # 16,37,37
        x = self.dropout_2d(x)
        x = torch.tanh(self.conv10_bn(self.conv10(x)))  # 8,37,37

        x = self.pool(x) # 8,18,18

        # FC
        x = x.view(-1, 8*18*18)
        x = torch.tanh(self.fc1_bn(self.fc1(x)))
        x = self.dropout_1d(x)
        x = torch.tanh(self.fc2_bn(self.fc2(x)))
        x = self.dropout_1d(x)
        x = torch.tanh(self.fc3_bn(self.fc3(x)))

        x = self.fc_out(x)

        return x

    def name(self):
        return "Aslan"
