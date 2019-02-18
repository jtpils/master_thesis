from TwoInputsNet import TwoInputsNet
from functions import *
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler


# TUTORIAL: https://blog.algorithmia.com/convolutional-neural-nets-in-pytorch/
'''
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

# The compose function allows for multiple transforms
# transforms.ToTensor() converts our PILImage to a tensor of shape (C x H x W) in the range [0,1]
# transforms.Normalize(mean,std) normalizes a tensor to a (mean, std) for (R, G, B)
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
train_set = torchvision.datasets.CIFAR10(root='./cifardata', train=True, download=True, transform=transform)
test_set = torchvision.datasets.CIFAR10(root='./cifardata', train=False, download=True, transform=transform)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Training
n_training_samples = 20000
train_sampler = SubsetRandomSampler(np.arange(n_training_samples, dtype=np.int64))

# Validation
n_val_samples = 5000
val_sampler = SubsetRandomSampler(np.arange(n_training_samples, n_training_samples + n_val_samples, dtype=np.int64))

# Test
n_test_samples = 5000
test_sampler = SubsetRandomSampler(np.arange(n_test_samples, dtype=np.int64))

# Test and validation loaders have constant batch sizes, so we can define them directly
test_loader = torch.utils.data.DataLoader(test_set, batch_size=4, sampler=test_sampler, num_workers=2)
val_loader = torch.utils.data.DataLoader(train_set, batch_size=128, sampler=val_sampler, num_workers=2)
'''

CNN = TwoInputsNet()
#trainNet(CNN, train_set, train_sampler, val_loader, batch_size=32, n_epochs=5, learning_rate=0.001)

sweep = np.load('/home/master04/Documents/master_thesis/ProcessingLiDARdata/data_test/sweeps/173504.npy')
cutout = np.load('/home/master04/Documents/master_thesis/ProcessingLiDARdata/data_test/cutouts/173504.npy')

# convert to tensors first!
sweep, cutout = torch.from_numpy(sweep).float(), torch.from_numpy(cutout).float()
sweep, cutout = np.reshape(sweep, [1, 4, 600, 600]),  np.reshape(cutout, [1, 4, 900, 900])
# sweep, cutout = sweep.type(torch.DoubleTensor), cutout.type(torch.DoubleTensor)

CNN.forward(sweep, cutout)

