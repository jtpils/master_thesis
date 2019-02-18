from TwoInputsNet import TwoInputsNet
from functions import *
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
from data_set_class import Lidar_data_set



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
print(type(train_set))
print(train_set)

# Training
n_training_samples = 7
train_sampler = SubsetRandomSampler(np.arange(n_training_samples, dtype=np.int64))

# Validation
n_val_samples = 2
val_sampler = SubsetRandomSampler(np.arange(n_training_samples, n_training_samples + n_val_samples, dtype=np.int64))

# Test
#n_test_samples = 5000
#test_sampler = SubsetRandomSampler(np.arange(n_test_samples, dtype=np.int64))

# Test and validation loaders have constant batch sizes, so we can define them directly
#test_loader = torch.utils.data.DataLoader(test_set, batch_size=4, sampler=test_sampler, num_workers=2)
val_loader = torch.utils.data.DataLoader(train_set, batch_size=1, sampler=val_sampler, num_workers=2)
'''


def trainNet(net, train_loader, batch_size, n_epochs, learning_rate):
    # Print all of the hyperparameters of the training iteration:
    print("===== HYPERPARAMETERS =====")
    print("batch_size=", batch_size)
    print("epochs=", n_epochs)
    print("learning_rate=", learning_rate)
    print("=" * 30)

    # Get training data
    #train_loader = get_train_loader(batch_size, train_set, train_sampler)
    n_batches = len(train_loader)

    # Create our loss and optimizer functions
    loss, optimizer = createLossAndOptimizer(net, learning_rate)

    # Time for printing
    training_start_time = time.time()

    # Loop for n_epochs
    for epoch in range(n_epochs):

        running_loss = 0.0
        print_every = n_batches // 10
        start_time = time.time()
        total_train_loss = 0

        for i, data in enumerate(train_loader, 1):

            i = 0
            # Get inputs
            #inputs, labels = data
            sweep = data['sweep']
            cutout = data['cutout']
            labels = data['labels']

            # Wrap them in a Variable object
            #inputs, labels = Variable(inputs), Variable(labels)
            sweep, cutout, labels = Variable(sweep), Variable(cutout), Variable(labels)
            print('sweep', np.shape(sweep))
            print('cutout', np.shape(cutout))

            # Set the parameter gradients to zero
            optimizer.zero_grad()

            # Forward pass, backward pass, optimize
            outputs = net.forward(sweep, cutout)
            print('outputs:', outputs)
            print('labels: ', labels)
            loss_size = loss(outputs, labels)
            loss_size.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss_size.item()
            total_train_loss += loss_size.item()

            # Print every 10th batch of an epoch
            if (i + 1) % (print_every + 1) == 0:
                print("Epoch {}, {:d}% \t train_loss: {:.2f} took: {:.2f}s".format(
                    epoch + 1, int(100 * (i + 1) / n_batches), running_loss / print_every, time.time() - start_time))
                # Reset running loss and time
                running_loss = 0.0
                start_time = time.time()

        '''# At the end of the epoch, do a pass on the validation set
        total_val_loss = 0
        for inputs, labels in val_loader:
            # Wrap tensors in Variables
            inputs, labels = Variable(inputs), Variable(labels)

            # Forward pass
            val_outputs = net(inputs)
            val_loss_size = loss(val_outputs, labels)
            total_val_loss += val_loss_size.item()

        print("Validation loss = {:.2f}".format(total_val_loss / len(val_loader)))'''

    print("Training finished, took {:.2f}s".format(time.time() - training_start_time))

csv_file = '/Users/annikal/Documents/master_thesis/ProcessingLiDARdata/data_test/labels.csv'
sweeps_dir = '/Users/annikal/Documents/master_thesis/ProcessingLiDARdata/data_test/sweeps/'
cutouts_dir = '/Users/annikal/Documents/master_thesis/ProcessingLiDARdata/data_test/cutouts/'

lidar_data_set = Lidar_data_set(csv_file, sweeps_dir, cutouts_dir)
#lidar_data_set.__getitem__(1)

n_training_samples = 9
train_sampler = SubsetRandomSampler(np.arange(n_training_samples, dtype=np.int64))
train_loader = torch.utils.data.DataLoader(lidar_data_set, sampler=train_sampler)

CNN = TwoInputsNet()

trainNet(CNN, train_loader, 1, 5, 0.01)

'''
CNN = TwoInputsNet()
#trainNet(CNN, train_set, train_sampler, val_loader, batch_size=1, n_epochs=5, learning_rate=0.01)

#sweep = np.load('/home/master04/Documents/master_thesis/ProcessingLiDARdata/data_test/sweeps/173504.npy')
#cutout = np.load('/home/master04/Documents/master_thesis/ProcessingLiDARdata/data_test/cutouts/173504.npy')

cutout = np.load('/Users/annikal/Documents/master_thesis/ProcessingLiDARdata/data_test/cutouts/1' + '.npy')
sweep = np.load('/Users/annikal/Documents/master_thesis/ProcessingLiDARdata/data_test/sweeps/1' + '.npy')


# convert to tensors first!
sweep = torch.from_numpy(sweep).float()
sweep = np.reshape(sweep, (1, 4, 600, 600))
cutout = torch.from_numpy(cutout).float()
cutout = np.reshape(cutout, (1, 4, 900, 900))

'''

