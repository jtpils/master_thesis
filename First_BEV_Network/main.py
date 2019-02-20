from first_bev_network import *
from data_loader import *
from train_network import *

path_training_data = '/home/master04/Documents/master_thesis/ProcessingLiDARdata/fake_training_data_8-channel-20feb'
batch_size_train = 2
batch_size_val = 2

# get data loaders
train_loader, val_loader = get_loaders(path_training_data, batch_size_train, batch_size_val)

# Create network instance
CNN = FirstBEVNet()

n_epochs = 125
learning_rate = 0.001

train_network(CNN, train_loader, val_loader, n_epochs, learning_rate)
