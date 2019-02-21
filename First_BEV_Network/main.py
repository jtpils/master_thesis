#from first_bev_network import *
from data_loader import *
from train_network import *
from super_simple_cnn import SuperSimpleCNN

path_training_data = '/Users/annikal/Desktop/fake_training_data_trans2'
batch_size_train = 2
batch_size_val = 2

# get data loaders
train_loader, val_loader = get_loaders(path_training_data, batch_size_train, batch_size_val, train_split=0.9)

# Create network instance
#CNN = FirstBEVNet()
CNN = SuperSimpleCNN()

n_epochs = 10
learning_rate = 0.01

train_loss, val_loss = train_network(CNN, train_loader, val_loader, n_epochs, learning_rate)

