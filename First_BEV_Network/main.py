from data_loader import *
from train_network import *
from super_simple_cnn import SuperSimpleCNN
import matplotlib.pyplot as plt
import os
import torch


# training folder:
# /home/master04/Documents/master_thesis/ProcessingLiDARdata/fake_training_data_trans_1
# /home/master04/Documents/master_thesis/ProcessingLiDARdata/fake_training_data_rot_5
# /Users/sabinalinderoth/Documents/master_thesis/ProcessingLiDARdata/fake_training_data_translated


# load old weights! change here manually
load_weights = False
load_weights_path = '/home/master04/Documents/master_thesis/First_BEV_Network/trans1/weights/epoch19.pt'

model_name = input('Type name of new folder: ')
n_epochs = int(input('Number of epochs: '))
learning_rate = float(input('Learning rate: '))
patience = 10 # Threshold for early stopping. Number of epochs that we will wait until brake

path_training_data = input('Path to training data folder: ')
batch_size_train = 2
batch_size_val = 2

print(' ')
print('Number of GPUs available: ', torch.cuda.device_count())
use_cuda = torch.cuda.is_available()
print('CUDA available: ', use_cuda)
device = torch.device("cuda" if use_cuda else "cpu")
print('Device: ', device)


# use_gpu = int(input('Enter 0 for cpu, 1 for gpu:'))  # check that the user really provides 0 or 1

# Create network instance
#CNN = FirstBEVNet().to(device)
CNN = SuperSimpleCNN().to(device)
# if use_gpu:
#     CNN = CNN.cuda()
print('Are model parameters on CUDA? ', next(CNN.parameters()).is_cuda)
print(' ')

# Load weights
if load_weights:
    network_param = torch.load(load_weights_path)
    CNN.load_state_dict(network_param['model_state_dict'])
CNN.train()


# get data loaders
kwargs = {'pin_memory': True} if use_cuda else {}
train_loader, val_loader = get_loaders(path_training_data, batch_size_train, batch_size_val, kwargs, train_split=0.8)

print(type(train_loader))
# create directory for model weights
current_path = os.getcwd()
model_path = os.path.join(current_path, model_name)
os.mkdir(model_path)
parameter_path = os.path.join(model_path, 'parameters')
os.mkdir(parameter_path)

# train!
train_loss, val_loss = train_network(CNN, train_loader, val_loader, n_epochs, learning_rate, patience, parameter_path, use_cuda)

# plot loss
np.shape(train_loss)
epochs_vec = np.arange(1, np.shape(train_loss)[0] + 1) # uses the shape of the train loss to plot to be the same of epochs before early stopping did its work.
plt.plot(epochs_vec, train_loss, label='train loss')
plt.plot(epochs_vec, val_loss, label='val loss')
plt.legend()
plt.show()


# save loss
loss_path = os.path.join(model_path, 'train_loss.npy')
np.save(loss_path, train_loss)
loss_path = os.path.join(model_path, 'val_loss.npy')
np.save(loss_path, val_loss)

