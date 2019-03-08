from data_loader import *
from train_network import *
from super_simple_cnn import SuperSimpleCNN
from simple_nets_iteration3 import *
import matplotlib.pyplot as plt
import os
import torch
from loaders_only_sweeps import *


# training folder:
# /home/master04/Documents/master_thesis/ProcessingLiDARdata/fake_training_data_trans_1
# /home/master04/Documents/master_thesis/ProcessingLiDARdata/fake_training_data_rot_5
# /Users/sabinalinderoth/Documents/master_thesis/ProcessingLiDARdata/fake_training_data_translated
# /Users/annikal/Desktop/fake_training_data_trans2

# load old weights! change here manually
load_weights = False
load_weights_path = '/home/master04/Desktop/networks_plots_190305/test_multiple_networks_6/parameters_net2/epoch_9_checkpoint.pt'

model_name = input('Type name of new folder: ')
n_epochs = int(input('Number of epochs: '))
learning_rate = float(input('Learning rate: '))
patience = int(input('Input patience for EarlyStopping: ')) # Threshold for early stopping. Number of epochs that we will wait until brake

path_training_data = input('Path to training data set folder: ')
path_validation_data = input('Path to validation data set folder: ')

batch_size = 1  # int(input('Input batch size: '))
plot_flag = input('Plot results? y / n: ')



print(' ')
print('Number of GPUs available: ', torch.cuda.device_count())
use_cuda = torch.cuda.is_available()
print('CUDA available: ', use_cuda)
device = torch.device("cuda" if use_cuda else "cpu")
print('Device: ', device)

CNN = SuperSimpleCNN().to(device)
#CNN = SimpleNet3_single_channel().to(device)
print('Are model parameters on CUDA? ', next(CNN.parameters()).is_cuda)
print(' ')



data_flag = input('Real or fake training data? ( real / fake ): ')
kwargs = {'pin_memory': True} if use_cuda else {}
if data_flag == 'real':
    train_loader, val_loader, test_loader = get_loaders(path_training_data, path_validation_data, batch_size, kwargs)
elif data_flag == 'fake':
    map_folder_path = input('Type path to map folder: ')

    map_path = os.path.join(map_folder_path, '/map.npy')
    map_minmax_values_path = os.path.join(map_folder_path, '/max_min.npy')
    train_loader, val_loader, test_loader = get_sweep_loaders(path_training_data, map_path, map_minmax_values_path,
                                                              path_validation_data, batch_size, kwargs)
else:
    print('You did not type real or fake. Follow the instructions next time, please.')


# Load weights
if load_weights:
    network_param = torch.load(load_weights_path)
    CNN.load_state_dict(network_param['model_state_dict'])

# create directory for model weights
current_path = os.getcwd()
model_path = os.path.join(current_path, model_name)
os.mkdir(model_path)
parameter_path = os.path.join(model_path, 'parameters')
os.mkdir(parameter_path)

# train!
train_loss, val_loss = train_network(CNN, train_loader, val_loader, n_epochs, learning_rate, patience, parameter_path, use_cuda)

if plot_flag is 'y':
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

