from data_loader import *
from train_network import *
from super_simple_cnn import SuperSimpleCNN
from nets_regularization import *
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

#path_training_data = '/home/master04/Desktop/Dataset/fake_training_set' #
#path_validation_data = '/home/master04/Desktop/Dataset/fake_validation_set' #
#path_test_data = '/home/master04/Desktop/Dataset/fake_test_set' #

path_training_data = '/home/annika_lundqvist144/Dataset/fake_training_set' #input('Path to training data set folder: ')
path_validation_data = '/home/annika_lundqvist144/Dataset/fake_validation_set' #input('Path to validation data set folder: ')
path_test_data = '/home/annika_lundqvist144/Dataset/fake_test_set' #input('Path to test data set folder: ')

batch_size = int(input('Input batch size: '))
plot_flag = input('Plot results? y / n: ')



print(' ')
print('Number of GPUs available: ', torch.cuda.device_count())
use_cuda = torch.cuda.is_available()
if use_cuda:
    id = torch.cuda.current_device()
    print('Device id: ', id)
print('CUDA available: ', use_cuda)
device = torch.device("cuda:0" if use_cuda else "cpu")
print('Device: ', device)

##########
#use_cuda = False
#device = "cpu"
##########


'''
if use_cuda:
    id = torch.cuda.current_device()
    print('device id', id)
    mem = torch.cuda.device(id)
    print('memory adress', mem)
    print('device name', torch.cuda.get_device_name(id))
    print('setting device...')
    torch.cuda.set_device(id)'''






CNN = Network_March2().to(device)
#if use_cuda:
#    CNN.cuda()

print('Are model parameters on CUDA? ', next(CNN.parameters()).is_cuda)
print(' ')



kwargs = {'pin_memory': True} if use_cuda else {}
train_loader, val_loader, test_loader = get_loaders(path_training_data, path_validation_data, path_test_data, batch_size, device, kwargs)

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

