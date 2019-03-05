from data_loader import *
from train_network import *
from super_simple_cnn import *
from simple_nets_iteration3 import *
import matplotlib.pyplot as plt
import os
import torch


n_epochs = int(input('number of epochs:'))
learning_rate = 0.001
patience = int(input('patience: '))
batch_size = 4
path_training_data = input('path to training data set:') # /home/master04/Desktop/Dataset/fake_training_data_small_set
plot_flag = input('Plot results? y / n: ')

# create directory for model weights
current_path = os.getcwd()
model_path = os.path.join(current_path, 'test_multiple_networks')
os.mkdir(model_path)

print(' ')
print('Number of GPUs available: ', torch.cuda.device_count())
use_cuda = torch.cuda.is_available()
print('CUDA available: ', use_cuda)
device = torch.device("cuda" if use_cuda else "cpu")
print('Device: ', device)
print(' ')

# get data loaders
kwargs = {'pin_memory': True} if use_cuda else {}
train_loader, val_loader, test_loader = get_loaders(path_training_data, batch_size, kwargs)

'''net1 = SuperSimpleCNN().to(device)
net2 = MoreConv().to(device)
net3 = LargerFilters().to(device)
net4 = MoreConvFC().to(device)'''

'''net1 = SimpleNet().to(device)
net2 = SimpleNet_large_kernels().to(device)
net3 = SimpleNet_many_kernels().to(device)
net4 = SimpleNet_more_conv().to(device)'''

'''net1 = SimpleNet2().to(device)
net2 = SimpleNet2_moreconv().to(device)
net3 = SimpleNet_more_more_conv().to(device)
net4 = SimpleNet_more_conv_fc().to(device)'''

net1 = SimpleNet3().to(device)
net2 = SimpleNet3_single_channel().to(device)
net3 = SimpleNet3_fc().to(device)
net4 = SimpleNet3_more_kernels().to(device)

networks = [net1, net2, net3, net4]


i = 1
path_list = list()
for net in networks:
    print(' ')
    print("=" * 27)
    print('Network number ', i)

    net_name = 'parameters_net' + str(i)
    parameter_path = os.path.join(model_path, net_name)
    os.mkdir(parameter_path)
    # train each network here
    train_loss, val_loss = train_network(net, train_loader, val_loader, n_epochs, learning_rate, patience, parameter_path, use_cuda)
    np.save(os.path.join(parameter_path, 'train_loss.npy'), train_loss)
    np.save(os.path.join(parameter_path, 'val_loss.npy'), val_loss)

    path_list.append(parameter_path)
    i = i + 1

if plot_flag is 'y':
    i=1
    for path in path_list:
        train_loss = np.load(os.path.join(path, 'train_loss.npy'))
        val_loss = np.load(os.path.join(path, 'val_loss.npy'))

        plot_number = int('22' + str(i))
        plt.subplot(plot_number)

        epochs_vec = np.arange(1, np.shape(train_loss)[0] + 1) # uses the shape of the train loss to plot to be the same of epochs before early stopping did its work.
        plt.plot(epochs_vec, train_loss, label='train loss')
        plt.plot(epochs_vec, val_loss, label='val loss')
        plt.legend()
        txt = 'network number ' + str(i)
        plt.title(txt)

        i = i + 1
    plt.show()
