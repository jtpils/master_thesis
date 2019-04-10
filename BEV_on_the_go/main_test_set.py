import matplotlib.pyplot as plt
import torch
from networks import *
from torch.autograd import Variable
import torch
import pandas as pd
import numpy as np
import os
import torch
from torch.autograd import Variable
import time
from DataSets import DataSetMapData_createMapOnTheGo
from torch.utils.data.sampler import SubsetRandomSampler


load_weights = True
path = '/home/master04/Desktop/network_parameters/Duchess_190410_6/'
load_weights_path = path + 'parameters/epoch_12_checkpoint.pt'
batch_size = 4

print('Number of GPUs available: ', torch.cuda.device_count())
use_cuda = torch.cuda.is_available()
print('CUDA available: ', use_cuda)

CNN = Duchess()
print('=======> NETWORK NAME: =======> ', CNN.name())
if use_cuda:
    CNN.cuda()
print('Are model parameters on CUDA? ', next(CNN.parameters()).is_cuda)
print(' ')
if load_weights:
    print('Loading parameters...')
    if use_cuda:
        network_param = torch.load(load_weights_path)
        CNN.load_state_dict(network_param['model_state_dict'])
    else:
        network_param = torch.load(load_weights_path, map_location='cpu')
        CNN.load_state_dict(network_param['model_state_dict'])

sample_path = '/home/master04/Desktop/Ply_files/validation_and_test/test_set/pc/'
csv_path = '/home/master04/Desktop/Ply_files/validation_and_test/test_set/test_set.csv'

kwargs = {'pin_memory': True, 'num_workers': 16} if use_cuda else {'num_workers': 8}

# USE MAP-CUTOUTS
'''
map_train_path = '/home/annika_lundqvist144/maps/map_Town01/map.npy'
map_minmax_train_path = '/home/annika_lundqvist144/maps/map_Town01/max_min.npy'
'''
#map_path = '/home/master04/Desktop/Maps/map_Town_test/map.npy'
#minmax_path = '/home/master04/Desktop/Maps/map_Town_test/max_min.npy'
grid_csv_path = '/home/master04/Desktop/Dataset/ply_grids/csv_grids_190409/csv_grids_test'

test_set = DataSetMapData_createMapOnTheGo(sample_path, csv_path, grid_csv_path)
n_test_samples = len(test_set)

print('Number of test samples: ', n_test_samples)
test_sampler = SubsetRandomSampler(np.arange(n_test_samples, dtype=np.int64))
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, sampler=test_sampler, **kwargs)

CNN.eval()

#predictions_list = list()
#labels_list = list()
predictions_array = np.zeros((1,3))
labels_array = np.zeros((1,3))

split_loss = True

if split_loss:
    loss_trans = torch.nn.MSELoss()
    loss_rot = torch.nn.SmoothL1Loss()
else:
    loss = torch.nn.MSELoss()

total_test_loss = 0
CNN = CNN.eval()
with torch.no_grad():
    for i, data in enumerate(test_loader, 1):
        sample = data['sample']
        labels = data['label']

        # Wrap them in a Variable object
        if use_cuda:
            sample, labels = sample.cuda(), labels.cuda()
        sample, labels = Variable(sample), Variable(labels)

        # Forward pass
        outputs = CNN.forward(sample)
        output_size = outputs.size()[0]

        if split_loss:
            loss_trans_size = loss_trans(outputs[:,0:2], labels[:,0:2].float())
            loss_rot_size = loss_rot(outputs[:,-1].reshape((output_size,1)), labels[:,-1].reshape((output_size,1)).float())

            alpha = 0.9
            beta = 1-alpha
            test_loss_size = alpha*loss_trans_size + beta*loss_rot_size
        else:
            test_loss_size = loss(outputs, labels.float())

        total_test_loss += test_loss_size.item()

        pred = outputs.data.numpy()
        predictions_array = np.concatenate((predictions_array, pred), axis=0)
        lab = labels.data.numpy()
        labels_array = np.concatenate((labels_array, lab), axis=0)

        #predictions_list.append(outputs.data.tolist())
        #labels_list.append(labels.data.tolist())

        if i%10 == 0:
            print('Batch ', i, ' of ', len(test_loader))

predictions = predictions_array[1:,:]
labels = labels_array[1:,:]
#predictions = np.array(predictions_list)  # shape: (minibatch, batch_size, 3)
#labels = np.array(labels_list)
#predictions = predictions.reshape((n_test_samples, 3))  # shape (samples, 3)
#labels = labels.reshape((n_test_samples, 3))
diff = labels - predictions
error_distances = np.hypot(diff[:,0], diff[:,1])

def plot_histograms():
    plt.subplot(1, 3, 1)
    plt.hist(diff[:,0], bins='auto', label='Difference x')
    plt.legend()
    plt.ylabel('Difference in meters: x')

    plt.subplot(1, 3, 2)
    plt.hist(diff[:,1], bins='auto', label='Difference y')
    plt.legend()
    plt.ylabel('Difference in meters: y')

    plt.subplot(1, 3, 3)
    plt.hist(diff[:,2], bins='auto', label='Difference angle')
    plt.legend()
    plt.ylabel('Difference in degrees')

    plt.show()


def visualize_samples():
    sorted = np.argsort(error_distances)
    # best samples
    print('==== Minimum error ====')
    for idx in sorted[:3]:
        data = test_set.__getitem__(idx)
        print('Error ', diff[idx,:])
        print('Error distance ', error_distances[idx])
        print('Labels ', labels[idx,:])
        print('Label distance ', np.hypot(labels[idx,0], labels[idx,1]))
        print(' ')

        visualize_detections(data['sample'],layer=0, fig_num=1)
        visualize_detections(data['sample'],layer=1, fig_num=2)
        plt.show()

    print('==== Maximum error ====')
    for idx in sorted[-3:   ]:
        data = test_set.__getitem__(idx)
        print('Error ', diff[idx,:])
        print('Error distance ', error_distances[idx])
        print('Labels ', labels[idx,:])
        print('Label distance ', np.hypot(labels[idx,0], labels[idx,1]))
        print(' ')

        visualize_detections(data['sample'],layer=0, fig_num=1)
        visualize_detections(data['sample'],layer=1, fig_num=2)

        plt.show()


def visualize_detections(discretized_point_cloud, layer=0, fig_num=1):
    detection_layer = discretized_point_cloud[layer, :, :]
    detection_layer[detection_layer > 0] = 255

    plt.figure(fig_num)
    plt.imshow(detection_layer, cmap='gray')


def plot_loss():
    #model_dict = torch.load(load_weights_path, map_location='cpu')
    train = np.load(path + 'train_loss.npy')
    val = np.load(path + 'val_loss.npy')

    train_batches = train[0]  #first element is the number of minibatches per epoch
    train_loss = train[1:]  #the following elements are the loss for each minibatch
    #val_batches = val[0]
    val_loss = val[1:]

    num_epochs = len(train_loss) / train_batches
    train_vec = np.linspace(1, num_epochs, len(train_loss))
    val_vec = np.linspace(1, num_epochs, len(val_loss))

    plt.plot(train_vec, train_loss, label='Training loss')
    plt.plot(val_vec, val_loss, label='Validation loss')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()


def plot_labels():
    plt.subplot(1, 3, 1)
    plt.hist(labels[:,0], bins='auto', label='Labels x')
    plt.legend()
    #plt.ylabel('Difference in meters: x')

    plt.subplot(1, 3, 2)
    plt.hist(labels[:,1], bins='auto', label='Labels y')
    plt.legend()
    #plt.ylabel('Difference in meters: y')

    plt.subplot(1, 3, 3)
    plt.hist(labels[:,2], bins='auto', label='Labels angle')
    plt.legend()
    #plt.ylabel('Difference in degrees')

    plt.show()


def main():

    plot_labels()
    plot_histograms()
    visualize_samples()
    plot_loss()


if __name__ == '__main__':
    main()
