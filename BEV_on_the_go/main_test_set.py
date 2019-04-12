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
path = '/home/master04/Desktop/network_parameters/Gustav_190412_1'
load_weights_path = os.path.join(path, 'parameters/epoch_27_checkpoint.pt')
batch_size = 8
translation = float(input('enter translation: '))
rotation = float(input('enter rotation: '))

f = open(os.path.join(path, 'outputs.txt'), "w+")
f.close()
with open(os.path.join(path, 'outputs.txt'), "a+") as f:
    f.write("Metrics from test  \n \n")


print('Number of GPUs available: ', torch.cuda.device_count())
use_cuda = torch.cuda.is_available()
print('CUDA available: ', use_cuda)

CNN = Gustav()
print(' ')
print('=======> NETWORK NAME: =======> ', CNN.name())
print(' ')
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

# ====== TEST SET =====
sample_path = '/home/master04/Desktop/Ply_files/validation_and_test/test_set/pc/'
csv_path = '/home/master04/Desktop/Ply_files/validation_and_test/test_set/test_set.csv'
grid_csv_path = '/home/master04/Desktop/Dataset/ply_grids/csv_grids_190409/csv_grids_test'

# ===== VALIDATION SET ======
#sample_path = '/home/master04/Desktop/Ply_files/validation_and_test/validation_set/pc/'
#csv_path = '/home/master04/Desktop/Ply_files/validation_and_test/validation_set/validation_set.csv'
#grid_csv_path = '/home/master04/Desktop/Dataset/ply_grids/csv_grids_190409/csv_grids_validation'

kwargs = {'pin_memory': True, 'num_workers': 16} if use_cuda else {'num_workers': 8}

test_set = DataSetMapData_createMapOnTheGo(sample_path, csv_path, grid_csv_path, translation=translation, rotation=rotation)
n_test_samples = len(test_set)

print('Number of test samples: ', n_test_samples)
test_sampler = SubsetRandomSampler(np.arange(n_test_samples, dtype=np.int64))
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, sampler=test_sampler, **kwargs)

CNN.eval()
predictions_array = np.zeros((1,3))
labels_array = np.zeros((1,3))

split_loss = True

if split_loss:
    alpha = float(input('Enter weight for alpha in custom loss: '))
    beta = 1-alpha
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
            test_loss_size = alpha*loss_trans_size + beta*loss_rot_size
        else:
            test_loss_size = loss(outputs, labels.float())

        total_test_loss += test_loss_size.item()

        pred = outputs.data.numpy()
        predictions_array = np.concatenate((predictions_array, pred), axis=0)
        lab = labels.data.numpy()
        labels_array = np.concatenate((labels_array, lab), axis=0)

        if i%10 == 0:
            print('Batch ', i, ' of ', len(test_loader))

predictions = predictions_array[1:,:]
labels = labels_array[1:,:]
diff = labels - predictions
error_distances = np.hypot(diff[:,0], diff[:,1])


def plot_histograms():
    fig = plt.figure(figsize=(15,5))
    fig.suptitle('Error distribution')

    ax = fig.add_subplot(131)
    ax.hist(diff[:,0], bins='auto', label='Difference x')
    ax.set_xlabel('Difference in meters: x')
    ax.set_ylabel('Number of samples')
    ax.legend()

    ax = fig.add_subplot(132)
    ax.hist(diff[:,0], bins='auto', label='Difference y')
    ax.set_xlabel('Difference in meters: y')
    ax.set_ylabel('Number of samples')
    ax.legend()

    ax = fig.add_subplot(133)
    ax.hist(diff[:,0], bins='auto', label='Difference angle')
    ax.set_xlabel('Difference in degrees')
    ax.set_ylabel('Number of samples')
    ax.legend()

    plt.show()

    file_name = 'errors.png'
    fig.savefig(os.path.join(path, file_name))


def visualize_samples():
    sorted = np.argsort(error_distances)
    # best samples
    print('==== Minimum error ====')
    i=0
    for idx in sorted[:3]:
        data = test_set.__getitem__(idx)
        print('Error ', diff[idx,:])
        print('Error distance ', error_distances[idx])
        print('Labels ', labels[idx,:])
        print('Label distance ', np.hypot(labels[idx,0], labels[idx,1]))
        print(' ')


        with open(os.path.join(path, 'outputs.txt'), "a+") as f:
            output_text = 'Sample ' + str(i) + '\n'
            f.write(output_text)
            output_text = 'Error ' + str(diff[idx,:]) + '\n'
            f.write(output_text)
            output_text = 'Error distance ' + str(error_distances[idx]) + '\n'
            f.write(output_text)
            output_text = 'Labels ' + str(labels[idx,:]) + '\n'
            f.write(output_text)
            output_text = 'Label distance ' + str(np.hypot(labels[idx,0], labels[idx,1])) + '\n \n'
            f.write(output_text)

        sample = data['sample']
        plot_sample(sample[0,:,:], sample[1,:,:], i)
        i = i+1

    print('==== Maximum error ====')
    for idx in sorted[-3:   ]:
        data = test_set.__getitem__(idx)
        print('Error ', diff[idx,:])
        print('Error distance ', error_distances[idx])
        print('Labels ', labels[idx,:])
        print('Label distance ', np.hypot(labels[idx,0], labels[idx,1]))
        print(' ')


        with open(os.path.join(path, 'outputs.txt'), "a+") as f:
            output_text = 'Sample ' + str(i) + '\n'
            f.write(output_text)
            output_text = 'Error ' + str(diff[idx,:]) + '\n'
            f.write(output_text)
            output_text = 'Error distance ' + str(error_distances[idx]) + '\n'
            f.write(output_text)
            output_text = 'Labels ' + str(labels[idx,:]) + '\n'
            f.write(output_text)
            output_text = 'Label distance ' + str(np.hypot(labels[idx,0], labels[idx,1])) + '\n \n'
            f.write(output_text)

        sample = data['sample']
        plot_sample(sample[0,:,:], sample[1,:,:], i)
        i=i+1


def plot_sample(sweep_image, cutout_image, fig_num):

    sweep_image[sweep_image > 0] = 255
    cutout_image[cutout_image > 0] = 255

    fig = plt.figure()

    ax = fig.add_subplot(121)
    ax.imshow(sweep_image, cmap='gray')
    ax.set_title('Sweep')
    ax.axis('off')

    ax = fig.add_subplot(122)
    ax.imshow(cutout_image, cmap='gray')
    ax.set_title('Map cut-out')
    ax.axis('off')

    file_name = 'sample' + str(fig_num) +'.png'
    fig.savefig(os.path.join(path, file_name))

    '''plt.subplot(1,2,1)
    plt.imshow(sweep_image, cmap='gray')
    plt.title('sweep')
    plt.axis('off')
    plt.subplot(1,2,2)
    plt.imshow(cutout_image, cmap='gray')
    plt.title('map cut-out')
    plt.axis('off')
    plt.show()'''


def plot_loss():
    train = np.load(os.path.join(path,'train_loss.npy'))
    val = np.load(os.path.join(path,'val_loss.npy'))

    train_batches = train[0]  #first element is the number of minibatches per epoch
    train_loss = train[1:]  #the following elements are the loss for each minibatch
    val_loss = val[1:]

    num_epochs = len(train_loss) / train_batches
    train_vec = np.linspace(1, num_epochs, len(train_loss))
    val_vec = np.linspace(1, num_epochs, len(val_loss))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(train_vec, train_loss, label='Training loss')
    ax.plot(val_vec, val_loss, label='Validation loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss per mini-batch')
    ax.legend()

    plt.show()

    file_name = 'losses.png'
    fig.savefig(os.path.join(path, file_name))


def plot_labels():
    fig = plt.figure(figsize=(15,5))
    fig.suptitle('Label distribution')

    ax = fig.add_subplot(131)
    ax.hist(labels[:,0], bins='auto', label='Labels x')
    ax.legend()

    ax = fig.add_subplot(132)
    ax.hist(labels[:,1], bins='auto', label='Labels y')
    ax.legend()

    ax = fig.add_subplot(133)
    ax.hist(labels[:,2], bins='auto', label='Labels rotation')
    ax.legend()

    plt.show()

    file_name = 'labels.png'
    fig.savefig(os.path.join(path, file_name))


def save_as_png(image, file_name):
    file_path = os.path.join(path, file_name+'.png')
    '''plt.savefig('foo.png', bbox_inches='tight')

    # create the png_path
    png_path = folder_path_png + 'channel_' + str(channel)+'.png'

    # Save images
    img = Image.fromarray(discretized_pointcloud_BEV[channel, :, :])
    new_img = img.convert("L")
    new_img.rotate(180).save(png_path)'''


def error_metrics():
    error_distances
    print('Error mean distance: ', np.mean(error_distances))
    print('Error median distance: ', np.median(error_distances))

    with open(os.path.join(path, 'outputs.txt'), "a+") as f:
        output_text = 'Error mean distance:  ' + str(np.mean(error_distances)) + '\n'
        f.write(output_text)
        output_text = 'Error median distance: ' + str(np.median(error_distances)) + '\n'
        f.write(output_text)



def main():

    plot_labels()
    plot_histograms()
    visualize_samples()
    plot_loss()
    error_metrics()


if __name__ == '__main__':
    main()
