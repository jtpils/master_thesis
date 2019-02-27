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
# /Users/annikal/Desktop/fake_training_data_trans2

# load old weights! change here manually
load_weights = True
load_weights_path = '/Users/annikal/Documents/master_thesis/First_BEV_Network/test/parameters/epoch_24_checkpoint.pt'

path_training_data = input('Path to data set folder: ')

print(' ')
print('Number of GPUs available: ', torch.cuda.device_count())
use_cuda = torch.cuda.is_available()
print('CUDA available: ', use_cuda)
device = torch.device("cuda" if use_cuda else "cpu")
print('Device: ', device)

CNN = SuperSimpleCNN().to(device)
# Load weights
if load_weights:
    network_param = torch.load(load_weights_path)
    CNN.load_state_dict(network_param['model_state_dict'])
print('Are model parameters on CUDA? ', next(CNN.parameters()).is_cuda)
print(' ')

# get data loaders
kwargs = {'pin_memory': True} if use_cuda else {}
batch_size = 1
train_loader, val_loader, test_loader = get_loaders(path_training_data, batch_size, kwargs, train_split=0.8)

CNN.eval()
x_list = list()
y_list = list()
angle_list = list()

x_label = list()
y_label = list()
angle_label = list()

for i, data in enumerate(test_loader, 1):
    # Get inputs
    sample = data['sample']
    labels = data['labels']

    # Wrap them in a Variable object
    if use_cuda:
        sample, labels = Variable(sample).cuda(), Variable(labels).cuda()  # maybe we should use # .to(deveice) here?
    else:
        sample, labels = Variable(sample), Variable(labels)

    output = CNN.forward(sample)


    x_list.append(output.data.tolist()[0][0])
    y_list.append(output.data.tolist()[0][1])
    angle_list.append(output.data.tolist()[0][2])

    x_label.append(labels.data.tolist()[0][0])
    y_label.append(labels.data.tolist()[0][1])
    angle_label.append(labels.data.tolist()[0][2])

vec = np.arange(len(x_list))
'''plt.plot(vec, x_list, label='prediction x')
plt.plot(vec, x_label, label='ground truth x')
plt.legend()
plt.title('Ground truth and prediction: x')
plt.show()'''

diff_x = np.array(x_label) - np.array(x_list)
diff_y = np.array(y_label) - np.array(y_list)
diff_angle = np.array(angle_label) - np.array(angle_list)

plt.plot(vec, diff_x, label='Difference x')
plt.plot(vec, diff_y, label='Difference y')
plt.plot(vec, diff_angle, label='Difference angle')
plt.legend()
plt.xlabel('Difference in meters or degrees')
plt.show()
