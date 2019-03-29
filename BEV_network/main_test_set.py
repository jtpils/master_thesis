from data_loader import get_test_loader
import matplotlib.pyplot as plt
import torch
from cat_networks import *
from torch.autograd import Variable
import torch


# load old weights! change here manually
load_weights = True
load_weights_path = '/home/master04/Desktop/network_parameters/Duchess_12/parameters/epoch_33_checkpoint.pt'

path_test_data = '/home/master04/Desktop/Dataset/BEV_samples/fake_test_set'#'/home/master04/Desktop/Dataset/BEV_samples/fake_test_set'

batch_size = 2 #int(input('Input batch size: '))


print(' ')
print(' ')
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


test_loader = get_test_loader(path_test_data, batch_size, use_cuda)
CNN.eval()

x_pred = list()
y_pred = list()
angle_pred = list()

x_label = list()
y_label = list()
angle_label = list()

loss = torch.nn.SmoothL1Loss()
total_val_loss = 0
CNN = CNN.eval()
with torch.no_grad():
    for i, data in enumerate(test_loader, 1):
        sample = data['sample']
        labels = data['labels']

        # Wrap them in a Variable object
        if use_cuda:
            sample, labels = sample.cuda(), labels.cuda()
        sample, labels = Variable(sample), Variable(labels)

        # Forward pass
        output = CNN.forward(sample)

        val_loss_size = loss(output, labels.float())
        total_val_loss += val_loss_size.item()

        x_pred.append(output.data.tolist()[0][0])
        y_pred.append(output.data.tolist()[0][1])
        angle_pred.append(output.data.tolist()[0][2])

        x_label.append(labels.data.tolist()[0][0])
        y_label.append(labels.data.tolist()[0][1])
        angle_label.append(labels.data.tolist()[0][2])

        if i%10 == 0:
            print('Batch ', i, ' of ', len(test_loader))


diff_x = np.array(x_label) - np.array(x_pred)
diff_y = np.array(y_label) - np.array(y_pred)
diff_angle = np.array(angle_label) - np.array(angle_pred)

print('diff_x: ', diff_x)
print('diff_y: ', diff_y)
print('diff_angle: ', diff_angle)

'''
plt.subplot(1, 2, 1)
plt.plot(vec, diff_x, label='Difference x')
plt.plot(vec, diff_y, label='Difference y')
plt.legend()
plt.ylabel('Difference in meters')

plt.subplot(1, 2, 2)
plt.plot(vec, diff_angle, label='Difference angle')
plt.legend()
plt.ylabel('Difference in degrees')
'''
plt.subplot(1, 3, 1)
plt.hist(diff_x, bins='auto', label='Difference x')
plt.legend()
plt.ylabel('Difference in meters: x')

plt.subplot(1, 3, 2)
plt.hist(diff_y, bins='auto', label='Difference y')
plt.legend()
plt.ylabel('Difference in meters: y')

plt.subplot(1, 3, 3)
plt.hist(diff_angle, bins='auto', label='Difference angle')
plt.legend()
plt.ylabel('Difference in degrees')

plt.show()


model_dict = torch.load(load_weights_path, map_location='cpu')
train_loss = model_dict['train_loss']
val_loss = model_dict['val_loss']
train_vec = np.arange(1,len(train_loss)+1)
val_vec = np.arange(1,len(val_loss)+1)

plt.plot(train_vec, train_loss, label='Training loss')
plt.plot(val_vec, val_loss, label='Validation loss')
plt.xlabel('Epochs')
plt.legend()
plt.show()
