#from first_bev_network import *
from data_loader import *
from train_network import *
from super_simple_cnn import SuperSimpleCNN
import matplotlib.pyplot as plt
import os


model_name = input('Folder name: ')
n_epochs = int(input('Number of epochs:'))
learning_rate = float(input('Learning rate:'))

path_training_data = input('Path to training data folder: ')
batch_size_train = 2
batch_size_val = 2

# get data loaders
train_loader, val_loader = get_loaders(path_training_data, batch_size_train, batch_size_val, train_split=0.5)

# Create network instance
# CNN = FirstBEVNet()
CNN = SuperSimpleCNN()


# create directory for model weights
current_path = os.getcwd()
model_path = os.path.join(current_path, model_name)
os.mkdir(model_path)
weights_path = os.path.join(model_path, 'weights')
os.mkdir(weights_path)

# train!
train_loss, val_loss = train_network(CNN, train_loader, val_loader, n_epochs, learning_rate, weights_path)

# plot loss
'''epochs_vec = np.arange(1, n_epochs+1)
plt.plot(epochs_vec, train_loss, label='train loss')
plt.plot(epochs_vec, val_loss, label='val loss')
plt.legend()
plt.show()'''

# save loss
loss_path = os.path.join(model_path, 'train_loss.npy')
np.save(loss_path, train_loss)
loss_path = os.path.join(model_path, 'val_loss.npy')
np.save(loss_path, val_loss)