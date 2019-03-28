
from preprocessing_data_functions import *
from train_network import *
from data_set_pc_samples import *
from point_cloud_net import *

#model_name = input('Type name of new folder: ')
n_epochs = 2 #int(input('Number of epochs: '))
learning_rate = 0.001 #float(input('Learning rate: '))
patience = 1 #int(input('Input patience for EarlyStopping: ')) # Threshold for early stopping. Number of epochs that we will wait until brake
batch_size = 2 #int(input('Input batch size: '))
plot_flag = 'n' #input('Plot results? y / n: ')

print(' ')
print('Number of GPUs available: ', torch.cuda.device_count())
use_cuda = torch.cuda.is_available()
print('CUDA available: ', use_cuda)
#device = torch.device("cuda:0" if use_cuda else "cpu")
#print('Device: ', device)

'''
# create directory for model weights
current_path = os.getcwd()
model_path = os.path.join(current_path, model_name)
os.mkdir(model_path)
parameter_path = os.path.join(model_path, 'parameters')
os.mkdir(parameter_path)'''

train_loss, val_loss = train_network(n_epochs, learning_rate, patience, 'a', use_cuda, batch_size)

