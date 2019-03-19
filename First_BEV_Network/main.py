from data_loader import *
from train_network import *
import matplotlib.pyplot as plt
#from loaders_only_sweeps import *
#from new_networks import *


# training folder:
# /home/master04/Documents/master_thesis/ProcessingLiDARdata/fake_training_data_trans_1
# /home/master04/Documents/master_thesis/ProcessingLiDARdata/fake_training_data_rot_5
# /Users/sabinalinderoth/Documents/master_thesis/ProcessingLiDARdata/fake_training_data_translated
# /Users/annikal/Desktop/fake_training_data_trans2

# load old weights! change here manually
load_weights = False
load_weights_path = '/home/annika_lundqvist144/master_thesis/First_BEV_Network/param/parameters/epoch_7_checkpoint.pt'

model_name = input('Type name of new folder: ')
n_epochs = 2  #int(input('Number of epochs: '))
learning_rate = 0.001 #float(input('Learning rate: '))
patience = n_epochs  #int(input('Input patience for EarlyStopping: ')) # Threshold for early stopping. Number of epochs that we will wait until brake

# /Users/sabinalinderoth/Desktop/fake_test

#path_training_data = '/Users/sabinalinderoth/Documents/master_thesis/ProcessingLiDARdata/fake_training_set' #
#path_validation_data = '/Users/sabinalinderoth/Documents/master_thesis/ProcessingLiDARdata/fake_validation_set' #
#path_test_data = '/Users/sabinalinderoth/Documents/master_thesis/ProcessingLiDARdata/fake_test_set' #



#path_training_data = '/home/annika_lundqvist144/Dataset/fake_training_set' #input('Path to training data set folder: ')
#path_validation_data = '/home/annika_lundqvist144/Dataset/fake_validation_set' #input('Path to validation data set folder: ')
#path_test_data = '/home/annika_lundqvist144/Dataset/fake_test_set' #input('Path to test data set folder: ')

#path_training_data = ''
#path_validation_data = 'test'
#path_test_data = 'test' #

batch_size = 2  #int(input('Input batch size: '))

plot_flag = 'n' #input('Plot results? y / n: ')


print(' ')
print('Number of GPUs available: ', torch.cuda.device_count())
use_cuda = torch.cuda.is_available()

##########
#use_cuda = False
#device = "cpu"
##########


#if use_cuda:
#    id = torch.cuda.current_device()
#    print('Device id: ', id)
print('CUDA available: ', use_cuda)
device = torch.device("cuda:0" if use_cuda else "cpu")
#print('Device: ', device)



'''
if use_cuda:
    id = torch.cuda.current_device()
    print('device id', id)
    mem = torch.cuda.device(id)
    print('memory adress', mem)
    print('device name', torch.cuda.get_device_name(id))
    print('setting device...')
    torch.cuda.set_device(id)'''



#CNN = Network_March2().to(device)
#CNN = MyBestNetwork().to(device)

#CNN = LookAtThisNet_downsampled()
'''CNN = LookAtThisNet()

print('=======> NETWORK NAME: =======> ', CNN.name())

if use_cuda:
    CNN.cuda()

print('Are model parameters on CUDA? ', next(CNN.parameters()).is_cuda)
print(' ')



kwargs = {'pin_memory': True} if use_cuda else {}
train_loader, val_loader = get_loaders(path_training_data, path_validation_data, path_test_data, batch_size, use_cuda, kwargs)'''

# Load weights
#if load_weights:
#    print('Loading parameters...')
#    network_param = torch.load(load_weights_path)
#    CNN.load_state_dict(network_param['model_state_dict'])

# create directory for model weights
current_path = os.getcwd()
model_path = os.path.join(current_path, model_name)
os.mkdir(model_path)
parameter_path = os.path.join(model_path, 'parameters')
os.mkdir(parameter_path)

# train!

#train_loss, val_loss = train_network(CNN, train_loader, val_loader, n_epochs, learning_rate, patience, parameter_path, device, use_cuda,batch_size)
train_loss, val_loss = train_network(n_epochs, learning_rate, patience, parameter_path, device, use_cuda, batch_size)


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

