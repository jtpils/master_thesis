from train_network import *
import matplotlib.pyplot as plt


# load old weights! change here manually
load_weights = False
load_weights_path = '/home/annika_lundqvist144/master_thesis/First_BEV_Network/param/parameters/epoch_7_checkpoint.pt'

model_name = input('Type name of new folder: ')
n_epochs = 1  #int(input('Number of epochs: '))
learning_rate = 0.001 #float(input('Learning rate: '))
patience = n_epochs  #int(input('Input patience for EarlyStopping: ')) # Threshold for early stopping. Number of epochs that we will wait until brake
batch_size = 2  #int(input('Input batch size: '))
plot_flag = 'n' #input('Plot results? y / n: ')
num_samples = 1 #int(input('Number of samples to train: (max:1400) '))

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
#device = torch.device("cuda:0" if use_cuda else "cpu")
#print('Device: ', device)

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
train_loss, val_loss = train_network(n_epochs, learning_rate, patience, parameter_path, use_cuda, batch_size, num_samples)


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

