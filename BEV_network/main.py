from train_network import *
#import matplotlib.pyplot as plt
import os

def main():
    # load old weights! change here manually
    load_weights = True
    load_weights_path = '/home/annika_lundqvist144/master_thesis/BEV_network/Duchess_190329_2/parameters/epoch_22_checkpoint.pt'

    save_parameters_folder = input('Type name of new folder: ')
    n_epochs = 50 #int(input('Number of epochs: '))
    learning_rate = 0.01 #float(input('Learning rate: '))
    patience = 50 #int(input('Input patience for EarlyStopping: ')) # Threshold for early stopping. Number of epochs that we will wait until brake
    batch_size = 45 #int(input('Input batch size: '))
    plot_flag = 'n' #input('Plot results? y / n: ')

    print(' ')
    print('Number of GPUs available: ', torch.cuda.device_count())
    use_cuda = torch.cuda.is_available()
    print('CUDA available: ', use_cuda)
    #device = torch.device("cuda:0" if use_cuda else "cpu")
    #print('Device: ', device)

    # create directory for model weights
    current_path = os.getcwd()
    save_parameters_path = os.path.join(current_path, save_parameters_folder)
    os.mkdir(save_parameters_path)
    parameter_path = os.path.join(save_parameters_path, 'parameters')
    os.mkdir(parameter_path)

    # train!
    train_loss, val_loss = train_network(n_epochs, learning_rate, patience, parameter_path, use_cuda, batch_size,
                                         load_weights, load_weights_path)

    '''
    if plot_flag is 'y':
        epochs_vec = np.arange(1, np.shape(train_loss)[0] + 1) # uses the shape of the train loss to plot to be the same of epochs before early stopping did its work.
        plt.plot(epochs_vec, train_loss, label='train loss')
        plt.plot(epochs_vec, val_loss, label='val loss')
        plt.legend()
        plt.show()
    '''


if __name__ == '__main__':
    main()
