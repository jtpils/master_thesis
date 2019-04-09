import torch
from networks import *
from DataSets import *
from torch.optim.lr_scheduler import StepLR
import time
from torch.autograd import Variable
from early_stopping import EarlyStopping


def main():
    n_epochs = 50 #int(input('Number of epochs: '))
    learning_rate = 0.01 #float(input('Learning rate: '))
    patience = 15 #int(input('Input patience for EarlyStopping: ')) # Threshold for early stopping. Number of epochs that we will wait until brake

    use_cuda = torch.cuda.is_available()
    print('CUDA available: ', use_cuda)
    if use_cuda:
        batch_size = 45
    else:
        batch_size = 2

    # create directory for model weights
    save_parameters_folder = input('Enter name for directory to save weights and losses: ')
    current_path = os.getcwd()
    save_parameters_path = os.path.join(current_path, save_parameters_folder)
    parameter_path = os.path.join(save_parameters_path, 'parameters')
    try:
        os.mkdir(save_parameters_path)
        os.mkdir(parameter_path)
        print('Created new directories.')
    except:
        print('Failed to create new directories.')

    # path to ply-files for town01 and validation
    '''
    path_training = '/home/master04/Desktop/Ply_files/_out_Town01_190402_1/pc/'
    path_training_csv = '/home/master04/Desktop/Ply_files/_out_Town01_190402_1/Town01_190402_1.csv'
    path_validation = '/home/master04/Desktop/Ply_files/validation_and_test/validation_set/pc/'
    path_validation_csv = '/home/master04/Desktop/Ply_files/validation_and_test/validation_set/validation_set.csv'
    '''

    path_training = '/home/annika_lundqvist144/ply_files/_out_Town01_190402_1/pc/'
    path_training_csv = '/home/annika_lundqvist144/ply_files/_out_Town01_190402_1/Town01_190402_1.csv'
    path_validation = '/home/annika_lundqvist144/ply_files//validation_set/pc/'
    path_validation_csv = '/home/annika_lundqvist144/ply_files/validation_set/validation_set.csv'


    CNN = Duchess()
    print('=======> NETWORK NAME: =======> ', CNN.name())
    if use_cuda:
        CNN.cuda()
    print('Are model parameters on CUDA? ', next(CNN.parameters()).is_cuda)
    print(' ')

    # get data loaders
    train_loader, val_loader = get_loaders(path_training, path_training_csv, path_validation, path_validation_csv, batch_size, use_cuda)
    n_batches = len(train_loader)
    val_batches = len(val_loader)
    train_loss_save = [len(train_loader)]  # append train loss for each mini batch later on, save this information to plot correctly
    val_loss_save = [len(val_loader)]
    print('Batch size: ', batch_size)

    # initialize the early_stopping object
    early_stopping = EarlyStopping(parameter_path, patience, verbose=True)

    loss = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(CNN.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    for epoch in range(n_epochs):
        scheduler.step()
        params = optimizer.state_dict()['param_groups']
        print(' ')
        print('learning rate: ', params[0]['lr'])

        running_loss = 0.0
        print_every = n_batches // 5  # how many mini-batches if we want to print stats x times per epoch
        start_time = time.time()
        batch_time = time.time()
        total_train_loss = 0.0
        val_loss_save = list()

        CNN = CNN.train()
        for i, data in enumerate(train_loader, 1):
            sample = data['sample']
            labels = data['label']

            if use_cuda:
                sample, labels = sample.cuda(async=True), labels.cuda(async=True)
            sample, labels = Variable(sample), Variable(labels)

            # Set the parameter gradients to zero
            optimizer.zero_grad()
            # Forward pass, backward pass, optimize
            outputs = CNN.forward(sample)
            loss_size = loss(outputs, labels.float())
            loss_size.backward()
            optimizer.step()

            running_loss += loss_size.item()
            total_train_loss += loss_size.item()
            train_loss_save.append(loss_size.item())

            if (i+1) % print_every == 0:
                print('Epoch [{}/{}], Batch [{}/{}], Loss: {:.4f}, Time: {:.2f}s'
                       .format(epoch+1, n_epochs, i, n_batches, running_loss/print_every, time.time()-batch_time))
                running_loss = 0.0
                batch_time = time.time()

        del data, sample, labels, outputs, loss_size

        print('===== Validation =====')
        total_val_loss = 0
        CNN = CNN.eval()
        val_time = time.time()
        with torch.no_grad():
            print('number of iterations: ', val_batches)
            for i, data in tqdm(enumerate(val_loader, 1)):
                sample = data['sample']
                labels = data['label']

                # Wrap them in a Variable object
                if use_cuda:
                    sample, labels = sample.cuda(), labels.cuda()
                sample, labels = Variable(sample), Variable(labels)

                # Forward pass
                val_outputs = CNN.forward(sample)

                val_loss_size = loss(val_outputs, labels.float())
                total_val_loss += val_loss_size.item()
                val_loss_save.append(val_loss_size.item())

                del data, sample, labels, val_outputs, val_loss_size

        # save the loss for each epoch
        train_path = os.path.join(save_parameters_path, 'train_loss.npy')
        val_path = os.path.join(save_parameters_path, 'val_loss.npy')
        np.save(train_path, train_loss_save)
        np.save(val_path, val_loss_save)

        print(' ')
        print("Training batch loss: {:.4f}".format(total_train_loss / len(train_loader)),
              ", Total training loss: {:.4f}".format(total_train_loss))
        print("Validation batch loss: {:.4f}".format(total_val_loss / len(val_loader)),
              ", Total training loss: {:.4f}".format(total_val_loss))
        print("Epoch time: {:.2f}s".format(time.time() - start_time))
        print(' ')

        # see if validation loss has decreased, if it has a checkpoint will be saved of the current model.
        early_stopping(epoch, total_train_loss, total_val_loss, CNN, optimizer)

        # If the validation has not improved in patience # of epochs the training loop will break.
        if early_stopping.early_stop:
            print("Early stopping")
            break


if __name__ == '__main__':
    main()
