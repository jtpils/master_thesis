import torch
from networks import *
from DataSets import *
from torch.optim.lr_scheduler import StepLR
import time
from torch.autograd import Variable


def main():
    n_epochs = 50 #int(input('Number of epochs: '))
    learning_rate = 0.01 #float(input('Learning rate: '))
    patience = 50 #int(input('Input patience for EarlyStopping: ')) # Threshold for early stopping. Number of epochs that we will wait until brake
    batch_size = 32 #45  #32 #int(input('Input batch size: '))

    use_cuda = torch.cuda.is_available()
    print('CUDA available: ', use_cuda)

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

    loss = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(CNN.parameters(), lr=learning_rate)

    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    for epoch in range(n_epochs):
        running_loss = 0.0
        print_every = 1 #n_batches // 5  # how many mini-batches if we want to print stats x times per epoch
        start_time = time.time()
        batch_time = time.time()
        total_train_loss = 0.0
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

            if (i+1) % print_every == 0:
                print('Epoch [{}/{}], Batch [{}/{}], Loss: {:.4f}, Time: '
                       .format(epoch+1, n_epochs, i, n_batches, running_loss/print_every), time.time()-batch_time)
                running_loss = 0.0
                batch_time = time.time()

        del data, sample, labels, outputs, loss_size

        # validation pass



        print("Training batch loss: {:.4f}".format(total_train_loss / len(train_loader)),
              "Total training loss: {:.4f}".format(total_train_loss),
              ", Time: {:.2f}s".format(time.time() - start_time))
        print(' ')





if __name__ == '__main__':
    main()
