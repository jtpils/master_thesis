import time
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import numpy as np
import os
from early_stopping import EarlyStopping
import torch
from point_cloud_net import *
from data_set_pc_samples import *


def create_loss_and_optimizer(net, learning_rate=0.001):

    # Loss function
    loss = torch.nn.SmoothL1Loss()
    # Optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-5)

    return loss, optimizer


# def train_network(net, train_loader, val_loader, n_epochs, learning_rate, patience, folder_path, device, use_cuda):
def train_network(n_epochs, learning_rate, patience, folder_path, use_cuda, batch_size):

    data_set_path = '/home/annika_lundqvist144/pc_samples/training_samples_190403'


    net = PointPillars(batch_size)
    print('=======> NETWORK NAME: =======> ', net.name())
    if use_cuda:
        net.cuda()
    #print('Are model parameters on CUDA? ', next(net.parameters()).is_cuda)
    print(' ')

    train_loader = get_train_loader_pc(batch_size, data_set_path, {'num_workers': 8})

    '''# Load weights
    if load_weights:
        print('Loading parameters...')
        network_param = torch.load(load_weights_path)
        net.load_state_dict(network_param['model_state_dict'])'''

    # Print all of the hyperparameters of the training iteration:
    print("===== HYPERPARAMETERS =====")
    print("batch_size =", batch_size)
    print("epochs =", n_epochs)
    print("initial learning_rate =", learning_rate)
    print('patience:', patience)
    print("=" * 27)

    # declare variables for storing validation and training loss to return
    val_loss = []
    train_loss = []

    # initialize the early_stopping object
    early_stopping = EarlyStopping(folder_path, patience, verbose=True)

    # Get training data
    n_batches = len(train_loader)
    print('Number of batches: ', n_batches)

    # Create our loss and optimizer functions
    loss, optimizer = create_loss_and_optimizer(net, learning_rate)
    scheduler = StepLR(optimizer, step_size=100, gamma=0.5)

    # Time for printing
    training_start_time = time.time()

    # Loop for n_epochs
    for epoch in range(n_epochs):
        scheduler.step()
        params = optimizer.state_dict()['param_groups']
        print(' ')
        print('learning rate: ', params[0]['lr'])

        running_loss = 0.0
        print_every = 13  # n_batches // 10  # how many mini-batches if we want to print stats x times per epoch
        start_time = time.time()
        total_train_loss = 0

        net = net.train()
        time_epoch = time.time()
        t1_get_data = time.time()
        for i, data in enumerate(train_loader, 1):
            t2_get_data = time.time()
            print('get data from loader: ', t2_get_data-t1_get_data)


            # The training samples contains 5 things. 1. sweep features (xp,yp,z) 2. sweep coordinates (x,y,z)
            # 3. map features (xp,yp,z) 4. map coordinates (x,y,z) 5. labels.
            sweep = data['sweep']
            sweep_coordinates = data['sweep_coordinates']
            cutout = data['cutout']
            cutout_coordinates = data['cutout_coordinates']
            labels = data['labels']


            if use_cuda:
                sweep, sweep_coordinates, cutout, cutout_coordinates, labels = sweep.cuda(async=True), \
                                                                               sweep_coordinates.cuda(async=True), \
                                                                               cutout.cuda(async=True), \
                                                                               cutout_coordinates.cuda(async=True), \
                                                                               labels.cuda(async=True)

            sweep, sweep_coordinates, cutout, cutout_coordinates, labels = Variable(sweep), Variable(sweep_coordinates), \
                                                                     Variable(cutout), Variable(cutout_coordinates), \
                                                                           Variable(labels)


            # Set the parameter gradients to zero
            optimizer.zero_grad()
            # Forward pass, backward pass, optimize
            #t1 = time.time()
            outputs = net.forward(sweep.float(), cutout.float(), sweep_coordinates.float(), cutout_coordinates.float())#, scatter)
            #t2 = time.time()
            #print('time for forward: ', t2 - t1)

            #t1 = time.time()
            loss_size = loss(outputs, labels.float())
            #t2 = time.time()
            #print('time for get loss size: ', t2 - t1)

            #t1 = time.time()
            loss_size.backward()
            #t2 = time.time()
            #print('time for backprop: ', t2-t1)

            #t1 = time.time()
            optimizer.step()
            #t2 = time.time()
            #print('update: ', t2-t1)

            # Print statistics
            running_loss += loss_size.item()
            total_train_loss += loss_size.item()

            if True:#(i + 1) % print_every == 0:
                print('Epoch [{}/{}], Batch [{}/{}], Loss: {:.4f}, Time: '
                      .format(epoch + 1, n_epochs, i, n_batches, running_loss / print_every), time.time() - time_epoch)
                running_loss = 0.0
                time_epoch = time.time()

            #t1_get_data = time.time()
            del data, sweep, cutout, labels, outputs, loss_size
            t1_get_data = time.time()
        # At the end of the epoch, do a pass on the validation set
        '''
        total_val_loss = 0
        net = net.eval()
        with torch.no_grad():
            for data in val_loader:
                sample = data['sample']
                labels = data['labels']

                # Wrap them in a Variable object
                #sample, labels = Variable(sample).to(device), Variable(labels).to(device)

                if use_cuda:
                    sample, labels = sample.cuda(), labels.cuda()
                sample, labels = Variable(sample), Variable(labels)

                # Forward pass
                val_outputs = net.forward(sample)

                val_loss_size = loss(val_outputs, labels.float())
                total_val_loss += val_loss_size.item()

                del data, sample, labels, val_outputs, val_loss_size

        print("Training loss: {:.4f}".format(total_train_loss / len(train_loader)),
              ", Validation loss: {:.4f}".format(total_val_loss / len(val_loader)),
              ", Time: {:.2f}s".format(time.time() - start_time))
        print(' ')
        # save the loss for each epoch
        train_loss.append(total_train_loss / len(train_loader))
        val_loss.append(total_val_loss / len(val_loader))

        # see if validation loss has decreased, if it has a checkpoint will be saved of the current model.
        early_stopping(epoch, total_train_loss, total_val_loss, net, optimizer)

        # If the validation has not improved in patience # of epochs the training loop will break.
        if early_stopping.early_stop:
            print("Early stopping")
            break'''

    print("Training finished, took {:.2f}s".format(time.time() - training_start_time))
    return train_loss, val_loss
