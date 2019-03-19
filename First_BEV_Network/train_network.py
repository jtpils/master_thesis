from functions import *
import time
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
import numpy as np
import os
from early_stopping import EarlyStopping
import torch


def create_loss_and_optimizer(net, learning_rate=0.001):
    # Loss function
    # loss = torch.nn.CrossEntropyLoss()
    # loss = torch.nn.MSELoss()
    loss = torch.nn.SmoothL1Loss()

    # Optimizer
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-5)
    # optimizer = optim.Adagrad(net.parameters(), lr=learning_rate, lr_decay=1e-3)
    # optimizer = optim.SGD(net.parameters(), lr=learning_rate)

    return loss, optimizer


def train_network(net, train_loader, val_loader, n_epochs, learning_rate, patience, folder_path, device, use_cuda):

    # Print all of the hyperparameters of the training iteration:
    print("===== HYPERPARAMETERS =====")
    # print("batch_size =", batch_size)
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
    # print('Number of batches: ', n_batches)

    # Create our loss and optimizer functions
    loss, optimizer = create_loss_and_optimizer(net, learning_rate)
    scheduler = StepLR(optimizer, step_size=20, gamma=0.5)

    # Time for printing
    training_start_time = time.time()

    # Loop for n_epochs
    for epoch in range(n_epochs):
        scheduler.step()
        params = optimizer.state_dict()['param_groups']
        print('learning rate: ', params[0]['lr'])

        running_loss = 0.0
        print_every = 5  #n_batches // 10  # how many mini-batches if we want to print stats x times per epoch
        start_time = time.time()
        total_train_loss = 0

        net.train()
        time_epoch = time.time()
        for i, data in enumerate(train_loader, 1):
            # Get inputs
            sample = data['sample']
            labels = data['labels']

            
            if use_cuda:
                sample, labels = sample.cuda(), labels.cuda()
            sample, labels = Variable(sample), Variable(labels)
            # Wrap them in a Variable object
            #sample, labels = Variable(sample).to(device), Variable(labels).to(device)

            # Set the parameter gradients to zero
            optimizer.zero_grad()

            # Forward pass, backward pass, optimize
            outputs = net.forward(sample)
            loss_size = loss(outputs, labels.float())
            loss_size.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss_size.item()
            total_train_loss += loss_size.item()
    
            if (i+1) % print_every == 0:
                print('Epoch [{}/{}], Batch [{}/{}], Loss: {:.4f}, Time: '
                       .format(epoch+1, n_epochs, i+1, n_batches, running_loss/print_every), time.time()-time_epoch)
                running_loss = 0.0
                time_epoch = time.time()
        
            del data, sample, labels, outputs

        # At the end of the epoch, do a pass on the validation set
        total_val_loss = 0
        net.eval()
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

                del data, sample, labels, val_outputs

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
            break

    print("Training finished, took {:.2f}s".format(time.time() - training_start_time))
    return train_loss, val_loss
