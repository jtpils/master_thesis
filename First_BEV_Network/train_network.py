from functions import *
import time
from torch.autograd import Variable
import numpy as np
import os


def train_network(net, train_loader, val_loader, n_epochs, learning_rate, folder_path, use_gpu):
    # Print all of the hyperparameters of the training iteration:
    print("===== HYPERPARAMETERS =====")
    # print("batch_size =", batch_size)
    print("epochs =", n_epochs)
    print("learning_rate =", learning_rate)
    print("=" * 30)

    # declare variables for storing validation and training loss
    val_loss = []
    train_loss = []

    # Get training data
    n_batches = len(train_loader)
    print('Number of batches: ', n_batches)

    # Create our loss and optimizer functions
    loss, optimizer = create_loss_and_optimizer(net, learning_rate)

    # Time for printing
    training_start_time = time.time()

    # Loop for n_epochs
    for epoch in range(n_epochs):

        running_loss = 0.0
        print_every = n_batches // 1
        start_time = time.time()
        total_train_loss = 0

        for i, data in enumerate(train_loader, 1):
            # Get inputs
            sample = data['sample']
            labels = data['labels']

            # Wrap them in a Variable object
            if use_gpu:
                sample, labels = Variable(sample).cuda(), Variable(labels).cuda()
            else:
                sample, labels = Variable(sample), Variable(labels)

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

            # Print every batch of an epoch
            if i % print_every == 0:
                print("Epoch {}, {:d}% \t train_loss: {:.2f} took: {:.2f}s".format(
                    epoch + 1, int(100 * i / n_batches), running_loss / print_every, time.time() - start_time))
                # Reset running loss and time
                running_loss = 0.0
                start_time = time.time()

        # At the end of the epoch, do a pass on the validation set
        total_val_loss = 0
        for data in val_loader:
            sample = data['sample']
            labels = data['labels']

            # Wrap them in a Variable object
            if use_gpu:
                sample, labels = Variable(sample).cuda(), Variable(labels).cuda()
            else:
                sample, labels = Variable(sample), Variable(labels)

            # Forward pass
            val_outputs = net.forward(sample)

            val_loss_size = loss(val_outputs, labels.float())
            total_val_loss += val_loss_size.item()

        print("Validation loss = {:.2f}".format(total_val_loss / len(val_loader)))

        # save the loss for each epoch
        train_loss.append(total_train_loss)
        val_loss.append(total_val_loss)

        if len(train_loss) > 1 and train_loss[-1] < train_loss[-2]: # if the loss is smaller this epoch (change to validation loss in the future)
            file_name = 'epoch' + str(epoch) + '.pt'
            path = os.path.join(folder_path, file_name)
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'train_loss': total_train_loss,
                'val_loss': total_val_loss
            }, path)

    print("Training finished, took {:.2f}s".format(time.time() - training_start_time))
    return train_loss, val_loss
