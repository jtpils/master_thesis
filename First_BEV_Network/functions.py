import torch.optim as optim
import torch


def output_size(in_size, kernel_size, stride, padding):
    output = int((in_size - kernel_size + 2*padding) / stride) + 1
    return output


# DataLoader takes in a dataset and a sampler for loading (num_workers deals with system level memory)
# def get_train_loader(batch_size, train_set, train_sampler):
#    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=train_sampler, num_workers=2)
#    return train_loader


def create_loss_and_optimizer(net, learning_rate=0.001):
    # Loss function
    #loss = torch.nn.CrossEntropyLoss()
    loss = torch.nn.MSELoss()

    # Optimizer
    #optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    optimizer = optim.Adagrad(net.parameters(), lr=learning_rate, lr_decay=1e-3)

    #optimizer = optim.SGD(net.parameters(), lr=learning_rate)

    return loss, optimizer
