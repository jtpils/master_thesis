import torch
from data_loader import get_train_loader
from models import *
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from early_stopping import EarlyStopping
import os
from tqdm import tqdm


model_name = input('Type name of new folder: ')
n_epochs = 20
learning_rate = 0.01
patience = 10
batch_size = 4
translation = 1 # float(input('Enter translation in metres: '))
rotation = 1 # float(input('Enter rotation in degrees: '))

print(' ')
print('Number of GPUs available: ', torch.cuda.device_count())
use_cuda = torch.cuda.is_available()
print('CUDA available: ', use_cuda)

# create directory for model weights
current_path = os.getcwd()
model_path = os.path.join(current_path, model_name)
os.mkdir(model_path)
parameter_path = os.path.join(model_path, 'parameters')
os.mkdir(parameter_path)

if use_cuda:
    data_set_path_train = '/home/annika_lundqvist144/ply_files/_out_Town01_190402_1/pc'
    csv_path_train = '/home/annika_lundqvist144/ply_files/_out_Town01_190402_1/Town01_190402_1.csv'
    grid_csv_path_train = '/home/annika_lundqvist144/csv_grids_190409/csv_grids_training/'
    data_set_path_val = '/home/annika_lundqvist144/ply_files/validation_set/pc'
    csv_path_val = '/home/annika_lundqvist144/ply_files/validation_set/validation_set.csv'
    grid_csv_path_val = '/home/annika_lundqvist144/csv_grids_190409/csv_grids_validation/'
else:
    data_set_path_train = '/home/master04/Desktop/Ply_files/_out_Town01_190402_1/pc/'
    csv_path_train = '/home/master04/Desktop/Ply_files/_out_Town01_190402_1/Town01_190402_1.csv'
    grid_csv_path_train = '/home/master04/Desktop/Dataset/ply_grids/csv_grids_190409/csv_grids_training'
    data_set_path_val = '/home/master04/Desktop/Ply_files/validation_and_test/validation_set/pc/'
    csv_path_val = '/home/annika_lundqvist144/ply_files/validation_set/validation_set.csv'
    grid_csv_path_val = '/home/master04/Desktop/Dataset/ply_grids/csv_grids_190409/csv_grids_validation'

kwargs = {'num_workers': 8, 'pin_memory':True} if use_cuda else {'num_workers': 0}
train_loader, val_loader = get_train_loader(batch_size, data_set_path_train, csv_path_train, grid_csv_path_train, data_set_path_val,
                     csv_path_val, grid_csv_path_val, translation, rotation, kwargs)

net = OurPointPillars(batch_size, use_cuda)
print('=======> NETWORK NAME: =======> ', net.name())
if use_cuda:
    net.cuda()
else:
    batch_size = 2

split_loss = True
if split_loss:
    alpha = float(input('Enter weight for alpha in custom loss: '))
    beta = 1-alpha
    loss_trans = torch.nn.MSELoss()
    loss_rot = torch.nn.SmoothL1Loss()
else:
    loss = torch.nn.MSELoss()


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
train_loss_save = [len(train_loader)]  # append train loss for each mini batch later on, save this information to plot correctly
val_loss_save = [len(val_loader)]

# initialize the early_stopping object
early_stopping = EarlyStopping(parameter_path, patience, verbose=True)

# Get training data
n_batches = len(train_loader)
print('Number of batches: ', n_batches)
val_batches = len(val_loader)

optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True, threshold=0.0001,
                              threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)

for epoch in range(n_epochs):
    params = optimizer.state_dict()['param_groups']
    print(' ')
    print('learning rate: ', params[0]['lr'])

    running_loss = 0.0
    print_every = n_batches // 5  # how many mini-batches if we want to print stats x times per epoch
    start_time = time.time()
    batch_time = time.time()
    total_train_loss = 0.0

    net = net.train()
    get_data_1 = time.time()
    for i, data in enumerate(train_loader, 1):
        sweep = data['sweep']
        sweep_coordinates = data['sweep_coordinates']
        cutout = data['cutout']
        cutout_coordinates = data['cutout_coordinates']
        label = data['label']
    if use_cuda:
        sweep, sweep_coordinates, cutout, cutout_coordinates, label = sweep.cuda(async=True), \
                                                                       sweep_coordinates.cuda(async=True), \
                                                                       cutout.cuda(async=True), \
                                                                       cutout_coordinates.cuda(async=True), \
                                                                       label.cuda(async=True)

    sweep, sweep_coordinates, cutout, cutout_coordinates, label = Variable(sweep), Variable(sweep_coordinates), \
                                                             Variable(cutout), Variable(cutout_coordinates), \
                                                                   Variable(label)

    # Set the parameter gradients to zero
    optimizer.zero_grad()
    # Forward pass, backward pass, optimize
    outputs = net.forward(sweep.float(), cutout.float(), sweep_coordinates.float(), cutout_coordinates.float())

    output_size = outputs.size()[0]

    if split_loss:
        loss_trans_size = loss_trans(outputs[:,0:2], label[:,0:2].float())
        loss_rot_size = loss_rot(outputs[:,-1].reshape((output_size,1)), label[:,-1].reshape((output_size,1)).float())
        loss_size = alpha*loss_trans_size + beta*loss_rot_size
    else:
        loss_size = loss(outputs, label.float())

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
    get_data_1 = time.time()
    del data, sweep, cutout, outputs, loss_size

    print('===== Validation =====')
    total_val_loss = 0
    net = net.eval()
    val_time = time.time()
    with torch.no_grad():
        print('number of iterations: ', val_batches)
        for i, data in tqdm(enumerate(val_loader, 1)):
            sweep = data['sweep']
            sweep_coordinates = data['sweep_coordinates']
            cutout = data['cutout']
            cutout_coordinates = data['cutout_coordinates']
            label = data['label']
        if use_cuda:
            sweep, sweep_coordinates, cutout, cutout_coordinates, label = sweep.cuda(async=True), \
                                                                           sweep_coordinates.cuda(async=True), \
                                                                           cutout.cuda(async=True), \
                                                                           cutout_coordinates.cuda(async=True), \
                                                                           label.cuda(async=True)

        sweep, sweep_coordinates, cutout, cutout_coordinates, label = Variable(sweep), Variable(sweep_coordinates), \
                                                                 Variable(cutout), Variable(cutout_coordinates), \
                                                                   Variable(label)

        # Forward pass
        val_outputs = net.forward(sweep.float(), cutout.float(), sweep_coordinates.float(), cutout_coordinates.float())
        output_size = val_outputs.size()[0]

        if split_loss:
            loss_trans_size = loss_trans(val_outputs[:,0:2], label[:,0:2].float())
            loss_rot_size = loss_rot(val_outputs[:,-1].reshape((output_size,1)), label[:,-1].reshape((output_size,1)).float())
            val_loss_size = alpha*loss_trans_size + beta*loss_rot_size
        else:
            val_loss_size = loss(val_outputs, label.float())

        total_val_loss += val_loss_size.item()
        val_loss_save.append(val_loss_size.item())

        del data, sweep, cutout, label, val_outputs, val_loss_size

    scheduler.step(total_val_loss)

    # save the loss for each epoch
    train_path = os.path.join(model_path, 'train_loss.npy')
    val_path = os.path.join(model_path, 'val_loss.npy')
    np.save(train_path, train_loss_save)
    np.save(val_path, val_loss_save)

    print(' ')
    print("Training batch loss: {:.4f}".format(total_train_loss / len(train_loader)),
          ", Total training loss: {:.4f}".format(total_train_loss))
    print("Validation batch loss: {:.4f}".format(total_val_loss / len(val_loader)),
          ", Total validation loss: {:.4f}".format(total_val_loss))
    print("Epoch time: {:.2f}s".format(time.time() - start_time))
    print(' ')

    # see if validation loss has decreased, if it has a checkpoint will be saved of the current model.
    early_stopping(epoch, total_train_loss, total_val_loss, net, optimizer)

    # If the validation has not improved in patience # of epochs the training loop will break.
    if early_stopping.early_stop:
        print("Early stopping")
        break


