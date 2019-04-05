import time
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from cat_networks import *
import torch
from torch.utils.data import Dataset
from generate_on_the_go.functions_for_smaller_data import *
from torch.utils.data.sampler import SubsetRandomSampler


def create_loss_and_optimizer(net, learning_rate=0.01):
    # Loss function
    # loss = torch.nn.CrossEntropyLoss()
    loss = torch.nn.MSELoss()
    #loss = torch.nn.SmoothL1Loss()

    # Optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)#, weight_decay=1e-5)
    # optimizer = optim.Adagrad(net.parameters(), lr=learning_rate, lr_decay=1e-3)
    # optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)
    return loss, optimizer


class DataSetGenerateOnTheGo(Dataset):
    """Lidar sample dataset."""

    def __init__(self, sample_path, csv_path, translation, rotation):
        """
        Args:
            sample_path (string): Directory with all the sweeps.
            csv_path with global coordinates
        """
        self.sample_dir = sample_path
        self.sweeps_file_names = os.listdir(sample_path)
        self.length = len(self.sweeps_file_names)
        self.csv_path = csv_path
        self.translation = translation
        self.rotation = rotation

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        file_name = self.sweeps_file_names[idx]
        pc, global_coords = load_data(os.path.join(self.sample_dir,file_name), self.csv_path)
        rand_trans = random_rigid_transformation(self.translation, self.rotation)

        rand_trans = np.array((10,10,0))

        # sweep
        sweep = training_sample_rotation_translation(pc, rand_trans)
        sweep = trim_pointcloud(sweep)
        sweep_image = discretize_pc_fast(sweep)
        # fake a map cutout
        cutout = trim_pointcloud(pc)
        cutout_image = discretize_pc_fast(cutout)

        # if we want to try occupancy grid, uncomment below:
        # sweep_image[sweep_image > 0] = 1
        # cutout_image[cutout_image > 0] = 1

        # concatenate the sweep and the cutout image into one image and save.
        sweep_and_cutout_image = np.concatenate((sweep_image, cutout_image))
        sweep_and_cutout_image = normalize_sample(sweep_and_cutout_image)

        training_sample = {'sample': torch.from_numpy(sweep_and_cutout_image).float(), 'labels': rand_trans}
        return training_sample


def get_loaders(batch_size, translation, rotation, use_cuda):
    # Training
    sample_path = '/home/master04/Desktop/Ply_files/_out_Town01_190402_1/pc/'
    csv_path = '/home/master04/Desktop/Ply_files/_out_Town01_190402_1/Town01_190402_1.csv'
    training_data_set = DataSetGenerateOnTheGo(sample_path, csv_path, translation, rotation)
    kwargs = {'pin_memory': True} if use_cuda else {}
    workers_train = 0
    print('Number of workers: ', workers_train)
    n_training_samples = len(training_data_set)
    print('Number of training samples: ', n_training_samples)
    train_sampler = SubsetRandomSampler(np.arange(n_training_samples, dtype=np.int64))
    train_loader = torch.utils.data.DataLoader(training_data_set, batch_size=batch_size, sampler=train_sampler, num_workers=workers_train, **kwargs)

    # validation
    sample_path = '/home/master04/Desktop/Ply_files/validation_and_test/validation_set/pc/'
    csv_path = '/home/master04/Desktop/Ply_files/validation_and_test/validation_set/validation_set.csv'
    val_data_set = DataSetGenerateOnTheGo(sample_path, csv_path, translation, rotation)
    kwargs = {'pin_memory': True} if use_cuda else {}
    n_val_samples = len(val_data_set)
    print('Number of validation samples: ', n_val_samples)
    val_sampler = SubsetRandomSampler(np.arange(n_val_samples, dtype=np.int64))
    val_loader = torch.utils.data.DataLoader(val_data_set, batch_size=batch_size, sampler=val_sampler, num_workers=workers_train, **kwargs)

    return train_loader, val_loader


def train_network(n_epochs, learning_rate, patience, use_cuda, batch_size, load_weights, load_weights_path, translation, rotation):

    CNN = Duchess()
    print('=======> NETWORK NAME: =======> ', CNN.name())
    if use_cuda:
        CNN.cuda()
    print('Are model parameters on CUDA? ', next(CNN.parameters()).is_cuda)
    print(' ')

    train_loader, val_loader = get_loaders(batch_size, translation, rotation, use_cuda)

    # Load weights
    if load_weights:
        print('Loading parameters...')
        network_param = torch.load(load_weights_path)
        CNN.load_state_dict(network_param['model_state_dict'])

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
    #early_stopping = EarlyStopping(folder_path, patience, verbose=True)

    # Get training data
    n_batches = len(train_loader)
    val_batches = len(val_loader)

    # Create our loss and optimizer functions
    loss, optimizer = create_loss_and_optimizer(CNN, learning_rate)

    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    # Time for printing
    training_start_time = time.time()
    train_loss_save = [len(train_loader)]  # append train loss for each mini batch later on, save this information to plot correctly
    val_loss_save = []  # [len(val_loader)]

    # Loop for n_epochs
    for epoch in range(n_epochs):
        scheduler.step()
        params = optimizer.state_dict()['param_groups']
        print(' ')
        print('learning rate: ', params[0]['lr'])

        running_loss = 0.0
        print_every = 1#n_batches // 5  # how many mini-batches if we want to print stats x times per epoch
        start_time = time.time()
        total_train_loss = 0

        CNN = CNN.train()
        time_epoch = time.time()
        t1_get_data = time.time()
        for i, data in enumerate(train_loader, 1):
            sample = data['sample']
            labels = data['labels']

            if use_cuda:
                sample, labels = sample.cuda(async=True), labels.cuda(async=True)
            sample, labels = Variable(sample), Variable(labels)
            t2_get_data = time.time()
            #print('get data: ', t2_get_data-t1_get_data)

            t1 = time.time()
            # Set the parameter gradients to zero
            optimizer.zero_grad()
            # Forward pass, backward pass, optimize
            outputs = CNN.forward(sample)
            loss_size = loss(outputs, labels.float())
            loss_size.backward()
            optimizer.step()
            t2 = time.time()
            #print('update weights: ', t2-t1)

            # Print statistics
            running_loss += loss_size.item()
            total_train_loss += loss_size.item()

            train_loss_save.append(loss_size.item())

            if True: #(i+1) % print_every == 0:
                print('Epoch [{}/{}], Batch [{}/{}], Loss: {:.4f}, Time: '
                       .format(epoch+1, n_epochs, i, n_batches, running_loss/print_every), time.time()-time_epoch)
                running_loss = 0.0
                time_epoch = time.time()

            del data, sample, labels, outputs, loss_size
            t1_get_data = time.time()


        print('===== Validation =====')
        # At the end of the epoch, do a pass on the validation set
        total_val_loss = 0
        CNN = CNN.eval()
        val_time = time.time()
        with torch.no_grad():
            for i, data in enumerate(val_loader, 1):
                sample = data['sample']
                labels = data['labels']

                # Wrap them in a Variable object
                if use_cuda:
                    sample, labels = sample.cuda(), labels.cuda()
                sample, labels = Variable(sample), Variable(labels)

                # Forward pass
                val_outputs = CNN.forward(sample)

                val_loss_size = loss(val_outputs, labels.float())
                total_val_loss += val_loss_size.item()

                val_loss_save.append(val_loss_size.item())

                if (i+1) % 5 == 0:
                    print('Validation: Batch [{}/{}], Time: '.format(i, val_batches), time.time()-val_time)
                    val_time = time.time()

                del data, sample, labels, val_outputs, val_loss_size

        print(' ')
        print("Training loss: {:.4f}".format(total_train_loss / len(train_loader)),
              ", Validation loss: {:.4f}".format(total_val_loss / len(val_loader)),
              ", Time: {:.2f}s".format(time.time() - start_time))
        print(' ')
        # save the loss for each epoch
        train_loss.append(total_train_loss / len(train_loader))
        val_loss.append(total_val_loss / len(val_loader))

        '''train_path = folder_path + '/train_loss.npy'
        val_path = folder_path + '/val_loss.npy'
        np.save(train_path, train_loss_save)
        np.save(val_path, val_loss_save)

        # see if validation loss has decreased, if it has a checkpoint will be saved of the current model.
        early_stopping(epoch, total_train_loss, total_val_loss, CNN, optimizer)

        # If the validation has not improved in patience # of epochs the training loop will break.
        if early_stopping.early_stop:
            print("Early stopping")
            break'''

    print("Training finished, took {:.2f}s".format(time.time() - training_start_time))
    return train_loss, val_loss



def main():
    load_weights = False
    load_weights_path = '/home/annika_lundqvist144/master_thesis/BEV_network/Duchess_190402_1/parameters/epoch_11_checkpoint.pt'

    #save_parameters_folder = input('Type name of new folder: ')
    n_epochs = 50
    learning_rate = 0.01
    patience = 50
    batch_size = 2
    translation, rotation = 0, 0

    print(' ')
    print('Number of GPUs available: ', torch.cuda.device_count())
    use_cuda = torch.cuda.is_available()
    print('CUDA available: ', use_cuda)


    # create directory for model weights
    #current_path = os.getcwd()
    #save_parameters_path = os.path.join(current_path, save_parameters_folder)
    #os.mkdir(save_parameters_path)
    #parameter_path = os.path.join(save_parameters_path, 'parameters')
    #os.mkdir(parameter_path)

    # train!
    train_loss, val_loss = train_network(n_epochs, learning_rate, patience, use_cuda, batch_size,
                                         load_weights, load_weights_path, translation, rotation)

if __name__ == '__main__':
    main()
