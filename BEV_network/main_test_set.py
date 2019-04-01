from data_loader import get_test_loader
import matplotlib.pyplot as plt
import torch
from cat_networks import *
from torch.autograd import Variable
import torch
from torch.utils.data.sampler import SequentialSampler
from LiDARDataSet import LiDARDataSet

load_weights = True
path = '/home/master04/Desktop/network_parameters/Duchess_190329_2/parameters'
load_weights_path = path + '/epoch_22_checkpoint.pt'
path_test_data = '/home/master04/Desktop/Dataset/BEV_samples/Res_01/fake_test_set'#'/home/master04/Desktop/Dataset/BEV_samples/fake_test_set'
batch_size = 2 #int(input('Input batch size: '))

print('Number of GPUs available: ', torch.cuda.device_count())
use_cuda = torch.cuda.is_available()
print('CUDA available: ', use_cuda)

CNN = Duchess()
print('=======> NETWORK NAME: =======> ', CNN.name())
if use_cuda:
    CNN.cuda()
print('Are model parameters on CUDA? ', next(CNN.parameters()).is_cuda)
print(' ')
if load_weights:
    print('Loading parameters...')
    if use_cuda:
        network_param = torch.load(load_weights_path)
        CNN.load_state_dict(network_param['model_state_dict'])
    else:
        network_param = torch.load(load_weights_path, map_location='cpu')
        CNN.load_state_dict(network_param['model_state_dict'])

csv_file = path_test_data + '/labels.csv'
sample_dir = path_test_data + '/samples/'
test_data_set = LiDARDataSet(csv_file, sample_dir, use_cuda)

kwargs = {'pin_memory': True} if use_cuda else {}
workers = 8
print('Number of workers: ', workers)

n_test_samples = len(test_data_set)
print('Number of training samples: ', n_test_samples)
test_sampler = SequentialSampler(np.arange(1, n_test_samples+1, dtype=np.int64))
test_loader = torch.utils.data.DataLoader(test_data_set, batch_size=batch_size, sampler=test_sampler, num_workers=workers, **kwargs)
print(' ')

CNN.eval()

predictions_list = list()
labels_list = list()

loss = torch.nn.SmoothL1Loss()
total_test_loss = 0
CNN = CNN.eval()
with torch.no_grad():
    for i, data in enumerate(test_loader, 1):
        sample = data['sample']
        labels = data['labels']

        # Wrap them in a Variable object
        if use_cuda:
            sample, labels = sample.cuda(), labels.cuda()
        sample, labels = Variable(sample), Variable(labels)

        # Forward pass
        output = CNN.forward(sample)

        test_loss_size = loss(output, labels.float())
        total_test_loss += test_loss_size.item()

        predictions_list.append(output.data.tolist())
        labels_list.append(labels.data.tolist())

        if i%10 == 0:
            print('Batch ', i, ' of ', len(test_loader))

predictions = np.array(predictions_list)  # shape: (minibatch, batch_size, 3)
labels = np.array(labels_list)
predictions = predictions.reshape((n_test_samples, 3))  # shape (samples, 3)
labels = labels.reshape((n_test_samples, 3))
diff = labels - predictions
error_distances = np.hypot(diff[:,0], diff[:,1])

def plot_histograms():
    plt.subplot(1, 3, 1)
    plt.hist(diff[:,0], bins='auto', label='Difference x')
    plt.legend()
    plt.ylabel('Difference in meters: x')

    plt.subplot(1, 3, 2)
    plt.hist(diff[:,1], bins='auto', label='Difference y')
    plt.legend()
    plt.ylabel('Difference in meters: y')

    plt.subplot(1, 3, 3)
    plt.hist(diff[:,2], bins='auto', label='Difference angle')
    plt.legend()
    plt.ylabel('Difference in degrees')

    plt.show()


def visualize_samples():
    sorted = np.argsort(error_distances)
    # best samples
    print('==== Minimum error ====')
    for idx in sorted[:3]:
        data = test_data_set.__getitem__(idx)
        print('Error ', diff[idx,:])
        print('Error distance ', error_distances[idx])
        print('Labels ', labels[idx,:])
        print('Label distance ', np.hypot(labels[idx,0], labels[idx,1]))
        print(' ')

        visualize_detections(data['sample'],layer=0, fig_num=1)
        visualize_detections(data['sample'],layer=1, fig_num=2)
        plt.show()

    print('==== Maximum error ====')
    for idx in sorted[-3:   ]:
        data = test_data_set.__getitem__(idx)
        print('Error ', diff[idx,:])
        print('Error distance ', error_distances[idx])
        print('Labels ', labels[idx,:])
        print('Label distance ', np.hypot(labels[idx,0], labels[idx,1]))
        print(' ')

        visualize_detections(data['sample'],layer=0, fig_num=1)
        visualize_detections(data['sample'],layer=1, fig_num=2)
        plt.show()


def visualize_detections(discretized_point_cloud, layer=0, fig_num=1):
    detection_layer = discretized_point_cloud[layer, :, :]
    detection_layer[detection_layer > 0] = 255

    plt.figure(fig_num)
    plt.imshow(detection_layer, cmap='gray')


def plot_loss():
    model_dict = torch.load(load_weights_path, map_location='cpu')
    train = np.load(path + '/train_loss.npy')
    val = np.load(path + '/val_loss.npy')

    train_batches = train[0]  #first element is the number of minibatches per epoch
    train_loss = train[1:]  #the following elements are the loss for each minibatch
    #val_batches = val[0]
    val_loss = val[1:]

    num_epochs = len(train_loss) / train_batches
    train_vec = np.linspace(1, num_epochs, len(train_loss))
    val_vec = np.linspace(1, num_epochs, len(val_loss))

    plt.plot(train_vec, train_loss, label='Training loss')
    plt.plot(val_vec, val_loss, label='Validation loss')
    plt.xlabel('Epochs')
    plt.legend()
    plt.show()


def plot_labels():
    plt.subplot(1, 3, 1)
    plt.hist(labels[:,0], bins='auto', label='Labels x')
    plt.legend()
    #plt.ylabel('Difference in meters: x')

    plt.subplot(1, 3, 2)
    plt.hist(labels[:,1], bins='auto', label='Labels y')
    plt.legend()
    #plt.ylabel('Difference in meters: y')

    plt.subplot(1, 3, 3)
    plt.hist(labels[:,2], bins='auto', label='Labels angle')
    plt.legend()
    #plt.ylabel('Difference in degrees')

    plt.show()


def main():

    plot_labels()
    plot_histograms()
    visualize_samples()
    plot_loss()


if __name__ == '__main__':
    main()
