import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import Dataset
from torch.autograd import Variable
import time


class LiDARDataSet(Dataset):
    """Lidar sample dataset."""

    def __init__(self, csv_file, sample_dir, use_cuda):
        """
        Args:
            csv_file (string): Path to the csv file with labels.
            sample_dir (string): Directory with all the samples.
        """
        #self.num_samples = len(pd.read_csv(csv_file))
        self.csv_labels = csv_file#pd.read_csv(csv_file)
        self.sample_dir = sample_dir
        self.use_cuda = use_cuda

    def __len__(self):
        return 1400 #self.num_samples

    def __getitem__(self, idx):
        sample_file = os.path.join(self.sample_dir, str(idx))

        #t1 = time.time()
        sample = torch.from_numpy(np.load(sample_file + '.npy')).float()
        #t2 = time.time()
        #sample2 = torch.load(sample_file + '.pt').float()
        #t3 = time.time()
        #print('npy: ', t2-t1, 'torch: ', t3-t2)
        #print('to load one sample: ', t2-t1)

        labels_csv = pd.read_csv(self.csv_labels)
        labels = labels_csv.iloc[idx-1, 1:4]
        #labels = self.csv_labels.iloc[idx-1, 1:4]

        training_sample = {'sample': sample, 'labels': labels.values}  #  This worked on Sabinas Mac.
        # training_sample = {'sample': sample, 'labels': labels.to_numpy()}

        del labels, labels_csv, sample, sample_file
        return training_sample
