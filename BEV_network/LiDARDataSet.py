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
        self.csv_labels = csv_file #
        #self.csv_labels = pd.read_csv(csv_file)
        self.sample_dir = sample_dir

    def __len__(self):
        return 1400 #self.

    def __getitem__(self, idx):
        sample_file = os.path.join(self.sample_dir, str(idx))
        sample = torch.from_numpy(np.load(sample_file + '.npy')).float()

        labels_csv = pd.read_csv(self.csv_labels)
        labels = labels_csv.iloc[idx-1, 1:4]

        #labels = self.csv_labels.iloc[idx-1, 1:4]  #old version when we loaded the whole csv as a self-variable

        training_sample = {'sample': sample, 'labels': labels.values}  #  This worked on Sabinas Mac.

        del sample, labels_csv
        return training_sample # sample, labels.values #training_sample
