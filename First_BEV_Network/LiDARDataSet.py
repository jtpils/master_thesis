import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import Dataset
from torch.autograd import Variable


class LiDARDataSet(Dataset):
    """Lidar sample dataset."""

    def __init__(self, csv_file, sample_dir, use_cuda):
        """
        Args:
            csv_file (string): Path to the csv file with labels.
            sample_dir (string): Directory with all the samples.
        """

        self.csv_labels = pd.read_csv(csv_file)
        self.sample_dir = sample_dir
        self.use_cuda = use_cuda

    def __len__(self):
        return len(self.csv_labels)

    def __getitem__(self, idx):

        sample_file = os.path.join(self.sample_dir, str(idx))

        sample = torch.from_numpy(np.load(sample_file + '.npy')).float()
        labels = self.csv_labels.iloc[idx-1, 1:4]

        training_sample = {'sample': sample, 'labels': labels.values}  #  This worked on Sabinas Mac.
        # training_sample = {'sample': sample, 'labels': labels.to_numpy()}

        return training_sample
