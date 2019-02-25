import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import Dataset


class LiDARDataSet(Dataset):
    """Lidar sample dataset."""

    def __init__(self, csv_file, sample_dir, use_gpu):
        """
        Args:
            csv_file (string): Path to the csv file with labels.
            input_dir (string): Directory with all the samples.
        """

        self.csv_labels = pd.read_csv(csv_file)
        self.sample_dir = sample_dir
        self.use_gpu = use_gpu

    def __len__(self):
        return len(self.csv_labels)

    def __getitem__(self, idx):

        sample_file = os.path.join(self.sample_dir, str(idx))

        if self.use_gpu:  # gpu
            sample = torch.from_numpy(np.load(sample_file + '.npy')).cuda.float()
        else:  # cpu
            sample = torch.from_numpy(np.load(sample_file + '.npy')).float()

        labels = self.csv_labels.iloc[idx-1, 1:4]
        training_sample = {'sample': sample, 'labels': labels.to_numpy()}

        return training_sample
