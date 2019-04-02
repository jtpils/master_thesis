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
        self.csv_labels = csv_file

        csv = pd.read_csv(csv_file)
        self.length = len(csv)
        del csv

        self.sample_dir = sample_dir


    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        #idx = idx + 1
        sample_file = os.path.join(self.sample_dir, str(idx))
        #t1 = time.time()
        sample = np.load(sample_file + '.npy')
        #t2 = time.time()
        sample = torch.from_numpy(sample).float()


        labels_csv = pd.read_csv(self.csv_labels)
        labels = labels_csv.iloc[idx-1, 1:4]

        training_sample = {'sample': sample, 'labels': labels.values}

        #print('get sample: ', t2-t1)
        del sample, labels_csv
        return training_sample
