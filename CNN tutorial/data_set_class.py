import pandas as pd
import numpy as np
import os
#import io
import torch
from torch.utils.data import Dataset



class Lidar_data_set(Dataset):
    """Lidar sweep cutouts dataset."""

    def __init__(self, csv_file, sweeps_dir, cutouts_dir):
        """
        Args:
            csv_file (string): Path to the csv file with labels.
            sweeps_dir (string): Directory with all the sweeps.
            cutouts_dir (string): Directory with all the cutouts.
        """

        self.csv_labels = pd.read_csv(csv_file)
        #self.first_frame_number =
        self.sweeps_dir = sweeps_dir
        self.cutouts_dir = cutouts_dir

    def __len__(self):
        return len(self.csv_labels)

    def __getitem__(self, idx):
        sweep_name = os.path.join(self.sweeps_dir, str(idx))
        sweep = torch.from_numpy(np.load(sweep_name + '.npy')).float()
        #sweep = np.reshape(sweep, (1, 4, 600, 600))

        cutout_name = os.path.join(self.cutouts_dir, str(idx))
        cutout = torch.from_numpy(np.load(cutout_name + '.npy')).float()
        #cutout = np.reshape(cutout, (1, 4, 900, 900))

        labels = self.csv_labels.iloc[idx-1, 1:4]

        sample = {'sweep': sweep, 'cutout': cutout, 'labels': labels.to_numpy()}


        return sample
