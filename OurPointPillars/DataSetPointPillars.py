import torch
from torch.utils.data import Dataset
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import time
from functions import *


class DataSetPointPillars(Dataset):
    """Lidar sample dataset."""

    def __init__(self, sample_dir, csv_path, grid_csv_path, translation=1, rotation=0):

        print('translation: ', translation)
        print('rotation: ', rotation)
        self.sample_dir = sample_dir
        self.csv_path = csv_path
        self.translation = translation
        self.rotation = rotation
        self.sweeps_file_names = sorted(os.listdir(sample_dir))
        self.num_sweeps = 5
        self.grid_edges = np.load(os.path.join(grid_csv_path, 'edges.npy'))

        list_of_csv = os.listdir(grid_csv_path)
        grid_dict = dict()
        print('loading all LiDAR detections...')
        for file in tqdm(list_of_csv):
            if 'grid' in file:
                pc = pd.read_csv(os.path.join(grid_csv_path, file))
                grid_dict[file] = pc
        self.grid_dict = grid_dict
        print('Done loading detections.')
        del grid_dict, pc

    def __len__(self):
        return len(self.sweeps_file_names) - (self.num_sweeps - 1)

    def __getitem__(self, idx):
        # load ply-file
        indices = [x-(self.num_sweeps //2) for x in np.arange(self.num_sweeps)]

        pc_multiple_sweeps = np.zeros((1,3))
        for i in indices:
            file_name = self.sweeps_file_names[idx + i]
            pc, global_coords_temp = load_data(os.path.join(self.sample_dir,file_name), self.csv_path)
            pc = trim_point_cloud_vehicle_ground(pc,  origin=global_coords_temp[:2], remove_vehicle=True, remove_ground=False)
            pc_multiple_sweeps = np.concatenate((pc_multiple_sweeps, pc), axis=0)
            if i==0:
                global_coords = global_coords_temp
        pc_multiple_sweeps = pc_multiple_sweeps[1:,:]

        # rotate and translate sweep
        rand_trans = random_rigid_transformation(self.translation, self.rotation)
        sweep = trim_point_cloud_range(pc_multiple_sweeps, origin=global_coords[:2], trim_range=45)
        sweep = rotate_point_cloud(sweep, global_coords[:2], rand_trans[-1], to_global=False)
        sweep = translate_point_cloud(sweep, rand_trans[:2])
        sweep = trim_point_cloud_range(sweep, origin=global_coords[:2],  trim_range=15)

        sweep_pillars, sweep_coordinates = create_pillars(sweep, origin=global_coords[:2])
        sweep_features , sweep_coordinates = get_feature_tensor(sweep_pillars, sweep_coordinates)


        # map cut-out
        trim_range = 15
        cut_out_coordinates = global_coords[:2]
        grid_x = np.floor((cut_out_coordinates[0] - self.grid_edges[0][0])/trim_range).astype(int)
        grid_y = np.floor((cut_out_coordinates[1] - self.grid_edges[1][0])/trim_range).astype(int)

        #find all neighbouring grids, store the key names in a list
        neighbouring_grids = list()
        for x in [-1, 0, 1]:
            for y in [-1, 0, 1]:
                key = 'grid_' + str(grid_x + x) + '_' + str(grid_y + y) + '.csv' # key name in grid_dict
                neighbouring_grids.append(key)

        # try to read these dict-keys, some may not exist if there are no detections in that grid
        cutout = list()
        for key in neighbouring_grids:
            try:
                pc = self.grid_dict[key]
                cutout.append(pc)
            except: # not all neighbouring grids exixst as a csv-file because there were no detections there. skip those.
                continue
        cutout = pd.concat(cutout)

        # try using numpy arrays directly instead, might be faster?
        cutout = cutout[cutout['x'] <= cut_out_coordinates[0]+trim_range] # NEW IDEA
        cutout = cutout[cutout['x'] >= cut_out_coordinates[0]-trim_range]
        cutout = cutout[cutout['y'] <= cut_out_coordinates[1]+trim_range]
        cutout = cutout[cutout['y'] >= cut_out_coordinates[1]-trim_range]
        cutout = cutout.values

        cutout_pillars, cutout_coordinates = create_pillars(cutout, origin=global_coords[:2])
        cutout_features , cutout_coordinates = get_feature_tensor(cutout_pillars, cutout_coordinates)
        training_sample = {'sweep': torch.from_numpy(sweep_features).float(),
                           'sweep_coordinates': sweep_coordinates,
                           'cutout': torch.from_numpy(cutout_features).float(),
                           'cutout_coordinates': cutout_coordinates,
                           'label': rand_trans}

        return training_sample
