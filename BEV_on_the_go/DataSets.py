import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import os
from functions import *


class DataSetFakeData(Dataset):
    """ Fake data, sweep+sweep """

    def __init__(self, path_pc, csv_path, translation=1, rotation=0):

        self.path_pc = path_pc
        self.ply_files = os.listdir(self.path_pc)
        self.path_csv = csv_path
        self.translation = translation
        self.rotation = rotation

    def __len__(self):
        return len(self.ply_files)

    def __getitem__(self, idx):
        # get point cloud
        ply_path = os.path.join(self.path_pc, self.ply_files[idx])
        pc, global_coords = load_data(ply_path, self.path_csv)

        # create fake cutout with no translation or rotation
        cutout = trim_point_cloud_range(pc, trim_range=15)
        cutout = trim_point_cloud_vehicle_ground(cutout, remove_vehicle=True, remove_ground=False)
        cutout_image = discretize_point_cloud(cutout, trim_range=15, spatial_resolution=0.1, image_size=300)

        # rotate and translate sweep
        rand_trans = random_rigid_transformation(self.translation, self.rotation)
        sweep = trim_point_cloud_range(pc, trim_range=20)
        sweep = trim_point_cloud_vehicle_ground(sweep, remove_vehicle=True, remove_ground=False)
        sweep = rotate_point_cloud(sweep, rand_trans[-1], to_global=False)
        sweep = translate_point_cloud(sweep, rand_trans[:2])
        sweep = trim_point_cloud_range(sweep, trim_range=15)
        sweep_image = discretize_point_cloud(sweep, trim_range=15, spatial_resolution=0.1, image_size=300)

        # create and normalize sample
        sample = np.concatenate((sweep_image, cutout_image))
        sample = normalize_sample(sample)
        training_sample = {'sample': torch.from_numpy(sample).float(),
                           'label': rand_trans}

        return training_sample


class DataSetMapData(Dataset):
    """Generate "real" data on the go, eg sweep+map-cut-out, using the map.npy file"""

    def __init__(self, sample_path, csv_path, map_path, minmax_path, translation=1, rotation=0):
        """
        Args:
            sample_path (string): Directory with all the sweeps.
            csv_path to csv with global coordinates
            map_path: path to map.npy
            minmax_path: path to min max values in the map
            translation, rotation; the amount of rigid transformation
        """
        self.sample_dir = sample_path
        self.sweeps_file_names = os.listdir(sample_path)
        self.csv_path = csv_path
        self.map = np.load(map_path)
        self.map_minmax = np.load(minmax_path)
        self.translation = translation
        self.rotation = rotation
        print('Done initializing data set.')

    def __len__(self):
        return len(self.sweeps_file_names)

    def __getitem__(self, idx):
        # Load one sweep and generate a random transformation
        file_name = self.sweeps_file_names[idx]
        pc, global_coords = load_data(os.path.join(self.sample_dir,file_name), self.csv_path)
        rand_trans = random_rigid_transformation(self.translation, self.rotation)

        # sweep
        # rotate and translate sweep
        rand_trans = random_rigid_transformation(self.translation, self.rotation)
        sweep = trim_point_cloud_range(pc, origin=global_coords[:2], trim_range=20)
        sweep = trim_point_cloud_vehicle_ground(sweep,  origin=global_coords[:2], remove_vehicle=True, remove_ground=False)
        sweep = rotate_point_cloud(sweep, rand_trans[-1], to_global=False)
        sweep = translate_point_cloud(sweep, rand_trans[:2])
        sweep = trim_point_cloud_range(sweep, origin=global_coords[:2],  trim_range=15)
        #move our origin
        origin = global_coords[:2] + rand_trans[:2]
        sweep_image = discretize_point_cloud(sweep, origin=origin,trim_range=15, spatial_resolution=0.1, image_size=300)

        # map cut-out
        cut_out_coordinates = global_coords[:2]
        spatial_resolution = 0.1
        x_min, x_max, y_min, y_max = self.map_minmax
        x_grid = np.floor((cut_out_coordinates[0]-x_min)/spatial_resolution).astype(int)
        y_grid = np.floor((cut_out_coordinates[1]-y_min)/spatial_resolution).astype(int)
        cutout_image = self.map[:, x_grid-150:x_grid+150, y_grid-150:y_grid+150] ###### change x and y?

        # concatenate the sweep and the cutout image into one image and save.
        sweep_and_cutout_image = np.concatenate((sweep_image, cutout_image))
        sweep_and_cutout_image = normalize_sample(sweep_and_cutout_image)

        training_sample = {'sample': torch.from_numpy(sweep_and_cutout_image).float(), 'label': rand_trans}
        return training_sample


class DataSetMapData_kSweeps(Dataset):
    """Generate "real" data on the go, eg sweep+map-cut-out, using the map.npy file. The sweep is made up by multiple ply-files."""

    def __init__(self, sample_path, csv_path, map_path, minmax_path, translation=1, rotation=1):
        """
        Args:
            sample_path (string): Directory with all the sweeps.
            csv_path to csv with global coordinates
            map_path: path to map.npy
            minmax_path: path to min max values in the map
            translation, rotation; the amount of rigid transformation
        """
        self.sample_dir = sample_path
        self.sweeps_file_names = sorted(os.listdir(sample_path))
        self.num_sweeps = 5
        self.csv_path = csv_path
        self.map = np.load(map_path)
        self.map_minmax = np.load(minmax_path)
        self.translation = translation
        self.rotation = rotation
        print('Done initializing data set.')

    def __len__(self):
        return len(self.sweeps_file_names) - (self.num_sweeps - 1)

    def __getitem__(self, idx):
        # Load one sweep and generate a random transformation

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

        # sweep
        # rotate and translate sweep
        rand_trans = random_rigid_transformation(self.translation, self.rotation)
        sweep = trim_point_cloud_range(pc_multiple_sweeps, origin=global_coords[:2], trim_range=20)
        del pc_multiple_sweeps, pc
        #sweep = trim_point_cloud_vehicle_ground(sweep,  origin=global_coords[:2], remove_vehicle=True, remove_ground=False)
        sweep = rotate_point_cloud(sweep, rand_trans[-1], to_global=False)
        sweep = translate_point_cloud(sweep, rand_trans[:2])
        sweep = trim_point_cloud_range(sweep, origin=global_coords[:2],  trim_range=15)
        #move our origin
        origin = global_coords[:2] + rand_trans[:2]
        sweep_image = discretize_point_cloud(sweep, origin=origin,trim_range=15, spatial_resolution=0.1, image_size=300)

        # map cut-out
        cut_out_coordinates = global_coords[:2]
        spatial_resolution = 0.1
        x_min, x_max, y_min, y_max = self.map_minmax
        x_grid = np.floor((cut_out_coordinates[0]-x_min)/spatial_resolution).astype(int)
        y_grid = np.floor((cut_out_coordinates[1]-y_min)/spatial_resolution).astype(int)
        cutout_image = self.map[:, x_grid-150:x_grid+150, y_grid-150:y_grid+150]

        # concatenate the sweep and the cutout image into one image and save.
        sweep_and_cutout_image = np.concatenate((sweep_image, cutout_image))
        sweep_and_cutout_image = normalize_sample(sweep_and_cutout_image)

        training_sample = {'sample': torch.from_numpy(sweep_and_cutout_image).float(), 'label': rand_trans}
        del sweep_image, sweep, cutout_image, sweep_and_cutout_image
        return training_sample


class DataSetMapData_createMapOnTheGo(Dataset):
    """Lidar sample dataset."""

    def __init__(self, sample_dir, csv_path, grid_csv_path, translation=1, rotation=0, occupancy_grid=False):
        """
        Args:
            sample_dir <string>: Directory with all ply-files.
        """
        print('translation: ', translation)
        print('rotation: ', rotation)
        self.sample_dir = sample_dir
        self.csv_path = csv_path
        self.translation = translation
        self.rotation = rotation
        self.sweeps_file_names = sorted(os.listdir(sample_dir))
        self.num_sweeps = 5
        self.grid_edges = np.load(os.path.join(grid_csv_path, 'edges.npy'))
        self.occupancy_grid = occupancy_grid

        list_of_csv = os.listdir(grid_csv_path)

        # NEW IDEA
        grid_dict = dict()
        print('loading all LiDAR detections...')
        for file in tqdm(list_of_csv):
            if 'grid' in file:
                pc = pd.read_csv(os.path.join(grid_csv_path, file))
                grid_dict[file] = pc
                #sweeps.append(pc)
        #self.lidar_points = pd.concat(sweeps)
        self.grid_dict = grid_dict
        print('Done loading detections.')
        del grid_dict, pc
        '''
        # OLD IDEA
        print('loading all LiDAR detections...')
        sweeps = list()
        for file in tqdm(list_of_csv):
            if 'grid' in file:
                pc = pd.read_csv(os.path.join(grid_csv_path, file))
                sweeps.append(pc)
        self.lidar_points = pd.concat(sweeps)
        print('Done loading detections.')
        del sweeps, pc
        '''

    def __len__(self):
        return len(self.sweeps_file_names) - (self.num_sweeps - 1)

    def __getitem__(self, idx):
        # load ply-file
        t1 = time.time()
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
        t3 = time.time()
        #print('==================================time to load sweep: ', t3-t1)
        # rotate and translate sweep
        rand_trans = random_rigid_transformation(self.translation, self.rotation)
        sweep = trim_point_cloud_range(pc_multiple_sweeps, origin=global_coords[:2], trim_range=45)
        sweep = rotate_point_cloud(sweep, global_coords[:2], rand_trans[-1], to_global=False)
        sweep = translate_point_cloud(sweep, rand_trans[:2])
        sweep = trim_point_cloud_range(sweep, origin=global_coords[:2],  trim_range=15)
        origin = global_coords[:2]
        sweep_image = discretize_point_cloud(sweep, origin=origin, trim_range=15, spatial_resolution=0.1, image_size=300)
        if self.occupancy_grid:
            sweep_image[sweep_image > 0] = 1

        t2 = time.time()
        #print('Time to create sweep image after the loading is done: ', t2-t3)
        #print(' ')

        # map cut-out
        trim_range = 15
        cut_out_coordinates = global_coords[:2]

        # NEW IDEA
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

        '''
        # OLD IDEA
        # we want all coordinates that in trim_range around cut_out_coordinates
        trim_range = 15
        # get all points around the sweep
        cutout = self.lidar_points[self.lidar_points['x'] <= cut_out_coordinates[0]+trim_range]  # OLD IDEA'''

        # try using numpy arrays directly instead, might be faster?
        cutout = cutout[cutout['x'] <= cut_out_coordinates[0]+trim_range] # NEW IDEA
        cutout = cutout[cutout['x'] >= cut_out_coordinates[0]-trim_range]
        cutout = cutout[cutout['y'] <= cut_out_coordinates[1]+trim_range]
        cutout = cutout[cutout['y'] >= cut_out_coordinates[1]-trim_range]

        cutout = cutout.values
        num_points_to_keep = len(sweep)*2
        points_to_keep = np.random.choice(len(cutout), num_points_to_keep)
        cutout = cutout[points_to_keep,:]

        # move all points such that the cut-out-coordinates become the origin
        cutout_image = discretize_point_cloud(cutout, origin=cut_out_coordinates[:2], trim_range=15, spatial_resolution=0.1, image_size=300)
        if self.occupancy_grid:
            cutout_image[cutout_image > 0] = 1

        t3 = time.time()
        #print('Time to create cutout image: ', t3-t2)  # up to 2 seconds
        # concatenate the sweep and the cutout image into one image and save.
        sweep_and_cutout_image = np.concatenate((sweep_image, cutout_image))
        sweep_and_cutout_image = normalize_sample(sweep_and_cutout_image)
        training_sample = {'sample': torch.from_numpy(sweep_and_cutout_image).float(), 'label': rand_trans}

        del pc_multiple_sweeps, pc, cutout, sweep, sweep_image, cutout_image, sweep_and_cutout_image

        return training_sample


def get_loaders(path_training, path_training_csv, path_validation, path_validation_csv, batch_size, use_cuda, translation, rotation):
    kwargs = {'pin_memory': True, 'num_workers': 16} if use_cuda else {'num_workers': 0}

    if use_cuda:
        path_training_grids = '/home/annika_lundqvist144/csv_grids_190409/csv_grids_training/'
    else:
        path_training_grids = '/home/master04/Desktop/Dataset/ply_grids/csv_grids_190409/csv_grids_training'
    train_set = DataSetMapData_createMapOnTheGo(path_training, path_training_csv, path_training_grids, translation=translation, rotation=rotation)

    n_training_samples = len(train_set)
    print('Number of training samples: ', n_training_samples)
    train_sampler = SubsetRandomSampler(np.arange(n_training_samples, dtype=np.int64))
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=train_sampler, **kwargs)

    if use_cuda:
        path_validation_grids = '/home/annika_lundqvist144/csv_grids_190409/csv_grids_validation/'
    else:
        path_validation_grids = '/home/master04/Desktop/Dataset/ply_grids/csv_grids_190409/csv_grids_validation/'
    val_set = DataSetMapData_createMapOnTheGo(path_validation, path_validation_csv, path_validation_grids, translation=translation, rotation=rotation)

    n_val_samples = len(val_set)
    print('Number of validation samples: ', n_val_samples)
    val_sampler = SubsetRandomSampler(np.arange(n_val_samples, dtype=np.int64))
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, sampler=val_sampler, **kwargs)

    return train_loader, val_loader
