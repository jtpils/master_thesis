import torch
from torch.utils.data import Dataset
from functions_for_smaller_data import *
from tqdm import tqdm
import time
import matplotlib.pyplot as plt


class DataSetFakeData(Dataset):
    """ Fake data, sweep+sweep """

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
        #self.translation = translation
        #self.rotation = rotation
        self.labels = [random_rigid_transformation(translation, rotation) for x in np.arange(self.length)]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        file_name = self.sweeps_file_names[idx]
        pc, global_coords = load_data(os.path.join(self.sample_dir,file_name), self.csv_path)
        #rand_trans = random_rigid_transformation(self.translation, self.rotation)
        rand_trans = self.labels[idx]

        # sweep
        sweep = training_sample_rotation(pc, rand_trans[-1])
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



class DataSetMapData(Dataset):
    """Generate "real" data on the go, eg sweep+map-cut-out, using the map.npy file"""

    def __init__(self, sample_path, csv_path, map_path, minmax_path, translation, rotation):
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
        self.length = len(self.sweeps_file_names)
        self.csv_path = csv_path
        #self.translation = translation
        #self.rotation = rotation
        self.map = np.load(map_path)
        self.map_minmax = np.load(minmax_path)
        self.labels = [random_rigid_transformation(translation, rotation) for x in np.arange(self.length)]
        print('Done initializing data set.')

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Load one sweep and generate a random transformation
        file_name = self.sweeps_file_names[idx]
        pc, global_coords = load_data(os.path.join(self.sample_dir,file_name), self.csv_path)
        #rand_trans = random_rigid_transformation(self.translation, self.rotation)
        rand_trans = self.labels[idx]

        # sweep
        sweep = rotate_pointcloud_to_global(pc, global_coords)  # rotate to align with map
        sweep = training_sample_rotation(sweep, rand_trans[2])  # rotate a bit more to create training sample
        sweep = trim_pointcloud(sweep)
        sweep_image = discretize_pc_fast(sweep)

        # map cut-out
        cut_out_coordinates = global_coords[0][:2] + rand_trans[:2]  # translation x, y
        spatial_resolution = 0.1
        x_min, x_max, y_min, y_max = self.map_minmax
        x_grid = np.floor((cut_out_coordinates[0]-x_min)/spatial_resolution).astype(int)
        y_grid = np.floor((cut_out_coordinates[1]-y_min)/spatial_resolution).astype(int)
        cutout_image = self.map[:, x_grid-150:x_grid+150, y_grid-150:y_grid+150]

        # concatenate the sweep and the cutout image into one image and save.
        sweep_and_cutout_image = np.concatenate((sweep_image, cutout_image))
        sweep_and_cutout_image = normalize_sample(sweep_and_cutout_image)

        training_sample = {'sample': torch.from_numpy(sweep_and_cutout_image).float(), 'labels': rand_trans}
        return training_sample


'''
class DataSetCreateMapData(Dataset):
    """Generate "real" data on the go, eg sweep+map-cut-out, using the map.npy file"""

    def __init__(self, sample_path, csv_path, grid_csv_path, translation, rotation):
        """
        Args:
            sample_path (string): Directory with all the sweeps.
            csv_path to csv with global coordinates
            grid_csv_path: path to all grid csv files
            translation, rotation; the amount of rigid transformation
        """
        self.sample_dir = sample_path
        self.sweeps_file_names = os.listdir(sample_path)
        self.length = len(self.sweeps_file_names)
        self.csv_path = csv_path
        #self.translation = translation
        #self.rotation = rotation
        self.labels = [random_rigid_transformation(translation, rotation) for x in np.arange(self.length)]

        list_of_files = os.listdir(grid_csv_path)
        sweeps = []
        print('loading all LiDAR detections...')
        for file in tqdm(list_of_files):
            if 'grid' in file:
                pc = pd.read_csv(os.path.join(grid_csv_path, file))
                sweeps.append(pc)
        self.lidar_points = pd.concat(sweeps)
        print('Done loading detections.')
        del sweeps, pc

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Load one sweep and generate a random transformation
        file_name = self.sweeps_file_names[idx]
        pc, global_coords = load_data(os.path.join(self.sample_dir,file_name), self.csv_path)
        #rand_trans = random_rigid_transformation(self.translation, self.rotation)
        rand_trans = self.labels[idx]

        # sweep
        t1 = time.time()
        sweep = rotate_pointcloud_to_global(pc, global_coords)  # rotate to align with map
        sweep = training_sample_rotation(sweep, rand_trans[2])  # rotate a bit more to create training sample
        sweep = trim_pointcloud(sweep)
        t2 = time.time()
        sweep_image = discretize_pc_fast(sweep)
        t3 = time.time()
        print('sweep ', t3-t1)
        print('discretize sweep ', t3-t2)

        # map cut-out
        cut_out_coordinates = global_coords[0][:2] + rand_trans[:2]  # translation x, y

        # we want all coordinates that in trim_range around cut_out_coordinates
        trim_range = 15

        # get all points around the sweep
        t1 = time.time()
        cutout = self.lidar_points[self.lidar_points['x'] <= cut_out_coordinates[0]+trim_range]
        cutout = cutout[cutout['x'] >= cut_out_coordinates[0]-trim_range]
        cutout = cutout[cutout['y'] <= cut_out_coordinates[1]+trim_range]
        cutout = cutout[cutout['y'] >= cut_out_coordinates[1]-trim_range]
        # if we want to use occupancy grid, sample points first
        # move all points such that the cut-out-coordinates becomes the origin
        cutout = cutout.values - np.array((cut_out_coordinates[0], cut_out_coordinates[1], 0))
        t2 = time.time()
        cutout_image = discretize_pc_fast(cutout)
        t3 = time.time()
        print('cutout ', t3-t1)
        print('discretize cutout ', t3-t2)

        # concatenate the sweep and the cutout image into one image and save.
        sweep_and_cutout_image = np.concatenate((sweep_image, cutout_image))
        sweep_and_cutout_image = normalize_sample(sweep_and_cutout_image)

        training_sample = {'sample': torch.from_numpy(sweep_and_cutout_image).float(), 'labels': rand_trans}
        return training_sample
'''
