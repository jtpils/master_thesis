import numpy as np
from lidar_data_functions import *
from PIL import Image
import matplotlib.pyplot as plt


########################################################################################################
# A FIRST SKETCH ON HOW TO CREATE FAKE TRAINING SAMPLES (of one single sweep) WITHOUT A NEED FOR A MAP #
########################################################################################################

#path_to_ply = '/Users/annikal/Documents/master_thesis/ProcessingLiDARdata/_out_Town03_190207_18/pc/173504.ply'
path_to_csv = '/Users/annikal/Documents/master_thesis/ProcessingLiDARdata/_out_Town03_190207_18/Town03_190207_18.csv'
path_to_pc = '/Users/annikal/Documents/master_thesis/ProcessingLiDARdata/_out_Town03_190207_18/pc/'

# create a list of all ply-files in a directory
# ply_files = ...

# create csv-file with header: frame_number, x, y, angle (i.e. the labels)
# labels_csv = ...

for file_name in ply_files:

    # Load data:
    path_to_ply = path_to_pc + file_name
    pc, global_lidar_coordinates, frame_number = load_data(path_to_ply, path_to_csv)

    # create the sweep
    sweep = rotate_pointcloud_to_global(pc, global_lidar_coordinates)
    sweep = translate_pointcloud_to_global(sweep, global_lidar_coordinates)
    # now, transform a bit to create training sample
    rand_trans = random_rigid_transformation(1, 10)
    sweep = training_sample_rotation_translation(sweep, rand_trans)
    # now trim it
    sweep = trim_pointcloud(sweep)

    # fake a map cutout
    cutout = rotate_pointcloud_to_global(pc, global_lidar_coordinates)
    cutout = translate_pointcloud_to_global(cutout, global_lidar_coordinates)
    cutout = trim_pointcloud(cutout, range=1.5*15)

    # discretize bot the sweep and the map
    sweep_image = discretize_pointcloud(sweep)
    cutout_image = discretize_pointcloud(cutout)

    # create a directory 'sweep' and a directory 'cutout' first and add it in the paths
    # os.mkdir() .....

    # save as numpy arrays
    path_sweep = 'sweep/' + str(frame_number)
    np.save(path_sweep, sweep)
    path_cutout= 'cutout/' + str(frame_number)
    np.save(path_cutout, cutout)

    # write to next row in csv
    # write frame_number in column 1, and the transformation in the next columns

