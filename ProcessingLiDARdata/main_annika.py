import numpy as np
from lidar_data_functions import *
from PIL import Image
import matplotlib.pyplot as plt
import csv


########################################################################################################
# A FIRST SKETCH ON HOW TO CREATE FAKE TRAINING SAMPLES (of one single sweep) WITHOUT A NEED FOR A MAP #
########################################################################################################


input_folder_name = input('Type name of new folder:')

folder_name = '/data_' + input_folder_name
current_path = os.getcwd()
folder_path = current_path + folder_name
path_sweeps = folder_path + '/sweeps'
path_cutouts = folder_path + '/cutouts'

try:
   os.mkdir(folder_path)
   os.mkdir(path_sweeps)
   os.mkdir(path_cutouts)
except OSError:
   print('Failed to create new directory.')
else:
   print('Successfully created new directory with subdirectories. ')


path_to_csv = '/home/master04/Documents/master_thesis/ProcessingLiDARdata/_out_Town03_190207_18/Town03_190207_18.csv'
path_to_pc = '/home/master04/Documents/master_thesis/ProcessingLiDARdata/_out_Town03_190207_18/pc/'

# create a list of all ply-files in a directory
ply_files = os.listdir(path_to_pc)

# create csv-file with header: frame_number, x, y, angle (i.e. the labels)
csv_path = folder_path + '/labels.csv'
with open(csv_path, mode='w') as csv_file:
    fieldnames = ['frame_number', 'x', 'y', 'angle']
    csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(fieldnames)

i=0
for file_name in ply_files:
    i = i+1
    print('File ', i, ' of ', len(ply_files))

    # Load data:
    path_to_ply = path_to_pc + file_name
    pc, global_lidar_coordinates, frame_number = load_data(path_to_ply, path_to_csv)

    # create the sweep, transform a bit to create training sample
    print('creating sweep...')
    rand_trans = random_rigid_transformation(1, 10)
    sweep = training_sample_rotation_translation(pc, rand_trans)
    sweep = trim_pointcloud(sweep)
    sweep_image = discretize_pointcloud(sweep)
    path = path_sweeps + '/' + str(frame_number)
    np.save(path, sweep_image)

    # fake a map cutout
    print('creating cutout...')
    cutout = trim_pointcloud(pc, range=1.5*15)
    cutout_image = discretize_pointcloud(cutout, array_size=600*1.5, trim_range=1.5*15)
    path = path_cutouts + '/' + str(frame_number)
    np.save(path, cutout_image)

    # write frame_number in column 1, and the transformation in the next columns
    with open(csv_path , mode = 'a') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',' , quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow([frame_number, rand_trans[0], rand_trans[1], rand_trans[2]])


'''s = np.load('/home/master04/Documents/master_thesis/ProcessingLiDARdata/data_test/sweeps/173506.npy')
c = np.load('/home/master04/Documents/master_thesis/ProcessingLiDARdata/data_test/cutouts/173506.npy')
array_to_png(s)
array_to_png(c)'''
