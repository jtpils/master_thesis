import numpy as np
from lidar_data_functions import *
from PIL import Image
import matplotlib.pyplot as plt
import csv


########################################################################################################
# A FIRST SKETCH ON HOW TO CREATE FAKE TRAINING SAMPLES (of one single sweep) WITHOUT A NEED FOR A MAP #
########################################################################################################

input_folder_name = input('Type name of new folder:')

current_path = os.getcwd()
# print('current path', current_path)
folder_path = current_path + '/fake_training_data_' + input_folder_name
# path_sweeps = folder_path + '/sweeps'
# path_cutouts = folder_path + '/cutouts'
path_samples = folder_path + '/samples'

try:
   os.mkdir(folder_path)
   # os.mkdir(path_sweeps)
   # os.mkdir(path_cutouts)
   os.mkdir(path_samples)
except OSError:
    print('Failed to create new directory.')
else:
    print('Successfully created new directory with subdirectory, at ', folder_path)

# Sabina's computer
# path_to_csv = '/Users/sabinalinderoth/Documents/master_thesis/ProcessingLiDARdata/_out_Town03_190207_18/Town03_190207_18.csv'
# path_to_pc = '/Users/sabinalinderoth/Documents/master_thesis/ProcessingLiDARdata/_out_Town03_190207_18/pc/'


path_to_csv = '/Users/annikal/Desktop/drive-download-20190220T155133Z-001/_out_framenumber/framenumber.csv'
path_to_pc = '/Users/annikal/Desktop/drive-download-20190220T155133Z-001/_out_framenumber/pc/'

# create a list of all ply-files in a directory
ply_files = os.listdir(path_to_pc)

# create csv-file with header: frame_number, x, y, angle (i.e. the labels)
csv_labels_path = folder_path + '/labels.csv'
with open(csv_labels_path, mode='w') as csv_file:
    fieldnames = ['frame_number', 'x', 'y', 'angle']
    csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(fieldnames)

i = 0
for file_name in ply_files[10:-1]:
    # Load data:

    '''try:
        path_to_ply = path_to_pc + file_name
        pc, global_lidar_coordinates = load_data(path_to_ply, path_to_csv)
        i = i + 1
        print('Creating training sample ', i, ' of ', int(len(ply_files)/10))
    except:
        print('Failed to load file ', file_name, '. Moving on to next file.')
        continue'''
    path_to_ply = path_to_pc + file_name
    pc, global_lidar_coordinates = load_data(path_to_ply, path_to_csv)
    i = i + 1
    print('Creating training sample ', i, ' of ', int(len(ply_files) / 10))

    # create the sweep, transform a bit to create training sample
    #print('creating sweep...')
    rand_trans = random_rigid_transformation(1, 0) ##### ONLY TRANSLATION

    sweep = training_sample_rotation_translation(pc, rand_trans)
    sweep = trim_pointcloud(sweep)
    # discretize and pad sweep
    sweep_image = discretize_pointcloud(sweep, array_size=600, trim_range=15, spatial_resolution=0.05, padding=True, pad_size=150)

    # path = path_sweeps + '/' + str(i)
    # np.save(path, sweep_image)

    # fake a map cutout
    #print('creating cutout...')
    cutout = trim_pointcloud(pc, range=1.5*15)
    cutout_image = discretize_pointcloud(cutout, array_size=600*1.5, trim_range=1.5*15)
    # path = path_cutouts + '/' + str(i)
    # np.save(path, cutout_image)

    # concatenate the sweep and the cutout image into one image and save.
    sweep_and_cutout_image = np.concatenate((sweep_image, cutout_image))
    path = path_samples + '/' + str(i)
    np.save(path, sweep_and_cutout_image)
    #print('sweep and cutout image', np.shape(sweep_and_cutout_image))

    # write frame_number in column 1, and the transformation in the next columns
    with open(csv_labels_path , mode ='a') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',' , quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow([i, rand_trans[0], rand_trans[1], rand_trans[2]])
