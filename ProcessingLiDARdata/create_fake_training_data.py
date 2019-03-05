from lidar_data_functions import *
import csv
import random


########################################################################################################
# A FIRST SKETCH ON HOW TO CREATE FAKE TRAINING SAMPLES (of one single sweep) WITHOUT A NEED FOR A MAP #
########################################################################################################

input_folder_name = input('Type name of new folder to save data:')
print(' ')

current_path = os.getcwd()
folder_path = current_path + '/fake_training_data_' + input_folder_name
path_samples = folder_path + '/samples'

try:
   os.mkdir(folder_path)
   os.mkdir(path_samples)
except OSError:
    print('Failed to create new directory.')
else:
    print('Successfully created new directory with subdirectory, at ', folder_path)

################################# CHANGE HERE ###########################################################
print(' ')
path_to_lidar_data = input('Type complete path to the folder that contains "pc"-folder with LiDAR data and a csv file, e.g. Town02_190222_1 :')
dir_list = os.listdir(path_to_lidar_data) # this should return a list where only one object is our csv_file
path_to_pc = os.path.join(path_to_lidar_data, 'pc/')  # we assume we follow the structure of creating lidar data folder with a pc folder for ply
for file in dir_list:
    if '.csv' in file:  # find csv-file
        path_to_csv = os.path.join(path_to_lidar_data, file)


translation = float(input('Translation in meters:'))
rotation = float(input('Rotation in degrees:'))
number_of_files_to_load = int(input('How many training samples do you want to create:'))
print(' ')
########################################################################################################
# create a list of all ply-files in a directory
ply_files = os.listdir(path_to_pc)
# create csv-file with header: frame_number, x, y, angle (i.e. the labels)
csv_labels_path = folder_path + '/labels.csv'
with open(csv_labels_path, mode='w') as csv_file:
    fieldnames = ['frame_number', 'x', 'y', 'angle']
    csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(fieldnames)

i = 0
# shuffle order of ply-files
random.shuffle(ply_files)

if number_of_files_to_load > len(ply_files):
    number_additional_files = number_of_files_to_load-len(ply_files)
    additional_files = np.random.choice(ply_files, number_additional_files)
    ply_files = np.concatenate((ply_files, additional_files))

for file_name in ply_files[:number_of_files_to_load]:
    
    # Load data:
    try:
        path_to_ply = path_to_pc + file_name
        pc, global_lidar_coordinates = load_data(path_to_ply, path_to_csv)
        i = i + 1
        print('Creating training sample ', i, ' of ', number_of_files_to_load)
    except:
        print('Failed to load file ', file_name, '. Moving on to next file.')
        continue

    # create the sweep, transform a bit to create training sample
    rand_trans = random_rigid_transformation(translation, rotation)

    sweep = training_sample_rotation_translation(pc, rand_trans)
    sweep = trim_pointcloud(sweep)
    # discretize and pad sweep
    sweep_image = discretize_pointcloud(sweep, array_size=600, trim_range=15, spatial_resolution=0.05, padding=True, pad_size=150)

    # fake a map cutout
    cutout = trim_pointcloud(pc, range=1.5*15)
    cutout_image = discretize_pointcloud(cutout, array_size=600*1.5, trim_range=1.5*15, padding=False)

    # concatenate the sweep and the cutout image into one image and save.
    sweep_and_cutout_image = np.concatenate((sweep_image, cutout_image))
    sweep_and_cutout_image = normalize_sample(sweep_and_cutout_image)
    path = path_samples + '/' + str(i)
    np.save(path, sweep_and_cutout_image)

    # write frame_number in column 1, and the transformation in the next columns
    with open(csv_labels_path , mode ='a') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csv_writer.writerow([i, rand_trans[0], rand_trans[1], rand_trans[2]])

