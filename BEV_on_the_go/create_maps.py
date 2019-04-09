from tqdm import tqdm
import time
import numpy as np
from functions import *
import os


t1 = time.time()
############################ Change stuff here ###########################################################
path_pc = '/home/master04/Desktop/Ply_files/validation_and_test/test_set/pc/'  # path to pc folder with ply-files
path_csv = '/home/master04/Desktop/Ply_files/validation_and_test/test_set/test_set.csv'  # path to csv with global coordinates
folder_name = 'map_Town_test' # name of new directory to save stuff
#########################################################################################################

files_in_ply_folder = os.listdir(path_pc)

# creates folder to store the png files
current_path = os.getcwd()
folder_path = os.path.join(current_path,folder_name)
folder_path_png = folder_path + '/map_png/'
try:
    os.mkdir(folder_path)
    os.mkdir(folder_path_png)
except OSError:
    print('Failed to create new directory.')
else:
    print('Successfully created new directory with path: ', folder_path, 'and', folder_path_png)


print('Loading files...')
pc_super_array = np.zeros((1, 3))
for file in tqdm(files_in_ply_folder):  # the last number is how large steps to take
    try:
        # Create the path to the ply file
        path_to_ply = os.path.join(path_pc, file)
        pc, global_coordinates = load_data(path_to_ply, path_csv)
    except:
        print('Failed to load file ', file, '. Moving on to next file.')
        continue

    pc = trim_point_cloud_range(pc, origin=global_coordinates[:2], trim_range=30)
    pc = trim_point_cloud_vehicle_ground(pc, origin=global_coordinates[:2], remove_vehicle=True, remove_ground=False)
    #pc = rotate_point_cloud(pc, global_coordinates[-1], to_global=True)  # rotate to global
    #pc = translate_point_cloud(pc, global_coordinates[0:2])  # translate to global

    pc_super_array = np.concatenate((pc_super_array, pc))

print('Done loading files. Creating map.')
pc_super_array = np.delete(pc_super_array, 0, axis=0)
min_max = [np.min(pc_super_array[:,0]), np.max(pc_super_array[:,0]), np.min(pc_super_array[:,1]), np.max(pc_super_array[:,1])]

discretized_map = discretize_map(pc_super_array, spatial_resolution=0.1)

# Save the map array and the max and min values of the map in the same folder as the BEV image
np.save(os.path.join(folder_path, 'map.npy'), discretized_map)
np.save(os.path.join(folder_path, 'max_min.npy'), min_max)

# NORMALIZE THE BEV IMAGE
max_value = np.max(discretized_map[0, :, :])
print('Max max_value inarray_to_png: ', max_value)

# avoid division with 0
if max_value == 0:
    max_value = 1

scale = 255/max_value
discretized_map[0, :, :] = discretized_map[0, :, :] * scale
print('Largest pixel value (should be 255) : ', np.max(discretized_map[0, :, :]))
png_path = folder_path_png + 'map.png'

# Save images
img = Image.fromarray(discretized_map[0, :, :])
new_img = img.convert("L")
new_img.rotate(180).save(png_path)
t2 = time.time()
print('time: ', t2-t1)




