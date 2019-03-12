from lidar_data_functions import *


path_to_ply_folder_list = ['/home/master04/Desktop/Ply_files/_out_Town02_190306_1/pc/',
                           '/home/master04/Desktop/Ply_files/_out_Town03_190306_1/pc/'] # Fill in these ''

path_to_csv_list = ['/home/master04/Desktop/Ply_files/_out_Town02_190306_1/Town02_190306_1.csv',
                    '/home/master04/Desktop/Ply_files/_out_Town03_190306_1/Town03_190306_1.csv'] # fill in the paths

k = 0

for folder in path_to_ply_folder_list:
    print('starting to create map from folder:', folder)

    files_in_ply_folder = os.listdir(folder)
    number_of_files_to_load = len(files_in_ply_folder)

    path_to_csv = path_to_csv_list[k]
    print('csv-file', path_to_csv)
    k += 1


    pc_super_array = np.zeros((1, 3))

    max_x_val = float("-inf")
    max_y_val = float("-inf")
    min_x_val = float("inf")
    min_y_val = float("inf")
    i = 0


    for file in files_in_ply_folder: #[0::150]:  # the last number is how large steps to take
        try:
            # Create the path to the ply file
            path_to_ply = folder + file
            point_cloud, global_coordinates = load_data(path_to_ply, path_to_csv)
            i = i + 1
        except:
            print('Failed to load file ', file, '. Moving on to next file.')
            continue

        # rotate, translate the point cloud to global coordinates and trim the point cloud
        trimmed_pc = trim_pointcloud(point_cloud, range=30, roof=100, floor=0)

        rotated_pc = rotate_pointcloud_to_global(trimmed_pc, global_coordinates)

        rotated_and_translated_pc = translate_pointcloud_to_global(rotated_pc, global_coordinates)

        # concatenate the rotated and translated array into a super array
        pc_super_array = np.concatenate((pc_super_array, rotated_and_translated_pc))

        # find max and min value of the x and y coordinates
        # If using the super array this can be removed and the max and min values can be found from the super_array outside
        # the for loop.
        max_x_val_tmp = np.max(rotated_and_translated_pc[:, 0])
        min_x_val_tmp = np.min(rotated_and_translated_pc[:, 0])

        max_y_val_tmp = np.max(rotated_and_translated_pc[:, 1])
        min_y_val_tmp = np.min(rotated_and_translated_pc[:, 1])

        if max_x_val_tmp > max_x_val:
            max_x_val = max_x_val_tmp

        if min_x_val_tmp < min_x_val:
            min_x_val = min_x_val_tmp

        if max_y_val_tmp > max_y_val:
            max_y_val = max_y_val_tmp

        if min_y_val_tmp < min_y_val:
            min_y_val = min_y_val_tmp


    print('Done loading files. Creating map.')
    # delete the zeros in the first row in the super array that was used at the initalization of the array.
    pc_super_array = np.delete(pc_super_array, 0, axis=0)

    # save the max and min values in an array. This is used to decide the size of the map
    min_max = np.array((min_x_val, max_x_val, min_y_val, max_y_val))

    # Discretize the point cloud
    discretized_pc = discretize_pointcloud_map(pc_super_array, min_max, spatial_resolution=0.05) ########

    '''
    Create a png-image of a discretized pointcloud. Create one image per layer. This is mostly for visualizing purposes.
    Also saves the map matrix and the max min values.
    '''

    # create a folder name
    folder_name = 'map_Town' + str(k+1) + '_night_run'

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

    discretized_pointcloud_BEV = discretized_pc  # Save map in new variable to be scaled
    # NORMALIZE THE BEV IMAGE
    for channel in range(np.shape(discretized_pointcloud_BEV)[0]):
        max_value = np.max(discretized_pointcloud_BEV[channel, :, :])
        print('Max max_value inarray_to_png: ', max_value)

        # avoid division with 0
        if max_value == 0:
            max_value = 1

        scale = 255/max_value
        discretized_pointcloud_BEV[channel, :, :] = discretized_pointcloud_BEV[channel, :, :] * scale
        print('Largest pixel value (should be 255) : ', np.max(discretized_pointcloud_BEV[channel, :, :]))
        # create the png_path
        png_path = folder_path_png + 'channel_' + str(channel)+'.png'

    # Save images
        img = Image.fromarray(discretized_pointcloud_BEV[channel, :, :])
        new_img = img.convert("L")
        new_img.rotate(180).save(png_path)

    # Save the map array and the max and min values of the map in the same folder as the BEV image
    np.save(os.path.join(folder_path, 'map.npy'), discretized_pc)
    np.save(os.path.join(folder_path, 'max_min.npy'), discretized_pc)

