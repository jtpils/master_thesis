import numpy as np
import sys
import pandas as pd


def load_data(path_to_ply, path_to_csv):
    '''
    Load the data from a .ply-file, toghether with the global coordinates for this fram from a csv-file.
    :param
        path_to_ply: string with path to .ply-file
        path_to_csv: string with path to csv-file
    :return:
        pointcloud: nd-array with xyz-coordinates for all detections in the .ply-file. Shape (N,3).
        global_lidar_coordinates: global xyz-coordinates and yaw in degrees for LiDAR position, [x y z yaw]
    '''

    # load pointcloud
    point_cloud = pd.read_csv(path_to_ply, delimiter=' ', skiprows=7, header=None, names=('x','y','z'))
    point_cloud = point_cloud.values
    point_cloud[:, -1] = - point_cloud[:, -1]  # z = -z

    # check if ply file is empty. Stop execution if it is.
    if np.shape(point_cloud)[0] == 0:
        print("ply file with point cloud is empty")
        # sys.exit()

    # extract frame_number from filename
    file_name = path_to_ply.split('/')[-1]  # keep the last part of the path, i.e. the file name
    frame_number = int(file_name[:-4])  # remove the part '.ply' and convert to int

    # load csv-file with global coordinates
    global_coordinates = pd.read_csv(path_to_csv)
    global_coordinates = global_coordinates.values

    # extract information from csv at given frame_number
    row = np.where(global_coordinates == frame_number)[0]  # returns which row the frame number is located on
    global_lidar_coordinates = global_coordinates[row, 1:5]

    # Fix yaw angle:
    if global_lidar_coordinates[0][3] < 0:
        global_lidar_coordinates[0][3] = global_lidar_coordinates[0][3] + 360  # change to interval [0,360]
    global_lidar_coordinates[0][3] = global_lidar_coordinates[0][3] + 90  # add 90

    global_lidar_coordinates[0][1] = -global_lidar_coordinates[0][1]  # y = -y

    return point_cloud, global_lidar_coordinates


def rotate_pointcloud_to_global(pointcloud, global_coordinates):
    '''
    Rotate pointcloud (before trimming it).
    :param pointcloud: input raw point cloud shape (N, 3)
    :param global_coordinates: global coordinates for LiDAR, which contains yaw angle
    :return: rotated_pointcloud: rotated pointcloud, but coordinates are stil relative to LiDAR (not global)
    '''

    yaw = np.deg2rad(global_coordinates[0, 3]) # convert yaw in degrees to radians
    c, s = np.cos(yaw), np.sin(yaw)
    Rz = np.array(((c, -s, 0), (s, c, 0), (0, 0, 1)))  # Rotation matrix
    rotated_pointcloud = Rz @ np.transpose(pointcloud)  # rotate each vector with coordinates, transpose to get dimensions correctly
    rotated_pointcloud = np.transpose(rotated_pointcloud)

    rotated_pointcloud[:, 1] = -rotated_pointcloud[:,1]  # y = -y

    return rotated_pointcloud


def translate_pointcloud_to_global(pointcloud, global_coordinates):
    '''
    Translates all the points in the pointcloud to the global coordinates. This should be done after rotating and then
    trimming the pointcloud.
    :param pointcloud:
    :param global_coordinates:
    :return: global_pointcloud
    '''

    global_pointcloud = pointcloud + global_coordinates[0, :3]  # only add xyz (not yaw)

    return global_pointcloud


def trim_pointcloud(point_cloud, range=15, roof=10, floor=0):  # the hard coded numbers are not absolute.
    '''
    Trim pointcloud to a range given by range_of_interest. Trim detections that are more than (roof) meters above LiDAR,
    and more than (floor) meters below LiDAR. After this, remove z.

    The floor and ground parameter represents the interval of points that should be kept. This means sending in floor=0
    and roof=0 only the detectons from the ground will be kept.

    :param
        pointcloud: ndarray size (N, 3)
        region_of_interest: range
        roof: meter above ground
        floor: meter above ground
    :return:
        2D_pointcloud: nd-array with xy-coordinates, with shape (N, 2)
    '''

    # remove points outside the range of interest
    points_in_range = np.max(np.absolute(point_cloud), axis=1) <= range  # this takes care of both x, y,and z
    point_cloud = point_cloud[points_in_range]

    z_coordinates = point_cloud[:, -1]
    # Remove points that are more then floor and roof meters above ground coordinate
    ground_coordinate = min(z_coordinates)

    floor = ground_coordinate + floor
    roof = ground_coordinate + roof

    coordinate_rows = list(map(lambda x: floor <= x <= roof, z_coordinates))

    point_cloud = point_cloud[coordinate_rows]

    return point_cloud


def random_rigid_transformation(bound_translation_meter, bound_rotation_degrees):
    '''
    This functions return an array with values for translation and rotation from ground truth. The values are drawn from
    a uniform distribution given by the user. This rotation/translation should be used to transform the LiDAR sweep to create a
    training sample. Yields equally probable values around 0, both negative and positive up to the given bound.
    Should be used before discretizing the sweep.
    :param bound_translation_meter: scalar, the largest translation value that is acceptable in meters
    :param bound_rotation_degrees: scalar, the largest rotation value that is acceptable in degrees
    :return: rigid_transformation: an array with 3 elements [x, y, angle_degrees]
    '''

    translation = np.random.uniform(-bound_translation_meter, bound_translation_meter, 2)
    rotation = np.random.uniform(-bound_rotation_degrees, bound_rotation_degrees, 1)

    rigid_transformation = np.concatenate((translation, rotation))

    return rigid_transformation


def training_sample_rotation_translation(pointcloud, rigid_transformation):
    '''
    Rotate and translate a pointcloud according to the random rigid transform. Use this when creating fake training samples.
    :param pointcloud: a lidar sweep that is to be rotated/translated in order to create training sample.

    Do this BEFORE trimming the sweep!

    :param rigid_transformation: get a random transformation with function random_rigid_transformation
    :return: training_pointcloud
    '''
    number_of_points = len(pointcloud)  # number of points in the pointcloud
    translation = np.append(rigid_transformation[:2], 0) # add a zero for z, since we do not want to translate the height
    rotation = np.deg2rad(rigid_transformation[-1])

    # rotate:
    c, s = np.cos(rotation), np.sin(rotation)
    Rz = np.array(((c, -s, 0), (s, c, 0), (0, 0, 1))) # Rotation matrix

    rotated_pointcloud = Rz @ np.transpose(pointcloud) # rotate each vector with coordinates, transpose to get dimensions correctly
    rotated_pointcloud = np.transpose(np.reshape(rotated_pointcloud, (3, number_of_points)))  # reshape and transpose back

    # translate:
    training_pointcloud = rotated_pointcloud + translation # add translation to every coordinate vector

    return training_pointcloud
