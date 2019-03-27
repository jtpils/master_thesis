from data_set_point_cloud import *
from lidar_processing_functions import *
from preprocessing_data_functions import *
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
from train_network import *

def PointPillarsScatter(PFN_input, PFN_output, batch_size):
    '''
    :param PFN_input: <tensor> size: [number_of_batches, 8, max_number_of_pillars, max_number_of_points]. The feature
            tensor that used as an input to the PFN layer. This is used to find the original location of the pillars
    :param PFN_output: <tensor> size: [number_of_batches, 64, max_number_of_pillars]. The output from the PFN layer.
            Containing the pillars that should be scattered back.
    :param batch_size: <int> size of the batch
    :return: batch_canvas: <list> lsit of canvas from each batch.
    '''

    num_channels = 64
    height = 282
    width = 282
    pillar_size = 0.16
    x_edges = np.arange(-22, 22, pillar_size)
    y_edges = np.arange(-22, 22, pillar_size)

    batch_canvas = []
    for batch in np.arange(batch_size):

        pillar_list = np.nonzero(PFN_input[batch, 0, :, 0])
        canvas = np.zeros((num_channels, height, width))
        for pillar in pillar_list[0]:
            x, y = PFN_input[batch, 0, pillar, 0], PFN_input[batch, 1, pillar, 0]
            xgrid, ygrid = get_grid(x, y, x_edges, y_edges)
            pillar_vector = PFN_output[batch, :, pillar]
            pillar_vector = np.squeeze(pillar_vector)
            #canvas[:, ygrid, xgrid] = pillar_vector  # swap x and y?
            canvas[:, xgrid, ygrid] = pillar_vector
        batch_canvas.append(canvas)

    return canvas


def fasterScatter(PFN_input, PFN_output, batch_size):

    num_channels = 64
    height = 282
    width = 282
    pillar_size = 0.16
    range = 22

    batch_canvas = []
    for batch in np.arange(batch_size):

        pillar_list = np.nonzero(PFN_input[batch, 0, :, 0])[0] # ???? List or array??? collect all non-empty pillars
        #pillars = np.resize(PFN_output, (64, 12000))

        x_coords = PFN_input[batch,0,pillar_list,0]  # 1 xcoord from each pillar # should we use the pillar_list here instead of ":" ?
        xgrids = np.floor((x_coords + range) / pillar_size).astype(int)
        y_coords = PFN_input[batch,1,pillar_list,0]  # first xvalue in each pillar
        ygrids = np.floor((y_coords + range) / pillar_size).astype(int)

        #convert these 2D-indices to 1D indices by declaring canvas as:
        canvas = np.zeros((num_channels, height*width))
        indices = xgrids*width + ygrids  # new indices along 1D-canvas. or maybe swap x and y here?
        #indices = (height-xgrids)*height -ygrids-1
        indices = np.squeeze(indices)
        #pillar_vector = np.resize(pillars, (64, height*width)) # reshape to 1 row
        pillars_to_canvas = PFN_output[batch,:,pillar_list]

        canvas[:, indices] = np.transpose(pillars_to_canvas)  # np.transpose(PFN_output[batch,:,pillar_list])
        canvas = np.resize(canvas, (num_channels, height, width))

    batch_canvas.append(canvas)


    return canvas



path_data_set = '/home/master04/Desktop/Dataset/point_cloud/pc_small_set'
sample = pickle.load(open('/home/master04/Desktop/Dataset/point_cloud/pc_small_set/training_sample_0','rb'))
cutout0 = sample['map']
sweep0 = sample['sweep']
input0 = np.resize(sweep0, (1,8,12000,100))
path_data_set = '/home/master04/Desktop/Dataset/point_cloud/pc_small_set'
sample = pickle.load(open('/home/master04/Desktop/Dataset/point_cloud/pc_small_set/training_sample_1','rb'))
cutout1 = sample['map']
sweep1 = sample['sweep']
input1 = np.resize(sweep1, (1,8,12000,100))
sample = np.concatenate((input0,input1), axis=0)

#input = np.ones((1,8,12000,100)) # (BATCH, D, P, N)
#input[:,:,2,:] = input[:,:,2,:]*5
output = np.ones((2,64,12000))*3  # 1 batch, (BATCH, C, P)

t1 = time.time()
c1 = PointPillarsScatter(sample, output, 2)
t2 = time.time()
c2 = fasterScatter(sample, output, 2)
t3 = time.time()

flag = (c1[0]==c2[0]).all()

print('Done', flag)
print(t2-t1, t3-t2)



'''
range = 22
pillar_size = 0.16
# find x_grids
x_coords = input[0,0,:,0]  # first xvalue in each pillar
# translate all points so that they begion from zero (add the range to move points from (-15,15) to (0,30)
# divide by pillar size, which will yield deciaml values where each new integer is a new pillar
# floor these values to get the actual grid indices
xgrids = np.floor((x_coords + range) / pillar_size).astype(int)
y_coords = input[0,1,:,0]  # first xvalue in each pillar
ygrids = np.floor((y_coords + range) / pillar_size).astype(int)

x_edges = np.arange(-22, 22, pillar_size)
y_edges = np.arange(-22, 22, pillar_size)

i = 111
x2,y2=get_grid(x_coords[i],y_coords[i],x_edges,y_edges)

print(xgrids[i],ygrids[i])
print(x2,y2)

'''




'''
print('discretizing sweep: ')
sweep = discretize_pointcloud(sample_dict['sweep'], array_size=60, trim_range=15, spatial_resolution=0.5, padding=False)
print(' ')
print('Creating png: ')
array_to_png(sweep, 'sweep_png')
del sweep

print('discretizing map: ')
cutout = discretize_pointcloud(sample_dict['map'], array_size=90, trim_range=22, spatial_resolution=0.5, padding=False)
print(' ')
print('Creating png: ')
array_to_png(cutout, 'map_png')
del cutout
'''

