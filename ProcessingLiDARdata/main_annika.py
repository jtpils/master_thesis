import numpy as np
from lidar_data_functions import *
from matplotlib import pyplot as plt
import pandas as pd


path = '/home/master04/Documents/master_thesis/ProcessingLiDARdata/fake_training_set/samples/1.npy'
pc = np.load(path)
visualize_detections(pc,layer=0, fig_num=1)

path = '/home/master04/Documents/master_thesis/ProcessingLiDARdata/fake_training_set/samples/1.npy'
pc = np.load(path)
visualize_detections(pc,layer=1, fig_num=2)
plt.show()

#plt.imshow(pc[0,:,:], cmap='gray')
#plt.show()
'''
# Uncomment for visualisation of the sweep and cut_out
layer = 2
max_value = np.max(pc[layer, :, :])
print('Max max_value in array_to_png: ', max_value)

# avoid division with 0
if max_value == 0:
    max_value = 1

scale = 255 / max_value
pc[layer, :, :] = pc[layer, :, :] * scale
print('Largest pixel value (should be 255) : ', np.max(pc[layer, :, :]))


img = Image.fromarray(pc[layer, :, :])
new_img = img.convert("L")
new_img.rotate(180).show()
'''


