import numpy as np
import pandas as pd
import random
import time
import pickle
'''
path_to_ply = '/Users/sabinalinderoth/Desktop/Ply_files_1/TEST_sorted_grid_ply_1/grid_13_10/070832.ply'
point_cloud = pd.read_csv(path_to_ply, delimiter=' ', skiprows=7, header=None, names=('x','y','z'))
point_cloud = point_cloud.values

pc_list = point_cloud.tolist()
print(len(pc_list[0][:]))

a = np.array([0,1,2,3,4,5])
print(type(a))
'''


#dict_sample = np.load('/Users/sabinalinderoth/Documents/master_thesis/point_cloud_input/test_3/training_sample_1.npy')
for k in list(range(0,10)):
    print('round ' + str(k))
    for i in list(range(1,11)):

        t1 = time.time()

        file_name = '/Users/sabinalinderoth/Documents/master_thesis/point_cloud_input/data_set_190321/training_sample_' + \
                    str(i)
        pickle_in = open(file_name, "rb")
        dict_sample = pickle.load(pickle_in)
        t2 = time.time()

        string = 'time to load sample ' + str(i) + ':'
        print(string ,t2-t1)




#sweep = dict_sample['sweep']
#map = dict_sample['map']
#label = dict_sample['labels']


#print(dict)