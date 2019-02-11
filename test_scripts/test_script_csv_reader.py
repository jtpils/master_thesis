
# This test script tests how to read the created csv file with the lidar data.
# It seems that it saves a vector that stacks all the rows in the csv file.
# It is possible to get a column and it is also possible to get single elements.
#







import glob
import os
import sys

try:
    sys.path.append(glob.glob('**/*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import numpy as np
import pandas as pd

import random
import time
import csv




def main():


    csv_file_name = 'lidardata_190130.csv'

    r = np.genfromtxt(csv_file_name, delimiter=',', names=True, case_sensitive=True)
    #print('prints the representation' , repr(r))

    #print('print just the frame_number' , r['frame_number'])

    #print('print the array: ' , r)

    #print('print the shape and size: ' , r.shape, r.size)

    #print('print the type: ' , type(r))

    #print('print the element: ' , r[0][1])



if __name__ == '__main__':

    main()


