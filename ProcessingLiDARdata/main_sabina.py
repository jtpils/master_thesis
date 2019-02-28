import numpy as np
from lidar_data_functions import *
from PIL import Image
import time
import matplotlib.pyplot as plt
import os




a = np.array([[1, 2, 3],[1, 1, 1],[1, 3, 2]])

b = np.array([[1, 1, 1, 2, 3, 1], [2, 3, 1, 4, 1, 1], [2, 3, 1, 4, 1, 1]])

print(a @ b)
