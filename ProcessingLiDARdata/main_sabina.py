import numpy as np
from lidar_data_functions import *
import random
from PIL import Image
import time
import matplotlib.pyplot as plt
import os
import sys
import random
import math

'''
def ReLU(x):

    return x * (x > 0)

x = np.arange(-5,5,0.1)
x_relu = np.arange(-5,1.1,0.1)

tanh_function = np.array([np.tanh(xi) for xi in x])
relu_function = np.array([ReLU(xi) for xi in x_relu])

yline = np.array([-1,1])
ylinex = np.array([0,0])


xline = np.array([-5,6])
xliney = np.array([0,0])

plt.plot(ylinex,yline,'k')
plt.plot(xline,xliney,'k')
plt.plot(x,tanh_function, label='tanh(x)')
plt.plot(x_relu, relu_function, label = 'ReLU(x)')
plt.xlabel('x')
plt.ylabel('y')
plt.title("Activation functions")
plt.legend()
plt.show()
'''

path_array = ['path_to_map_1', 'path_to_map_2']

for i in path_array:
    print(i)





