import torch
#import pycuda.driver as cuda


#cuda.init()

print(torch.cuda.current_device())

#print(cuda.Device(0).name())

print(torch.cuda.is_available())
