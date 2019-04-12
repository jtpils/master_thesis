import torch


num_gpu = torch.cuda.device_count()

print('Number of gpu:s: ', num_gpu)
