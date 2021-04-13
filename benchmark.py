# basic testing
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import time

import torch
import MVC

#torch.manual_seed(0)
input = torch.rand(1, 1, 64, 64, 64, 45).to('cuda')
input = input/torch.norm(input,dim=5)[...,None]
kernel_size = 3
weights = torch.ones(1, kernel_size, kernel_size, kernel_size, input.shape[1]).to('cuda')

start = time.time()
output = MVC.forward(input, weights)
end = time.time()
torch.cuda.current_stream().synchronize()
print("Custom Kernel Memory: ", torch.cuda.max_memory_allocated())
print("Custom Kernel Time: ", end-start)

torch.cuda.reset_max_memory_allocated()

from baseline.intra_voxel import IntraVoxelConv
conv_og = IntraVoxelConv(weights.shape[-1], weights.shape[0], kernel_size, 1)
weights_ = weights.permute(0, 4, 1, 2, 3)
conv_og.weight_matrix = torch.nn.Parameter(weights_.reshape(weights.shape[0], input.shape[1]*(kernel_size**3)), requires_grad=True)

start = time.time()
with torch.no_grad():
    output_og = conv_og(input)
torch.cuda.current_stream().synchronize()
end = time.time()

print("Original Memory: ", torch.cuda.max_memory_allocated())
print("Original Time: ", end-start)


