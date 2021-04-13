# basic testing
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import time

import torch
import MVC

#torch.manual_seed(0)
input = torch.rand(1, 4, 32, 32, 32, 45).to('cuda')
input = input/torch.norm(input,dim=5)[...,None]
kernel_size = 3
weights = torch.ones(8, kernel_size, kernel_size, kernel_size, input.shape[1]).to('cuda')
output = MVC.forward(input, weights)

print(output.mean())

from baseline.intra_voxel import IntraVoxelConv
conv_og = IntraVoxelConv(weights.shape[-1], weights.shape[0], kernel_size, 1)
weights_ = weights.permute(0, 4, 1, 2, 3)
conv_og.weight_matrix = torch.nn.Parameter(weights_.reshape(weights.shape[0], input.shape[1]*(kernel_size**3)), requires_grad=True)

with torch.no_grad():
    output_og = conv_og(input)


print(output_og.mean())
