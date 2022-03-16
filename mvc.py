import MVC
import torch
from mv import MVLog, MVExp

class MVC_function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weights):
        output_log = MVC.log_forward(input, weights)
        output_exp = MVC.exp_forward(output_log, input)
        variables = [input, output_log, weights]
        ctx.save_for_backward(*variables)
        return output_exp

    @staticmethod
    def backward(ctx, grad_exp):
        input, output_log, weights = ctx.saved_tensors
        grad_log, grad_M = MVC.exp_backward(output_log, input, grad_exp)
        grad_x = MVC.log_backward(input, weights, grad_log)
        grad_weights = MVC.log_weights_backward(input, weights, grad_log)
        return grad_x+grad_M, grad_weights
MVC_apply = MVC_function.apply

class ManifoldValuedConv(torch.nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, zero_init=False):
        super().__init__()
        if zero_init:
            self.weight_mask = torch.nn.Parameter(torch.zeros([output_channels, kernel_size, kernel_size, kernel_size, input_channels], requires_grad=True))
        else:
            self.weight_mask = torch.nn.Parameter(torch.rand([output_channels, kernel_size, kernel_size, kernel_size, input_channels], requires_grad=True))
    
    def forward(self, x):
        #return MVC_apply(x, self.weight_mask)
        return MVExp(MVLog(x, self.weight_mask), x)

class ManifoldValuedVolterra(torch.nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size=3, zero_init=False):
        super().__init__()
        if zero_init:
            self.weight_mask1 = torch.nn.Parameter(torch.zeros([output_channels, kernel_size, kernel_size, kernel_size, input_channels], requires_grad=True))
            self.weight_mask2 = torch.nn.Parameter(torch.zeros([output_channels, kernel_size, kernel_size, kernel_size, input_channels], requires_grad=True))
            self.weight_mask3 = torch.nn.Parameter(torch.zeros([output_channels, kernel_size, kernel_size, kernel_size, input_channels], requires_grad=True))
        else:
            self.weight_mask1 = torch.nn.Parameter(torch.rand([output_channels, kernel_size, kernel_size, kernel_size, input_channels], requires_grad=True))
            self.weight_mask2 = torch.nn.Parameter(torch.rand([output_channels, kernel_size, kernel_size, kernel_size, input_channels], requires_grad=True))
            self.weight_mask3 = torch.nn.Parameter(torch.rand([output_channels, kernel_size, kernel_size, kernel_size, input_channels], requires_grad=True))
    
    def forward(self, x):
        log1 = MVLog(x, self.weight_mask1)
        log2 = MVLog(x, self.weight_mask2)
        log3 = MVLog(x, self.weight_mask3)
        return MVCExp(log1+log2*log3, x)
