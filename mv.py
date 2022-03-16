import MVC
import torch

class LogFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weights):
        output_log = MVC.log_forward(input, weights)
        variables = [input, weights]
        ctx.save_for_backward(*variables)
        return output_log

    @staticmethod
    def backward(ctx, grad_log):
        input, weights = ctx.saved_tensors
        grad_x = MVC.log_backward(input, weights, grad_log)
        grad_weights = MVC.log_weights_backward(input, weights, grad_log)
        return grad_x, grad_weights
MVLog = LogFunction.apply

class ExpFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, original_input):
        output_exp = MVC.exp_forward(input, original_input)
        variables = [input, original_input]
        ctx.save_for_backward(*variables)
        return output_exp

    @staticmethod
    def backward(ctx, grad_exp):
        input, original_input = ctx.saved_tensors
        grad_log, grad_M = MVC.exp_backward(input, original_input, grad_exp)
        return grad_log, grad_M
MVExp = ExpFunction.apply
