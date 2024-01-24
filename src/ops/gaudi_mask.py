import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class GaudiMask(torch.autograd.Function):
    @staticmethod
    def forward(ctx, width, loc, freq, sigma, n):
        k = freq
        w = width
        l = loc
        k_n = k/n

        Ck_exponent = torch.pi * 1.0j * (k_n  - 2.0 * k * l)
        if sigma is not None:
            Ck_exponent -= 0.5 * k **2 / sigma**2
        Ck = torch.exp(Ck_exponent) / torch.sinc(k_n) # C_k
        
        #S = torch.exp(-2.0j * torch.pi * k * l) # Shift

        result = w * torch.sinc(w*k) * torch.exp(-1.0j * torch.pi * k * w) * Ck
        ctx.save_for_backward(w, l, k, torch.tensor(n), Ck, result)
        return torch.fft.irfft(result, n=n, norm='forward')

    @staticmethod
    def backward(ctx, grad_output):
        w, l, k, n, Ck, result = ctx.saved_tensors
        grad_width = grad_loc = grad_freq = grad_sigma = grad_n = None
        if ctx.needs_input_grad[0]:
            grad_width_freq = torch.exp(-2.0j * torch.pi * k * (w)) * Ck
            grad_width = grad_output *torch.fft.irfft(grad_width_freq, n=n, norm='forward')
        if ctx.needs_input_grad[1]:
            grad_loc = grad_output * torch.fft.irfft(result * k * (-2.0j) * (torch.pi), n=n, norm='forward')

        return grad_width, grad_loc, grad_freq, grad_sigma, grad_n



def gaudi_mask(width, loc, freq, sigma, n):
    width = width.view(-1,1)
    loc = loc.view(-1, 1)
    freq = freq.view(1,-1)
    result = GaudiMask.apply(width, loc, freq, sigma, n)
    return result
