from typing import Union
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.nn import Linear, init
from src.ops.gaudi_mask import gaudi_mask
from src.models.modules.olb import find_olb

try:
    from src.ops.low_rank import low_rank_project
except ModuleNotFoundError:
    pass

from einops import rearrange


class GaudiGBLRConv2d(nn.Module):
    def __init__(self, conv2d_layer, gaudi_params, unified_mask=False, init='lr'): #in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, 
                 #groups=1, bias=True, padding_mode='zeros', device=None, dtype=None,
                 #gaudi_params=None):
        super().__init__()

        self.unified_mask = unified_mask

        self.padding = conv2d_layer.padding
        self.stride = conv2d_layer.stride
        self.dilation = conv2d_layer.dilation
        self.groups = conv2d_layer.groups
        if conv2d_layer.bias is not None:
            self.bias = Parameter(conv2d_layer.bias.data)
        else:
            self.register_parameter('bias', None)

        gaudi_params['bias'] = False
        self.kernel_size = conv2d_layer.kernel_size
        module_list = []
        self.num_modules = self.kernel_size[0] * self.kernel_size[1]
        for i in range(self.num_modules):
            m = GaudiGBLR(in_features=conv2d_layer.in_channels, out_features=conv2d_layer.out_channels,
                                         **gaudi_params)
            M = conv2d_layer.weight[:,:,i//self.kernel_size[1], i%self.kernel_size[1]]
            if init == 'lr':
                m.set_weights_from_projection(M)
            elif init == 'gblr':
                budget = int(0.125 * min(*M.shape) * sum(M.shape))
                w,l,U,Vt = find_olb(M=M, budget=budget,
                                    thres_row_list=[0.95],
                                    thres_col_list=[0.95],
                                    weight_lr=0.005,
                                    structure_lr_base=1,
                                    verbose=False,
                                    niter=100,
                                    sched_params={'start_factor': 1.0, 'end_factor': 0.01},
                                    )
                m.lr_weight1.data = Vt.data
                m.lr_weight2.data = U.data
                m.widths.data = w.data
                m.locations.data = l.data

            module_list.append(m)
        self.gaudi_modules = nn.ModuleList(module_list)




    #def set_weights_from_projection(self, w):
    #    w_shape = w.shape
    #    #w = rearrange(w, 'out_c in_c k1 k2 -> out_c (in_c k1 k2)')
    #    w = rearrange(w, 'out_c in_c k1 k2 -> out_c (k1 k2 in_c)')
    #    self.gaudi_module.set_weights_from_projection(w)


    def forward(self, x):
        self.input_shape = x.shape[2:]
        weight_list = []
        mask1 = self.gaudi_modules[0].get_mask_by_ind(0) if self.unified_mask else None
        mask2 = self.gaudi_modules[0].get_mask_by_ind(1) if self.unified_mask else None
        for i in range(self.num_modules):
            w = self.gaudi_modules[i].get_masked_matrix(mask1, mask2)
            weight_list.append(w)
        weight = rearrange(weight_list, '(k1 k2) oc ic -> oc ic k1 k2', k1=self.kernel_size[0], k2=self.kernel_size[1])
        #x = self.conv2d_layer(x)
        #w = self.gaudi_module.get_masked_matrix()
        #w = rearrange(w, 'out_c (k1 k2 in_c) -> out_c in_c k1 k2', k1=self.kernel_size[0], k2=self.kernel_size[1], out_c=self.out_channels, in_c=self.in_channels)
        #w = w.view(x.size(1), x.size(1), 1, 1)
        x = F.conv2d(x, weight, self.bias, padding=self.padding, stride=self.stride, groups=self.groups, dilation=self.dilation)
        return x



class GaudiGBLRConv2dIntegrated(nn.Module):
    def __init__(self, conv2d_layer, gaudi_params, unified_mask=False, init='lr'): 
        super().__init__()

        self.unified_mask = unified_mask

        self.padding = conv2d_layer.padding
        self.stride = conv2d_layer.stride
        self.dilation = conv2d_layer.dilation
        self.groups = conv2d_layer.groups
        if conv2d_layer.bias is not None:
            self.bias = Parameter(conv2d_layer.bias.data)
        else:
            self.register_parameter('bias', None)

        gaudi_params['bias'] = False
        self.kernel_size = conv2d_layer.kernel_size
        m = GaudiGBLR(in_features=conv2d_layer.in_channels * self.kernel_size[0]* self.kernel_size[1], 
                          out_features=conv2d_layer.out_channels,
                          **gaudi_params)
        M = rearrange(conv2d_layer.weight, 'oc ic k1 k2 -> oc (k1 k2 ic)')

        if init == 'lr':
            m.set_weights_from_projection(M)
        elif init == 'gblr':
            budget = int(0.35 * min(*M.shape) * sum(M.shape))
            w,l,U,Vt = find_olb(M=M, budget=budget,
                                thres_row_list=[0.95],
                                thres_col_list=[0.95],
                                weight_lr=0.005,
                                structure_lr_base=10,
                                verbose=False,
                                niter=1000,
                                sched_params={'start_factor': 1.0, 'end_factor': 0.01},
                                )
            m.lr_weight1.data = Vt.data
            m.lr_weight2.data = U.data
            m.widths.data = w.data
            m.locations.data = l.data



        self.gaudi_module = m




    def forward(self, x):
        self.input_shape = x.shape[2:]
        w = self.gaudi_module.get_masked_matrix()
        weight = rearrange(w, 'oc (k1 k2 ic) -> oc ic k1 k2', k1=self.kernel_size[0], k2=self.kernel_size[1])
        x = F.conv2d(x, weight, self.bias, padding=self.padding, stride=self.stride, groups=self.groups, dilation=self.dilation)
        return x







class GaudiGBLR(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int,
                 rank_per_component: Union[int, float],
                 num_components: int = None,
                 total_rank_ratio: float = None,
                 sigma: float = 100.0,
                 min_widths=None,
                 max_widths=None,
                 compute_mode="dense",
                 no_gaussian=False,
                 no_ste=False,
                 adam_betas=(0.0,0.999),
                 momentum=0.9,
                 fixed_location=False,
                 fixed_width=False,
                 fixed_weight=False,
                 width_weight_decay=None,
                 width_learning_rate=None,
                 location_learning_rate=None,
                 bias: bool=True, init='linear', weight_decay: bool = True,
                 width_init='splr',
                 location_init='splr',
                 custom_grad=False,
                 device=None, dtype=None, **kwargs) -> None:

        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.custom_grad = custom_grad
        self.no_gaussian = no_gaussian
        self.no_ste = no_ste
        self.compute_mode = compute_mode
        self.buffer = None
        assert self.compute_mode in ['lr', 'dense']


        if isinstance(rank_per_component, float):
            self.rank_per_component = int(rank_per_component * min(self.in_features, self.out_features))
        else:
            self.rank_per_component = rank_per_component
        if total_rank_ratio is not None:
            self.total_rank_ratio = total_rank_ratio
            self.num_components = int(total_rank_ratio * min(self.in_features, self.out_features) // self.rank_per_component)
        elif num_components is not None:
            self.num_components = min(min(self.in_features, self.out_features) // self.rank_per_component, num_components)
            self.total_rank_ratio = min(1.0, self.num_components * self.rank_per_component / min(self.in_features, self.out_features))
        else:
            raise ValueError("Either num_components or total_rank must be given.")


        # Mask Params
        self.widths = Parameter(torch.empty(2, self.num_components))
        self.locations = Parameter(torch.empty(2, self.num_components))


        self.widths._optim = {"betas": adam_betas, "momentum": momentum}
        self.locations._optim = {"weight_decay": 0.0, "betas": adam_betas, "momentum": momentum}
        if width_learning_rate is not None:
            self.widths._optim["lr"] = width_learning_rate
        if width_weight_decay is not None:
            self.widths._optim["weight_decay"] = width_weight_decay
        if location_learning_rate is not None:
            self.locations._optim["lr"] = location_learning_rate

        self.optim_dict={"widths": self.widths._optim, "locations": self.locations._optim}


        if min_widths is None:
            min_widths = (0., 0.)
        elif isinstance(min_widths[0], int):
            min_widths = (min_widths[0] / float(self.in_features), min_widths[1] / float(self.out_features))
        if max_widths is None:
            max_widths = (1.0, 1.0)
        elif isinstance(max_widths[0], int):
            max_widths = (max_widths[0] / float(self.in_features), max_widths[1] / float(self.out_features))

        if max_widths[0] < min_widths[0]:
            max_widths[0] = min_widths[0]
            print("Warning: max_widths[0] updated")
        if max_widths[1] < min_widths[1]:
            max_widths[1] = min_widths[1]
            print("Warning: max_widths[1] updated")

        self.min_widths = min_widths
        self.max_widths = max_widths



        freq_len_in = self.in_features // 2 + 1
        freq_len_out = self.out_features // 2 + 1

        self.register_buffer("freq_in", 
                    torch.arange(0.0, freq_len_in, 1)
                )
        self.register_buffer("freq_out", 
                    torch.arange(0.0, freq_len_out, 1)
                )
        self.register_buffer("sigma", torch.tensor(sigma))
        

        self.width_init = width_init
        self.location_init = location_init


        self.weight_decay = weight_decay
        self.set_weights()

        if fixed_width:
            self.widths.requires_grad = False
        if fixed_location:
            self.locations.requires_grad = False
        if fixed_weight:
            self.lr_weight1.requires_grad = False
            self.lr_weight2.requires_grad = False

        if bias:
            self.bias = Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()


    def set_weights(self):
        self.rank = self.num_components * self.rank_per_component
        self.lr_weight1 = Parameter(torch.empty((self.num_components, self.rank_per_component, self.in_features)))
        self.lr_weight2 = Parameter(torch.empty((self.num_components, self.out_features, self.rank_per_component)))
        if not self.weight_decay:
            self.lr_weight1._no_weight_decay = True
            self.lr_weight2._no_weight_decay = True

        for lr_weight in [self.lr_weight1, self.lr_weight2]:
            fan_in = lr_weight.shape[-1]
            gain = init.calculate_gain(nonlinearity='leaky_relu', param=math.sqrt(5))
            std = gain / math.sqrt(fan_in)
            bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
            with torch.no_grad():
                lr_weight.uniform_(-bound, bound)


    def reset_parameters(self) -> None:

        if self.bias is not None:
            try:
                w = self.lr_weight1
            except AttributeError:
                w = self.weights
            fan_in, _ = init._calculate_fan_in_and_fan_out(w)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

        self.reset_mask_parameters()
       


    def reset_mask_parameters(self) -> None:
        if isinstance(self.location_init, float):
            init.constant_(self.locations, self.location_init)

        elif 'linspace' in self.location_init:
            loc = torch.linspace(0.,self.num_components-1,self.num_components).unsqueeze(0) / self.num_components
            self.locations.data = torch.cat([loc, loc], dim=0)
        else:
            init.uniform_(self.locations, 0.0, 1.0)
 
        if isinstance(self.width_init, float):
            init.constant_(self.widths, self.width_init)
        elif isinstance(self.width_init, str):
            if 'unif' in self.width_init:
                init.uniform_(self.widths, 0.0, 1.0)

            elif 'splr' in self.width_init:
                ratio = self.width_init.split("lr")[-1]
                if len(ratio) == 0:
                    ratio = 1 / self.saving
                else:
                    ratio = float(ratio)
                num_nonzero_comp = int(self.num_components * ratio * 0.5)
                with torch.no_grad():
                    ones = torch.ones_like(self.widths[:, :num_nonzero_comp])
                    shorts = torch.ones_like(self.widths[:, num_nonzero_comp:]) * 0.5 * ratio / (1 - 0.5 * ratio)
                    self.widths.data = torch.cat([ones, shorts], dim=1)
 

            elif 'lr' in self.width_init:
                ratio = self.width_init.split("lr")[-1]
                if len(ratio) == 0:
                    ratio = 1 / self.saving
                else:
                    ratio = float(ratio)
                num_nonzero_comp = int(self.num_components * ratio)
                with torch.no_grad():
                    nonzeros = torch.ones_like(self.widths[:, :num_nonzero_comp])
                    zeros = torch.zeros_like(self.widths[:, num_nonzero_comp:])
                    #zeros[0,:]=1/self.in_features
                    zeros[0,::2]=1/self.in_features
                    zeros[1,1::2] = 1 / self.out_features
                    self.widths.data = torch.cat([nonzeros, zeros], dim=1)
            elif 'zero' in self.width_init:
                self.widths.data[0,::2] = 0.0
                self.widths.data[0,1::2] = 1 / self.in_features 
                self.widths.data[1,1::2] = 0.0
                self.widths.data[1,::2] = 1 / self.out_features 
                    
                   


    def set_weights_from_projection(self, weight):
        with torch.no_grad():
            U, Vt = low_rank_project(weight, rank=self.rank)

            Vt = Vt.view(self.num_components, self.rank_per_component, self.in_features)
            U = U.view(self.out_features, self.num_components, self.rank_per_component).transpose(0,1)

            self.lr_weight1.copy_(Vt)
            self.lr_weight2.copy_(U)

    def set_weights_from_dense_init(self, dense_init_fn_):
        with torch.no_grad():
            dense_weight = torch.empty(self.out_features, self.in_features,
                                       device=self.lr_weight1.device, dtype=self.lr_weight1.dtype)
            dense_init_fn_(dense_weight)
            self.set_weights_from_projection(dense_weight)

    @property
    def saving(self):
        return ((self.lr_weight1.numel() + self.lr_weight2.numel())
                / (self.in_features * self.out_features))


    def get_width(self, ind):
        assert ind==0 or ind==1
        w = self.widths[ind, :]
        if not (self.no_ste and self.training):
            if ind==0:
                n = self.in_features
            else:
                n = self.out_features
            w_quantized = torch.round((w*n)) / n
            w_ste = w - w.detach() + w_quantized
        return w_ste 

    def get_nonzero_width_mask(self, width1=None, width2=None):
        if width1 is None:
            width1 = self.get_width(0)
        if width2 is None:
            width2 = self.get_width(1)
        m = torch.logical_and(width1 > 1.0 / self.in_features, width2 > 1.0 / self.out_features).float()
        return m

    def get_mean_width(self, ind):
        m = self.get_nonzero_width_mask()
        return (m * self.get_width(ind)).mean()

    def get_num_params(self):
        with torch.no_grad():
            w1 = self.get_width(0)
            w2 = self.get_width(1)
            m = self.get_nonzero_width_mask(w1, w2)

            m1 = self.get_mask_by_ind(0).squeeze()
            m2 = self.get_mask_by_ind(1).squeeze()
            w1 = torch.mean((m1 > 1e-3).float(), dim=1) * m
            w2 = torch.mean((m2 > 1e-3).float(), dim=1) * m

            U_num_el = torch.ceil(w2 *self.out_features)
            V_num_el = torch.ceil(w1 *self.in_features)
            num_flops_mm = U_num_el * V_num_el 
            num_flops_lr = (U_num_el + V_num_el) * self.rank_per_component
            flops = torch.minimum(num_flops_lr, num_flops_mm).sum()
            return flops


    def resize_model(self):
        with torch.no_grad():
            w1 = self.get_width(0)
            w2 = self.get_width(1)
            m = self.get_nonzero_width_mask(w1, w2)
            self.num_components = torch.sum(m).int().item()
            m = m > 0
            new_width = self.widths[:,m].clone().detach()
            new_loc = self.locations[:,m].clone().detach()
            new_lr_weight1 = self.lr_weight1[m,:,:].clone().detach()
            new_lr_weight2 = self.lr_weight2[m,:,:].clone().detach()
            self.widths.data = new_width
            self.locations.data = new_loc
            self.lr_weight1.data = new_lr_weight1.squeeze()
            self.lr_weight2.data = new_lr_weight2.squeeze().T
            
            


    def get_mask(self, w, loc, freq, sigma, n):
        # Inputs
        #   w: (num_components,) 
        #   loc: (num_components,) 
        #   freq: (channels,)
        #   sigma: (1,)
        # Output
        #   mask: (num_components, channels)

        if not self.training:
            w = F.hardshrink(w, 1.0/n)

        if not (self.training and self.no_ste):
            loc_quantized = torch.round((loc*n)) / n
            loc_ste = loc - loc.detach() + loc_quantized
            loc = loc_ste

        if self.custom_grad:
            if self.no_gaussian:
                sigma = None
            mask = gaudi_mask(w,loc,freq,sigma,n)
            if not self.training:
                mask[torch.abs(mask) < 1e-3] = 0.0
            return mask


        w = w.view(-1, 1)
        loc = loc.view(-1, 1)
        loc = loc * 2 * torch.pi
        freq = freq.view(1, -1)
        exponent_imag = (-freq * loc + torch.pi * freq * (1.0 / n - w)) 
        if self.no_gaussian or sigma is None:
            exponent_real = 0.0
        else:
            exponent_real = -0.5 / sigma**2 * freq ** 2

        exponent = exponent_real + exponent_imag * torch.tensor(1.0j, device=freq.device)
        mask = w * torch.sinc(freq * w) / torch.sinc(freq / n) * torch.exp(exponent)
        mask = torch.fft.irfft(mask, n=n, norm='forward')

        if not self.training:
            mask[torch.abs(mask) < 1e-3] = 0.0
        return mask


    def get_mask_by_ind(self, ind):
        if ind == 0:
            in_f = self.in_features
            return self.get_mask(
                        self.get_width(0),
                        self.locations[0,:], 
                        self.freq_in, 
                        self.sigma, 
                        in_f,
                        ).unsqueeze(1)
        elif ind == 1:
            out_f = self.out_features
            return self.get_mask(
                        self.get_width(1),
                        self.locations[1,:],
                        self.freq_out,
                        self.sigma,
                        out_f,
                        ).unsqueeze(2)


        else:
            raise ValueError("ind must be either 0 or 1, got:", ind)


    def get_masked_matrix(self, mask1=None, mask2=None):
        m1 = self.get_mask_by_ind(0) if mask1 is None else mask1
        m2 = self.get_mask_by_ind(1) if mask2 is None else mask2
        w1 = rearrange(self.lr_weight1 * m1, 'nc rpc in_f -> (nc rpc) in_f')
        w2 = rearrange(self.lr_weight2 * m2, 'nc out_f rpc -> out_f (nc rpc)')
        masked_w = w2 @ w1
        return masked_w


    def forward(self, input: Tensor) -> Tensor:

        B, N, C = input.size()
        if C < self.in_features:
            input = F.pad(input, (0, self.in_features - C))

        if self.compute_mode == "lr":

            #m1 = self.get_mask_by_ind(0)
            #m2 = self.get_mask_by_ind(1)
            #w1 = rearrange(self.lr_weight1 * m1, 'nc rpc in_f -> (nc rpc) in_f')
            #w2 = rearrange(self.lr_weight2 * m2, 'nc out_f rpc -> out_f (nc rpc)')
            w1 = self.lr_weight1
            w2 = self.lr_weight2
            out = F.linear(F.linear(input, w1), w2, self.bias)

            #out = out[:,:,:self.out_features]

            #if self.bias is not None:
            #    out += self.bias.to(dtype=out.dtype)
            
        elif self.compute_mode=='dense':
            if hasattr(self, 'weight'):
                out = F.linear(input, self.weight, self.bias.to(dtype=input.dtype)) 
            else:
                masked_w = self.get_masked_matrix() 
                out = F.linear(input, masked_w, self.bias.to(dtype=input.dtype))

        return out



def test():
    in_features = 512
    out_features = 512 * 4
    rank_per_component = 16
    num_components = in_features // rank_per_component


    m = GaudiGBLR(in_features, out_features, rank_per_component, num_components)

    x = torch.randn(3,4,512)
    y = m(x)
    print(y.size())


if __name__=="__main__":
    test()

