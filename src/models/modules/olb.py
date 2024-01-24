import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.nn import init
from einops import rearrange
from tqdm import tqdm
from scipy.ndimage import gaussian_filter1d
from numba import jit
import time

from src.utils import utils
log = utils.get_logger(__name__)

try:
    from src.ops.low_rank import low_rank_project
except ModuleNotFoundError:
    pass


def get_trainable_clone(tensors):
    clone_list = []
    for tensor in tensors:
        clone_list.append(tensor.detach().clone().requires_grad_())
    return clone_list

def _gaussian_kernel1d(sigma, radius):
    """
    Computes a 1-D Gaussian convolution kernel.
    """
    order = 0
    exponent_range = np.arange(order + 1)
    sigma2 = sigma * sigma
    x = np.arange(-radius, radius+1)
    phi_x = np.exp(-0.5 / sigma2 * x ** 2)
    phi_x = phi_x / phi_x.sum()
    return phi_x

@jit(nopython=True, parallel=True)
def get_longest_mask(mask, corr, mode='loop'):
    mask = mask.astype(np.float64)
    corr = corr.astype(np.float64)
    if sum(mask) == 0:
        return mask, 0.0, 0.0

    cyclic = mask[-1] > 0 and mask[0] > 0

    if cyclic:
        #mask_cat = np.concatenate([mask, mask[:len(mask)//2]], axis=0)
        mask_cat = np.empty((len(corr)//2*3,))
        mask_cat[:len(mask)] = mask
        mask_cat[len(mask):] = mask[:len(mask)//2]

    else:
        mask_cat = mask

    if mode == 'largest':

        corr = np.abs(corr)
        if cyclic:
            corr_cat = np.empty((len(corr)//2*3,))
            corr_cat[:len(corr)] = corr
            corr_cat[len(corr):] = corr[:len(corr)//2]
        else:
            corr_cat = corr

        corr_max_ind = np.argmax(corr_cat)



        mask_len_below = 0
        for k in range(1, corr_max_ind):
            if mask_cat[corr_max_ind - k] > 0:
                mask_len_below +=1
            else:
                break

        mask_len_above = 0
        for k in range(1, len(mask_cat)-corr_max_ind):
            if mask_cat[corr_max_ind + k] > 0:
                mask_len_above += 1
            else:
                break


        mask_cyclic = np.zeros_like(mask_cat)
        #if mask_len_above + mask_len_below > 0:
        mask_cyclic[corr_max_ind-mask_len_below:corr_max_ind+mask_len_above] = 1.0

        mask = mask_cyclic[:len(mask)]
        if cyclic:
            mask[:len(mask)//2] += mask_cyclic[len(mask):len(mask)+len(mask)//2]
        mask = mask.clip(0.0, 1.0)

        width = min(1.0, (mask_len_below + mask_len_above + 1) / len(mask))
        loc = (corr_max_ind - mask_len_below) / len(mask) 
        loc = loc - np.floor(loc)
        return mask, width, loc


    elif mode == 'loop':
        ws = {}
        was_mask_val_one=False
        mask_len = 0
        mask_len_max = 0


        for k in range(len(mask_cat)):
            if not was_mask_val_one and mask_cat[k] > 0:
                was_mask_val_one = True
                mask_len=1
            elif was_mask_val_one:
                if mask_cat[k] > 0:
                    mask_len += 1

                if mask_cat[k] == 0 or k == len(mask_cat)-1:
                    ws[mask_len]= k - mask_len + 1
                    if mask_len > mask_len_max:
                        mask_len_max = mask_len
                    if mask_cat[k] == 0:
                        was_mask_val_one=False
            
        mask_cyclic = np.zeros_like(mask_cat)
        mask_cyclic[ws[mask_len_max]: ws[mask_len_max] + mask_len_max] = 1.0
        mask = mask_cyclic[:len(mask)]
        if cyclic:
            mask[:len(mask)//2] += mask_cyclic[len(mask):len(mask)+len(mask)//2]
        mask = mask.clip(0.0, 1.0)
        width = min(1.0, mask_len_max / len(mask))
        loc = ws[mask_len_max] / len(mask)
        loc = loc - np.floor(loc)
        return mask, width, loc 
    else:
        raise NotImplementedError()






def gaussian_filter1d_torch(input, sigma, axis=-1, order=0, output=None,
                            mode='reflect', cval=0.0, truncate=4.0, *, radius=None):
    nr, nc = input.size()
    sd = float(sigma)
    lw = int(truncate * sd + 0.5)
    if radius is not None:
        lw = radius
    weight = _gaussian_kernel1d(sigma, lw)[::-1].copy()
    weight = torch.tensor(np.reshape(weight, (1,1,-1)), dtype=input.dtype, device=input.device)
    input = input.view(nr,1,nc)
    input = F.pad(input, (lw, lw), mode='circular')
    out = F.conv1d(input, weight, padding='valid')
    return out.view(nr, nc)

    


def batched_mask(w, loc, freq, sigma, n):
    """
    w: (b, c)
    loc: (b, c)
    freq: (n,)
    sigma: float or FloatTensor, (1,)
    n: float, (1,)
    """
    w = w.unsqueeze(1)
    loc = loc.unsqueeze(1) * 2 * torch.pi
    freq = freq.unsqueeze(0).unsqueeze(0)

    exponent_imag = (-freq * loc + torch.pi * freq * (1.0 / n - w)) 
    if self.no_gaussian or sigma is None:
        exponent_real = 0.0
    else:
        exponent_real = -0.5 / sigma**2 * freq ** 2

    exponent = exponent_real + exponent_imag * torch.tensor(1.0j, device=freq.device)
    mask = w * torch.sinc(freq * w) / torch.sinc(freq / n) * torch.exp(exponent)
    mask = torch.fft.irfft(mask, n=n, norm='forward')
    return mask






class OLB(nn.Module):
    def __init__(self, size, rank_per_component, num_components,
                 use_direct_mask=False,
                 no_gaussian=False,
                 width_init=1.0,
                 location_init='rand',
                 alpha=1.0,
                 beta=1.0,
                 ):
        super().__init__()
        self.size = size
        self.num_rows = size[0] 
        self.num_cols = size[1] 
        self.rank_per_component = rank_per_component
        self.num_components = num_components
        self.no_gaussian = no_gaussian
        self.width_init = width_init
        self.location_init = location_init
        self.alpha = alpha
        self.beta = beta


        # Mask Params
        self.widths = Parameter(torch.empty(2, self.num_components))
        self.locations = Parameter(torch.empty(2, self.num_components))


        freq_len_row = self.num_rows // 2 + 1
        freq_len_col = self.num_cols // 2 + 1

        self.register_buffer("freq_row", 
                    torch.arange(0.0, freq_len_row, 1)
                )
        self.register_buffer("freq_col", 
                    torch.arange(0.0, freq_len_col, 1)
                )


        self.U = Parameter(torch.empty((self.num_components, self.num_rows, self.rank_per_component)))
        self.Vt = Parameter(torch.empty((self.num_components, self.rank_per_component, self.num_cols)))
        self.use_direct_mask = use_direct_mask
        if use_direct_mask:
            self.masks_row = Parameter(torch.empty((self.num_components, self.num_rows)))
            self.masks_col = Parameter(torch.empty((self.num_components, self.num_cols)))
        else:
            self.masks_row = None
            self.masks_col = None
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
            elif 'rand' in self.width_init:
                with torch.no_grad():
                    device = self.widths.device
                    dist = torch.distributions.beta.Beta(self.alpha, self.beta)
                    sample = dist.sample(self.widths.size())
                    #sample[0,:] *= self.max_widths[0]
                    #sample[1,:] *= self.max_widths[1]
                    if 'sort' in self.width_init:
                        sample, _ = torch.sort(sample, dim=-1, descending=True)
                    self.widths.data = sample.to(device)

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
                    self.widths.data = torch.cat([nonzeros, zeros], dim=1)
                    

    def set_weights(self, mat):
        with torch.no_grad():
            U, Vt = low_rank_project(mat, rank=self.rank_per_component * self.num_components)
            U = rearrange(U, 'row (nc rpc) -> nc row rpc', nc=self.num_components, rpc=self.rank_per_component)
            Vt = rearrange(Vt, '(nc rpc) col -> nc rpc col', nc=self.num_components, rpc=self.rank_per_component)
            self.U.copy_(U)
            self.Vt.copy_(Vt)

    def set_weights_from_dense_init(self, dense_init_fn_):
        with torch.no_grad():
            dense_weight = torch.empty(self.out_features, self.in_features,
                                       device=self.U.device, dtype=self.U.dtype)
            dense_init_fn_(dense_weight)
            self.set_weights(dense_weight)


    def get_quantized_param(self, param, n):
        p_qauntized = torch.round(param * n) / n
        return param + p_qauntized - param.detach()

    @property
    def saving(self):
        return ((self.U.numel() + self.Vt.numel())
                / (self.in_features * self.out_features))


    def get_num_params(self, wu=None, wv=None, exact=False, sigma=None):
        if self.use_direct_mask:
            return self.masks_row.sum() + self.masks_col.sum()
        if wu is None:
            wu = self.widths[0,:]
        if wv is None:
            wv = self.widths[1,:]
        if exact:
            nonzeros = torch.logical_and(wu >= 1/self.num_rows, wv >= 1/self.num_cols).float()
            wu = wu * nonzeros
            wv = wv * nonzeros
            return self.rank_per_component * (torch.sum(self.U_mask(w=wu, sigma=sigma)>1e-3) + torch.sum(self.V_mask(w=wv, sigma=sigma)>1e-3))
        else:
            return self.rank_per_component * torch.sum(wu * self.num_rows + wv * self.num_cols)


    def get_mask(self, w, loc, freq, sigma, n):
        w = w.unsqueeze(1)
        loc = loc.unsqueeze(1) * 2 * torch.pi
        freq = freq.unsqueeze(0)
    
        exponent_imag = (-freq * loc + torch.pi * freq * (1.0 / n - w)) 
        if self.no_gaussian or sigma is None:
            exponent_real = 0.0
        else:
            exponent_real = -0.5 / sigma**2 * freq ** 2

        exponent = exponent_real + exponent_imag * torch.tensor(1.0j, device=freq.device)
        mask = w * torch.sinc(freq * w) / torch.sinc(freq / n) * torch.exp(exponent)
        mask = torch.fft.irfft(mask, n=n, norm='forward')
        return mask



    def U_mask(self, w=None, l=None, sigma=None):
        if self.masks_row is not None:
            return self.masks_row
        if w is None:
            w = self.get_quantized_param(self.widths[0,:], self.num_rows)
        if l is None:
            l = self.get_quantized_param(self.locations[0,:], self.num_rows)
        return self.get_mask(w, l, self.freq_row, sigma, self.num_rows)

    def V_mask(self, w=None, l=None, sigma=None):
        if self.masks_col is not None:
            return self.masks_col
        if w is None:
            w = self.get_quantized_param(self.widths[1,:], self.num_rows)
        if l is None:
            l = self.get_quantized_param(self.locations[1,:], self.num_rows)
        return self.get_mask(w, l, self.freq_col, sigma, self.num_cols)


    def get_mask_mat(self):
        wu = self.get_quantized_param(self.widths[0,:], self.num_rows)
        lu = self.get_quantized_param(self.locations[0,:], self.num_rows)
        wv = self.get_quantized_param(self.widths[1,:], self.num_rows)
        lv = self.get_quantized_param(self.locations[1,:], self.num_rows)
        mu = self.U_mask(wu, lu).unsqueeze(2)
        mv = self.V_mask(wv, lv).unsqueeze(1)
        return torch.addbmm(torch.zeros(self.num_rows, self.num_cols, device=mu.device), mu, mv)

    def get_dense_matrix(self, mu, mv, U, Vt):
        mu = mu.unsqueeze(2)
        mv = mv.unsqueeze(1)
        U_masked = rearrange(U * mu, 'nc row rpc -> row (nc rpc)')
        Vt_masked = rearrange(Vt * mv, 'nc rpc col -> (nc rpc) col')
        return U_masked @ Vt_masked


    def get_sigma(self, current, total, start_sigma=1.0, end_sigma=1000.0, p=8):
        t = (current / total) ** p
        return start_sigma * (1-t) + end_sigma * t




    def filter_columns(self, M, sigma):
        if len(M.size()) == 1:
            M = M.view(1,-1)
        M = M.abs() / M.abs().max(1, keepdim=True)[0]
        M = gaussian_filter1d_torch(M, sigma)
        return M

    def thres_columns(self, M, thres):
        if len(M.size()) == 1:
            M = M.view(1,-1)
        M_col_max = M.abs().max(1, keepdim=True)[0]
        thres = thres * M_col_max
        M_thres = M > thres
        return M_thres.float()

    def filter_thres_columns(self, M, sigma, thres):
        return self.thres_columns(self.filter_columns(M, sigma), thres)

    @torch.no_grad()
    def find_mask(self, 
                  A_target,
                  budget,
                  k=0,
                  thres_row=0.05,
                  thres_col=0.05,
                  sigma=1.0,
                  p=1.0,
                  randomized=False,
                  verbose=False,
                  remove_processed=False,
                  use_longest_mask=True,
                  ):

        if self.use_direct_mask:
            self.masks_row.data = torch.zeros_like(self.masks_row)
            self.masks_col.data = torch.zeros_like(self.masks_col)

        #if thres_row == 0 and thres_col == 0:
        #    self.widths.fill_(1.0)
        #    self.locations.fill_(0.0)
        #    return self.widths[0,:], self.widths[1,:], self.locations[0,:], self.locations[1,:]

        R = A_target @ A_target.T
        C = A_target.T @ A_target

        for _ in range(k):
            R = R @ R
            C = C @ C

        R_f = self.filter_columns(R.T, sigma).T
        R_fth = self.thres_columns(R_f.T, thres_row).T
        C_f = self.filter_columns(C, sigma)
        C_fth = self.thres_columns(C_f, thres_col)

        device = A_target.device

        R = R.cpu()
        C = C.cpu()
        R_f = R_f.cpu()
        C_f = C_f.cpu()
        R_fth = R_fth.cpu()
        C_fth = C_fth.cpu()
        A_target = A_target.cpu()

        col_norms = torch.norm(C, dim=0)
        row_norms = torch.norm(R, dim=1)
        _, col_indices = torch.sort(col_norms, descending=True)
        _, row_indices = torch.sort(row_norms, descending=True)

        L = min(*A_target.size())
        
        current_num_mult = 0

        if randomized:
            probs = col_norms 
            if probs.sum() < 1e-6:
                return 
            cat_col = torch.distributions.categorical.Categorical(probs=probs**p)

        col_index = col_indices[0]
        for j in range(L):
            if randomized:
                col_index = cat_col.sample()


            mask_col = C_fth[:,col_index]
            if use_longest_mask:
                mask_col, wc, locc = get_longest_mask(mask_col.cpu().numpy(), C_f[:,col_index].cpu().numpy())
                mask_col = torch.tensor(mask_col, dtype=A_target.dtype, device=A_target.device)

            if randomized:
                probs = A_target[:,col_index].abs() 
                if probs.sum() < 1e-6:
                    row_index = 0
                else:
                    cat_row = torch.distributions.categorical.Categorical(probs=probs**p)
                    row_index = cat_row.sample()
            else:
                if mask_col.sum() == A_target.size(1):
                    row_index = row_indices[j]
                    row_f = R_f[row_index,:]
                    mask_row = R_fth[row_index,:]
                else:
                    A_col_masked = A_target[:, mask_col>1e-3]
                    A_col_masked = A_col_masked.to(device)

                    R_col_masked = A_col_masked @ A_col_masked.T
                    row_norms = torch.norm(R_col_masked, dim=1)
                    row_index = torch.argmax(row_norms)

                    row = R_col_masked[row_index,:]
                    row_f = self.filter_columns(row, sigma)[0,:]
                    mask_row = self.thres_columns(row_f, thres_row)[0,:]
            #mask_row = R_fth[row_index,:]

            if use_longest_mask:
                #mask_row, wr, locr = get_longest_mask(mask_row.cpu().numpy(), R_f[row_index,:].cpu().numpy())
                mask_row, wr, locr = get_longest_mask(mask_row.cpu().numpy(), row_f.cpu().numpy())
                mask_row = torch.tensor(mask_row, dtype=A_target.dtype, device=A_target.device)

            if self.use_direct_mask:
                self.masks_row.data[j,:] = mask_row.to(self.masks_row.device)
                self.masks_col.data[j,:] = mask_col.to(self.masks_col.device)
            else:
                self.widths.data[:,j] = torch.tensor([wr,wc], device=self.widths.device, dtype=self.widths.dtype)
                self.locations.data[:,j] = torch.tensor([locr, locc], device=self.locations.device, dtype=self.locations.dtype)
            if mask_row.sum() >0 and  mask_col.sum()>0:
                current_num_mult +=  mask_row.sum().item() + mask_col.sum().item()

            if j < L-1:
                col_index = col_indices[j+1]


            if verbose:
                print("{} - current num mult: {} - budget: {}".format(j, current_num_mult, budget))
            if current_num_mult > budget:
                if self.use_direct_mask:
                    self.masks_row.data[j:,:] = 0.
                    self.masks_col.data[j:,:] = 0.
                self.widths.data[:,j:]=0.0
                break
        return self.widths[0,:], self.widths[1,:], self.locations[0,:], self.locations[1,:]














           


    def project_parameters(self, vmin, vmax):
        wu = self.get_quantized_param(self.widths[0,:], self.num_rows)
        wv = self.get_quantized_param(self.widths[1,:], self.num_cols)
        lu = self.get_quantized_param(self.locations[0,:], self.num_rows)
        lv = self.get_quantized_param(self.locations[1,:], self.num_cols)
        self.widths.data[0,:] = wu
        self.widths.data[1,:] = wv
        self.locations.data[0,:] = lu
        self.locations.data[1,:] = lv
        self.widths.data = self.widths.data.clamp_(vmin, vmax)
        self.locations.data = self.locations.data - self.locations.data.floor()


    def find_mats(self, 
                  A_target,
                  niter=1000, 
                  lr=1e-3,
                  betas=(0.9,0.999),
                  sched_params={'start_factor':1.0, 'end_factor': 1.0},
                  sigma_params=None,
                  print_freq=100,
                  train_widths=False,
                  train_locations=False,
                  structure_lr=1e-2,
                  vmin=0.0,
                  vmax=1.0,
                  thres=1.0,
                  decay=0.0,
                  budget=None,
                  one_by_one=False,
                  verbose=False):

        assert not (train_widths and budget is None)

        param_groups = [{'params': [self.U, self.Vt], 'lr': lr, 'betas': betas, 'weight_decay': decay}]
        if train_locations:
            param_groups.append({'params': [self.locations], 'lr': structure_lr, 'betas': betas, 'weight_decay': 0.0})
        if train_widths:
            param_groups.append({'params': [self.widths], 'lr': structure_lr, 'betas': betas, 'weight_decay': 0.0})
        opt = torch.optim.AdamW(param_groups)
        sched_params['total_iters'] = niter
        sched = torch.optim.lr_scheduler.LinearLR(opt, **sched_params)

        A_fro_norm = torch.norm(A_target.detach()) 

        sigma = None

        self.project_parameters(vmin, vmax)

        for t in range(niter): 
            if one_by_one:
                self.U.requires_grad_(t%4==0)
                self.Vt.requires_grad_(t%4==1)
                self.widths.requires_grad_(t%4==2)
                self.locations.requires_grad_(t%4==3)
            opt.zero_grad()
            if sigma_params is not None:
                sigma = self.get_sigma(t, niter, **sigma_params)

            wu = self.get_quantized_param(self.widths[0,:], self.num_rows)
            wv = self.get_quantized_param(self.widths[1,:], self.num_cols)
            lu = self.get_quantized_param(self.locations[0,:], self.num_rows)
            lv = self.get_quantized_param(self.locations[1,:], self.num_cols)
            mask_u = self.U_mask(wu, lu, sigma=sigma)
            mask_v = self.V_mask(wv, lv, sigma=sigma)
            #mask_u[mask_u<1e-3]=0.0 
            #mask_v[mask_v<1e-3]=0.0 

            mu = mask_u.unsqueeze(2)
            mv = mask_v.unsqueeze(1)
            U_masked = rearrange(self.U * mu, 'nc row rpc -> row (nc rpc)')
            Vt_masked = rearrange(self.Vt * mv, 'nc rpc col -> (nc rpc) col')
            A_pred_masked = U_masked @ Vt_masked

            nrmse = torch.norm(A_pred_masked - A_target.detach()) / A_fro_norm



            with torch.no_grad():
                budget_loss = (self.get_num_params() / budget - 1.0)  ** 2

            loss = nrmse ** 2 

            loss.backward()
            opt.step()
            sched.step()

            if verbose and (t % print_freq == print_freq - 1 or t == 0):
                print("{:4d} - Loss: {:.4f} - Budget Loss: {:.4f} - NRMSE_masked: {} - Num Params: {}".format(t, loss.item(), budget_loss.item(), nrmse.item(), self.get_num_params().int().item()))


            with torch.no_grad():
                if train_widths:
                    #if one_by_one:
                    #    if t%2:
                    #        self.widths.data[0,:] = wu
                    #        self.locations.data[0,:] = lu
                    #    else:
                    #        self.widths.data[1,:] = wv
                    #        self.locations.data[1,:] = lv
                    budget_in_ratio = budget / (self.num_rows + self.num_cols)/min(*A_target.size())

                    if self.widths.mean() > budget_in_ratio: 
                    #flops = self.widths[0,:].sum() * A_target.size(0) + self.widths[1,:].sum() * A_target.size(1)
                    #if flops > budget:
                        #self.widths.data = F.softshrink(self.widths.data, opt.param_groups[-1]['lr'] * thres)
                        self.widths.data = self.widths.data -  (self.widths.mean() - budget_in_ratio)
                        #self.widths.data = self.widths.data - (flops - budget)/(self.widths.size(1) * (A_target.size(0)+A_target.size(1)))
                    #self.project_parameters(vmin, vmax)
                    self.widths.data = self.widths.data.clamp_(vmin, vmax)
                    self.locations.data = self.locations.data - self.locations.data.floor()
    


        return self.widths, self.locations, self.U, self.Vt

    def dense_matrix(self, sigma=None):
        wu, wv = self.get_quantized_param(self.widths[0,:], self.num_rows), self.get_quantized_param(self.widths[1,:], self.num_cols)
        lu, lv = self.get_quantized_param(self.locations[0,:], self.num_rows), self.get_quantized_param(self.locations[1,:], self.num_cols)
        mask_u = self.U_mask(wu, lu, sigma=sigma)
        mask_v = self.V_mask(wv, lv, sigma=sigma)
        mask_u[mask_u<1e-3] = 0.0
        mask_v[mask_v<1e-3] = 0.0
        mask_u = torch.clamp(mask_u, 0.0, 1.0)
        mask_v = torch.clamp(mask_v, 0.0, 1.0)

        #mask_u[mask_u>0.5] = 1.0
        #mask_v[mask_v>0.5] = 1.0
        #mask_u[mask_u<=0.5] = 0.0
        #mask_v[mask_v<=0.5] = 0.0
        return self.get_dense_matrix(mask_u, mask_v, self.U, self.Vt)



    def forward(self, input):
        mat = self.dense_matrix(sigma=self.sigma)
        out = F.linear(input, masked_w, self.bias.to(dtype=input.dtype))
        return out




def find_olb(M, budget, thres_row_list, thres_col_list, stddev=10.0, rank_per_component=1, 
             weight_lr=5e-3, structure_lr_base=10.0, scale_structure_lr=True,
             sched_params={'start_factor':1.0, 'end_factor': 0.01},
             niter=1000,
             one_by_one=False,
             verbose=False,
             opnorm_target=None,
             find_mask=True,
             train_widths=True,
             train_locations=True,
             widths=None,
             locations=None,
             use_sigma=False,
             ):
    start_time = time.time()
    assert rank_per_component==1
    with torch.no_grad():
        M_svdvals = torch.linalg.svdvals(M.cpu())
        if opnorm_target is not None:
            rank = torch.sum((M_svdvals / M_svdvals[0] > opnorm_target).float()).int().item()
        M_opnorm = M_svdvals[0]
        rank = budget // (M.size(0)+M.size(1))
        if rank >= len(M_svdvals)-1:
            ref_op_error = torch.tensor(0.0)
        else:
            ref_op_error = M_svdvals[rank+1] / M_opnorm
    structure_lr = structure_lr_base
    if scale_structure_lr:
        structure_lr = structure_lr / min(*M.size()) 
    best_op_error = 100000.0
    best_params = ()
    for thres_row in thres_row_list:
        for thres_col in thres_col_list:
            olb = OLB(size=M.size(), 
                      rank_per_component=rank_per_component, 
                      num_components=min(*M.size())//rank_per_component,
                      use_direct_mask=False)

            olb.set_weights(M)
            olb = olb.to(M.device)
            if widths is not None:
                olb.widths.data = widths.to(M.device)
            if locations is not None:
                olb.locations.data = locations.to(M.device)

            if find_mask:
                wu, wv, lu, lv = olb.find_mask(M, budget=budget,
                                               thres_row=thres_row,
                                               thres_col=thres_col,
                                               sigma=stddev,
                                               verbose=False)

            with torch.no_grad():
                num_flops_before = olb.get_num_params(exact=True).int().item()

            if use_sigma:
                sigma_params = {'start_sigma': 1.0, 'end_sigma': 1.0, 'p': 8.0}
                sigma = 1.0
            else:
                sigma_params = None
                sigma = None
            w, l, U, Vt = olb.find_mats(M, budget=budget,
                                        niter=niter,
                                        lr=weight_lr, structure_lr=structure_lr,
                                        train_widths=train_widths, train_locations=train_locations,
                                        sched_params=sched_params,
                                        one_by_one=one_by_one,
                                        sigma_params=sigma_params,
                                        betas=(0.0,0.999),
                                        verbose=verbose)
            M_pred = olb.dense_matrix(sigma=sigma)
            with torch.no_grad():
                num_flops = olb.get_num_params(exact=False).int().item()
                nrmse = torch.norm(M - M_pred) / torch.norm(M)
                op_norm = torch.linalg.svdvals((M_pred - M).cpu())[0] / M_opnorm
                #print("Thres row: {}, Thres col: {}, opnorm: {:.5f}, opnorm_lr: {:.5f}, NRMSE: {:.5f}, Flops: {}, Budget: {}".format(thres_row, thres_col, op_norm.item(), ref_op_error.item(), nrmse.item(), num_flops, budget * (M.size(0)+M.size(1))))
                if op_norm.item() < best_op_error:
                    best_op_error = op_norm.item() 
                    best_nrmse = nrmse.item()
                    best_params = (w, l, U, Vt)
                    best_flops = num_flops
                    best_flops_before = num_flops_before
                    best_tr = thres_row
                    best_tc = thres_col
    #print("Best thres: ({}, {}), opnorm: {:.5f}, opnorm_lr: {:.5f}, NRMSE: {:.5f}, Flops: {}, Flops_before: {}, Budget: {}, Time taken: {} sec".format(best_tr, best_tc, best_op_error, ref_op_error.item(), best_nrmse, best_flops, best_flops_before, budget*(M.size(0)+M.size(1)), time.time() - start_time))
    log.info("Best thres: ({}, {}), opnorm: {:.5f}, opnorm_lr: {:.5f}, NRMSE: {:.5f}, Flops: {}, Flops_before: {}, Budget: {}, Time taken: {:.1f} sec".format(best_tr, best_tc, best_op_error, ref_op_error.item(), best_nrmse, best_flops, best_flops_before, budget, time.time() - start_time))
    return best_params




                      
             

def low_rank_project(M, rank):
    """Supports batches of matrices as well.
    """
    U, S, Vt = torch.linalg.svd(M)
    S_sqrt = S[..., :rank].sqrt()
    U = U[..., :rank] * rearrange(S_sqrt, '... rank -> ... 1 rank')
    Vt = rearrange(S_sqrt, '... rank -> ... rank 1') * Vt[..., :rank, :]
    return U, Vt



def test():
    r = 2
    W = torch.randn(10,8)
    olb = OLB(W.size(), 1, min(*W.size()), use_direct_mask=False, width_init=1.0, alpha=2.0, beta=2.0)
    olb.find_mask(W, r*(W.size(0)+W.size(1)), thres_row=0.4, thres_col=0.4, use_longest_mask=True)
    olb.find_mats_svd(W, olb.widths, olb.locations)

if __name__=='__main__':
    test()
