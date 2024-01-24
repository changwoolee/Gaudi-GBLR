import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.beta import Beta

from src.models.layers.fouriermask import FourierMaskLR, FourierMaskHard

from einops import rearrange

class MatLoss(nn.Module):

    def __init__(self, lambd=1e-4, loss_type='norm'):
        super().__init__()
        self.lambd = lambd
        self.loss_type = loss_type
        assert self.loss_type in ['mse', 'norm', 'normsq']


    def forward(self, pl_module):
        mask_loss = 0.0

        if self.lambd == 0:
            return mask_loss

        count = 0
        for mn, m in pl_module.model.named_modules():
            if isinstance(m, FourierMaskLR):
                masked_w1 = rearrange(m.lr_weight1.detach() * m.buffer['m1'], 'nc rpc in_f -> (nc rpc) in_f')
                masked_w2 = rearrange(m.lr_weight2.detach() * m.buffer['m2'], 'nc out_f rpc -> out_f (nc rpc)')

                masked_w = masked_w2 @ masked_w1

                with torch.no_grad():
                    w1 = rearrange(m.lr_weight1, 'nc rpc in_f -> (nc rpc) in_f')
                    w2 = rearrange(m.lr_weight2, 'nc out_f rpc -> out_f (nc rpc)')
                    w_orig = w2 @ w1
                if self.loss_type == 'mse':
                    mask_loss += F.mse_loss(masked_w, w_orig.detach())
                elif self.loss_type == 'norm':
                    mask_loss += torch.norm(masked_w - w_orig.detach())
                elif self.loss_type == 'normsq':
                    mask_loss += torch.norm(masked_w - w_orig.detach())**2

                count += 1
        mask_loss /= count
        pl_module.log("mask loss", mask_loss, rank_zero_only=True, prog_bar=True, sync_dist=True)
        return self.lambd * mask_loss










class BetaRegularizer(nn.Module):
    def __init__(self, beta1, beta0, lambd, eps=1e-6, **kwargs):
        super().__init__()
        #self.dist = Beta(beta1, beta0)
        self.beta1 = beta1
        self.beta0 = beta0
        self.lambd = lambd
        self.eps = eps

    def forward(self, model):
        loss = 0.
        for mn, m in model.named_modules():
            if isinstance(m, FourierMaskLR):
                #loss += self.dist.log_prob(p)
                w1, w2 = m.get_width(0), m.get_width(1)
                p = torch.stack([w1, w2])
                nll = - (self.beta1 - 1.0) * torch.log(p + self.eps) - (self.beta0 - 1.0) * torch.log(1-p + self.eps)
                loss += nll.sum()
        return self.lambd * loss
                

class AverageRegularizer(nn.Module):
    def __init__(self, complexity, lambd, p=2, exclude_qkv=False, **kwargs):
        super().__init__()
        self.complexity = complexity
        self.lambd = lambd
        self.p = p
        self.exclude_qkv = exclude_qkv
        
    def forward(self, model):
        numel = 0
        widths = 0.
        #mat_loss = 0.
        for mn, m in model.named_modules():
            if isinstance(m, FourierMaskLR) and not isinstance(m, FourierMaskHard):
                if self.exclude_qkv and 'attn' in mn:
                    continue
                w1, w2 = m.get_width(0), m.get_width(1)
                p = torch.stack([w1, w2])
                widths += p.sum() * m.total_rank_ratio
                numel += p.numel()

        loss = torch.abs(widths / numel - self.complexity) ** self.p

        return self.lambd * loss 

class AreaRegularizer(AverageRegularizer):
    def forward(self, model):
        if self.lambd == 0:
            return 0.0
        areas = 0.
        module_count = 0
        for mn, m in model.named_modules():
            if isinstance(m, FourierMaskLR):
                w1, w2 = m.get_width(0), m.get_width(1)
                #w1 -= m.min_widths[0]
                #w2 -= m.min_widths[1]
                area = w1 * w2
                areas += area.mean()
                module_count += 1

        loss = torch.abs(areas - module_count * self.complexity**2) ** self.p

        return self.lambd * loss


class EntropyRegularizer(AverageRegularizer):
    pass
