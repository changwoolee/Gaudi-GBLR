# Adapted from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
# and https://github.com/yitu-opensource/T2T-ViT/blob/main/models/transformer_block.py
import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

import hydra

from torchvision.ops import StochasticDepth

from src.models.modules.seq_common import Mlp

import numpy as np

class ArccosKernel(nn.Module):
    def __init__(self, n, *args, **kwargs):
        super().__init__()
        self.n = n

    def forward(self, Q, KV):
        Q_norm = torch.norm(Q, p=2, dim=-1, keepdim=True) 
        KV_norm = torch.norm(KV, p=2, dim=-2, keepdim=True)
        #QKV_norm = Q_norm * KV_norm

        QKV = torch.bmm(Q / Q_norm, KV / KV_norm)
        theta = torch.arccos(QKV)

        QKV = QKV * Q_norm * KV_norm

        pi_inv = 1 / torch.pi

        if self.n == 1:
            # Arccos kernel with n=1
            out = QKV * ( 1 - pi_inv * theta) + pi_inv * Q_norm * KV_norm * torch.sin(theta) 
        elif self.n == 0:
            # Product of dot product kernel and arccos kernel with n=0.
            out = QKV * ( 1 - pi_inv * theta)
        else:
            raise NotImplementedError()
        return out


class GaussianKernel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.log_sigmas = nn.Parameter(torch.zeros((1,)))

    def forward(self, Q, KV):
        Q_norm = torch.norm(Q, p=2, dim=-1, keepdim=True) 
        KV_norm = torch.norm(KV, p=2, dim=-2, keepdim=True)
        #QKV_norm = Q_norm * KV_norm

        QKV = torch.bmm(Q, KV)

        out = torch.exp(- 0.5 / torch.exp(2*self.log_sigmas) * (Q_norm**2 + KV_norm**2 - 2 * QKV))
        return out


class HLapKernel(nn.Module):
    """
    Homogeneous Laplace Kernel
    """
    def __init__(self):
        super().__init__()
        self.log_sigmas = nn.Parameter(torch.zeros(1))

    def forward(self, Q, KV):
        Q_norm = torch.norm(Q, p=2, dim=-1, keepdim=True) 
        KV_norm = torch.norm(KV, p=2, dim=-2, keepdim=True)
        QKV_norm = Q_norm * KV_norm

        QKV = torch.bmm(Q, KV)

        r = torch.sqrt(1 - QKV/QKV_norm) * torch.exp(self.log_sigmas).view(1,1,-1)
        return QKV_norm * torch.exp(-r) # Homogeneous Laplacian Kernel


class PolynomialKernel(nn.Module):
    def __init__(self, degree):
        super().__init__()
        self.degree = degree
        self.bias = nn.Parameter(torch.ones(degree) / self.degree)
        self.coeffs = nn.Parameter(torch.ones(degree) / self.degree)

    def forward(self, X, Y):
        X_norm = torch.norm(X, p=2, dim=-1, keepdim=True)
        Y_norm = torch.norm(Y, p=2, dim=-2, keepdim=True)

        XY = torch.bmm(X / X_norm, Y / Y_norm) 
        #out = torch.zeros_like(XY)
        out = XY * self.coeffs[0] + self.bias[0]
        for d in range(1, self.degree):
            #out += XY ** (d+1)  * self.coeffs[d] 
            out = out * XY * self.coeffs[d] + self.bias[d]
        return out * X_norm * Y_norm 


        




class AttentionSimple(nn.Module):
    """This attention class makes several simplifying assumptions (commonly satisfied in vision
       applications):
    1. q = k = v
    2. No masks: no attention mask, no key padding mask
    3. Embed dimension = Input dimension, i.e. projection matrices are square.
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 packed_linear=True, linear_cfg=None, enforce_packed_linear=False, 
                 kernelized_attention=True, sparsify=False):
        """packed_linear: whether to pack all 3 q_proj, k_proj, v_proj into 2 matrix.
        This option is to be compatible with T2T-ViT pretrained weights, where there's only one
        projection weight matrix.
        """
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        if not enforce_packed_linear and linear_cfg is not None:
            packed_linear = False
        self.packed_linear = packed_linear
        if packed_linear:
            #self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
            if linear_cfg is None or "None":
                linear_cfg = {'_target_': 'torch.nn.Linear'}
            self.qkv = hydra.utils.instantiate(linear_cfg, dim, dim*3, bias=qkv_bias,
                                                  _recursive_=False)
        else:
            if linear_cfg is None or not sparsify:
                linear_cfg = {'_target_': 'torch.nn.Linear'}
            self.q_proj = hydra.utils.instantiate(linear_cfg, dim, dim, bias=qkv_bias,
                                                  _recursive_=False)
            self.k_proj = hydra.utils.instantiate(linear_cfg, dim, dim, bias=qkv_bias,
                                                  _recursive_=False)
            self.v_proj = hydra.utils.instantiate(linear_cfg, dim, dim, bias=qkv_bias,
                                                  _recursive_=False)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.kernelized_attention = kernelized_attention
        if kernelized_attention:
            self.kernel = ArccosKernel(n=0)
            #self.kernel = PolynomialKernel(degree=5)

    def forward(self, x):
        B, N, C = x.shape
        if self.packed_linear:
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)
        else:
            q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
            q, k, v = [rearrange(x, 'b n (h d) -> b h n d', h=self.num_heads) for x in (q, k, v)]

        # attn = (q @ k.transpose(-2, -1) * self.scale)
        # Use `torch.baddbmm` (a bit more efficient w/ alpha param for scaling -- from Megatron-LM)
        bsz, num_heads, q_seq_len, dk = q.size()
        _, _, k_seq_len, _ = k.size()
        _, _, _, dv = v.size()
        q = rearrange(q, 'b h t d -> (b h) t d')
        k = rearrange(k, 'b h s d -> (b h) d s')

        # Fast Kernelized Attention
        if self.kernelized_attention:  
            _,_,_, dv 
            v = rearrange(v, 'b h n d -> (b h) n d')
            #kv = torch.empty(bsz * num_heads, dk, dv, dtype=q.dtype, device=q.device)
            #kv = torch.baddbmm(kv, k, v, beta=0, alpha=self.scale)
            kv = torch.bmm(k, v) * self.scale

            qkv = torch.bmm(F.gelu(q), F.gelu(kv))#self.kernel(q, kv)
            x = rearrange(qkv, '(b h) t s -> b t (h s)', h = self.num_heads)
        # Original Softmax Attention
        else:
            # Preallocate attn_weights for `baddbmm`
            attn = torch.empty(bsz * num_heads, q_seq_len, k_seq_len, dtype=q.dtype, device=q.device)
            attn = rearrange(torch.baddbmm(attn, q, k, beta=0, alpha=self.scale),
                             '(b h) t s -> b h t s', h = self.num_heads)

            attn = F.softmax(attn, dim=-1, dtype=v.dtype)
            attn = self.attn_drop(attn)

            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 attnlinear_cfg=None, mlp_cfg=None,
                 enforce_packed_linear=False,
                 packed_linear=True,
                 kernelized_attention=False,
                 sparsify=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = AttentionSimple(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop,
            linear_cfg=attnlinear_cfg,
            enforce_packed_linear=enforce_packed_linear,
            packed_linear=packed_linear,
            kernelized_attention=kernelized_attention,
            sparsify=sparsify)
        self.drop_path = StochasticDepth(drop_path, mode='row')
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        if mlp_cfg is None or not sparsify:
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        else:
            self.mlp = hydra.utils.instantiate(mlp_cfg, in_features=dim, hidden_features=mlp_hidden_dim,
                                               act_layer=act_layer, drop=drop, _recursive_=False)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
