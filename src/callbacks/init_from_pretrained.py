from lightning.pytorch import Callback
from lightning.pytorch.utilities import rank_zero_only
import hydra

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange

from src.models.modules.olb import find_olb
from src.models.layers.gblr import GaudiGBLR
from src.models.layers.fastlinear import LowRank
from src.models.layers.monarch_linear import MonarchLinear, get_nblocks
from src.models.layers.fastlinear import SparseLRLinear

from src.utils import utils
from copy import deepcopy
import cvxpy as cp
from cvxpy.error import SolverError
log = utils.get_logger(__name__)


class ReplaceBertLayers(Callback):
    def __init__(self, budget_in_ratio,
                 layer_type, 
                 weight_lr=0.005, structure_lr_base=20.0,
                 thres_row_list=[0.98], thres_col_list=[0.98],
                 niter=1000,
                 skip_loading=False,
                 decompose=True,
                 load_from_decomposed=None,
                 gaudi_params=None):
        self.layer_type = layer_type
        self.budget_in_ratio = budget_in_ratio
        self.weight_lr = weight_lr
        self.structure_lr_base = structure_lr_base
        self.thres_row_list = thres_row_list
        self.thres_col_list = thres_col_list
        self.niter = niter
        self.skip_loading = skip_loading
        self.gaudi_params = gaudi_params
        self.load_from_decomposed = load_from_decomposed
        self.decompose = decompose

    def setup(self, trainer, pl_module, stage):
        if stage != "fit" or self.skip_loading:
            return

        new_model = deepcopy(pl_module.model)
        for mn, m in pl_module.model.named_modules():
            if isinstance(m, nn.Linear) and 'bert.encoder.layer' in mn:
                log.info(mn)
                M = m.weight
                device = M.device
                budget_in_ratio = self.budget_in_ratio
                if self.layer_type == 'lr':
                    rank = int(budget_in_ratio * m.in_features * m.out_features / (m.in_features + m.out_features))
                    log.info("rank: {}".format(rank))
                    new_layer = LowRank(in_features=m.in_features,
                                        out_features=m.out_features,
                                        rank=rank)
                    new_layer.set_weights_from_projection(M)
                    if m.bias is not None:
                        new_layer.bias.data = m.bias.data

                elif self.layer_type == 'monarch':
                    nblocks = get_nblocks(m.in_features, m.out_features, budget_in_ratio)
                    new_layer = MonarchLinear(m.in_features, m.out_features, nblocks)
                    if self.decompose:
                        new_layer.bias.data = torch.zeros_like(new_layer.bias)
                        M = M.cuda()
                        new_layer = new_layer.cuda()
                        I = torch.eye(new_layer.in_features_extended).cuda()
                        opt = torch.optim.Adam([new_layer.blkdiag1, new_layer.blkdiag2], lr=1e-3)
                        loss = torch.tensor(0.0)
                        for t in range(1000):
                            opt.zero_grad()
                            loss = torch.mean((M.T.detach() - new_layer(I))**2)
                            loss.backward()
                            opt.step()
                        log.info("Loss: {}".format(loss.item()))
                        new_layer = new_layer.to(device)
                    if m.bias is not None:
                        new_layer.bias.data = m.bias.data

                elif self.layer_type == 'gaudi':
                    new_layer = GaudiGBLR(m.in_features, m.out_features, **self.gaudi_params).to(device)
                    if self.decompose:
                        budget = int(6/7*budget_in_ratio * min(m.in_features, m.out_features) * (m.in_features + m.out_features))
                        M = M.cuda()
                        w,l,U,Vt = find_olb(M=M, budget=budget,
                                            thres_row_list=self.thres_row_list,
                                            thres_col_list=self.thres_col_list,
                                            weight_lr=self.weight_lr,
                                            structure_lr_base=self.structure_lr_base,
                                            verbose=False,
                                            niter=self.niter,
                                            sched_params={'start_factor': 1.0, 'end_factor': 0.01},
                                            )
                        w = w.flip(0)
                        l = l.flip(0)
                        new_layer.lr_weight1.data = Vt.to(device).data
                        new_layer.lr_weight2.data = U.to(device).data
                        new_layer.widths.data = w.to(device).data
                        new_layer.locations.data = l.to(device).data
                        if m.bias is not None:
                            new_layer.bias.data = m.bias.data


                else:
                    raise NotImplementedError()               

                with torch.no_grad():
                    parent_name = ".".join(mn.split(".")[:-1])
                    child_name = mn.split(".")[-1]
                        
                    for new_mn, new_m in new_model.named_modules():
                        if new_mn == parent_name:
                            new_m.add_module(child_name, deepcopy(new_layer))

        pl_module.model = new_model
        if self.load_from_decomposed is not None:
            sd = torch.load(self.load_from_decomposed, map_location=device)
            del sd['classifier.weight']
            del sd['classifier.bias']
            missing, unexpected = pl_module.model.load_state_dict(sd, strict=False)
            log.info("Parameters restored from {}".format(self.load_from_decomposed))
            if missing is not None:
                log.info("Missing Keys: {}".format(missing))
            if unexpected is not None:
                log.info("Unexpected Keys: {}".format(unexpected))


     


class InitFromPretrained(Callback):

    def __init__(self, ckpt, budget, 
                 weight_lr=0.005, structure_lr_base=20.0,
                 thres_row_list=[0.98], thres_col_list=[0.98],
                 niter=1000,
                 one_by_one=False,
                 opnorm_target=None,
                 verbose=False,
                 use_sigma=False,
                 thres=1.0,
                 skip_loading=False, target_class="src.tasks.seq.SequenceModel",
                 ):
        super().__init__()
        self.ckpt = ckpt
        self.budget = budget
        self.weight_lr = weight_lr
        self.structure_lr_base = structure_lr_base
        self.thres_row_list = thres_row_list
        self.thres_col_list = thres_col_list
        self.niter = niter
        self.one_by_one = one_by_one
        self.opnorm_target = opnorm_target
        self.verbose = verbose
        self.skip_loading = skip_loading
        self.cls = hydra.utils.get_class(target_class)
        self.thres = thres
        self.use_sigma = use_sigma

   
    def setup(self, trainer, pl_module, stage):
        if stage != "fit" or self.skip_loading:
            return
        model = pl_module
        log.info("Loading the pretrained model...")
        if self.ckpt is None:
            pretrained = deepcopy(model)
        else:
            pretrained = model.load_from_checkpoint(checkpoint_path=self.ckpt, strict=True)
        with torch.no_grad():
            for mn, m in pretrained.model.named_modules():
                if 'attn' in mn and hasattr(m, 'packed_linear') and m.packed_linear:
                    qw, kw, vw = m.qkv.weight.chunk(3)
                    q = torch.nn.Linear(qw.size(1), qw.size(0), bias=m.qkv.bias is not None)
                    q.weight.data = qw.data
                    k = torch.nn.Linear(kw.size(1), qw.size(0), bias=m.qkv.bias is not None)
                    k.weight.data = kw.data
                    v = torch.nn.Linear(vw.size(1), qw.size(0), bias=m.qkv.bias is not None)
                    v.weight.data = vw.data
                    if m.qkv.bias is not None:
                        qb, kb, vb = m.qkv.bias.chunk(3)
                        q.bias.data = qb.data
                        k.bias.data = kb.data
                        v.bias.data = vb.data

                    m.packed_linear = False
                    del m.qkv
                    m.add_module("q_proj", q)
                    m.add_module("k_proj", k)
                    m.add_module("v_proj", v)

        pretrained_params = dict(pretrained.model.named_parameters())
        pt_dict = dict(pretrained.model.named_modules())

        for mn, m in model.model.named_modules():
            if isinstance(m, (nn.Linear, nn.GELU, nn.LayerNorm, nn.Conv2d)):
                for pn, p in m.named_parameters():
                    fpn = "{}.{}".format(mn,pn) if mn else pn
                    try:
                        p.data = pretrained_params[fpn].data
                    except KeyError:
                        log.info("{}.{} was not processed.".format(mn, pn))
            elif isinstance(m, LowRank):
                try:
                    M = pt_dict[mn].weight.data
                    m.set_weights_from_projection(M)
                    m.bias.data = pt_dict[mn].bias.data
                    with torch.no_grad():
                        parent_name = ".".join(mn.split(".")[:-1])
                        child_name = mn.split(".")[-1]
                        for pmn, pm in pretrained.model.named_modules():
                            if pmn == parent_name:
                                pm.add_module(child_name, deepcopy(m))
                except KeyError:
                    log.info("{} was not processed.".format(mn)) 

            elif isinstance(m, MonarchLinear):
                log.info(mn)
                M = pt_dict[mn].weight.data
                m.bias.data = torch.zeros_like(m.bias.data)
                device = m.blkdiag1.device
                M = M.cuda()
                m = m.cuda()
                I = torch.eye(m.in_features_extended).cuda()
                opt = torch.optim.Adam(m.parameters(), lr=self.weight_lr)
                #import pdb; pdb.set_trace()
                loss = torch.tensor(0.0)
                for t in range(self.niter):
                    opt.zero_grad()
                    loss = torch.mean((M.T.detach() - m(I))**2)
                    loss.backward()
                    opt.step()
                log.info("Loss: {}".format(loss.item()))
                m = m.to(device)
                m.bias.data = pt_dict[mn].bias.data


                with torch.no_grad():
                    parent_name = ".".join(mn.split(".")[:-1])
                    child_name = mn.split(".")[-1]
                    for pmn, pm in pretrained.model.named_modules():
                        if pmn == parent_name:
                            pm.add_module(child_name, deepcopy(m))

            elif isinstance(m, SparseLRLinear):
                log.info(mn)
                M = pt_dict[mn].weight.data
                m.bias.data = torch.zeros_like(m.bias.data)
                device = m.bias.device
                M = M.cuda()
                m = m.cuda()
                I = torch.eye(m.in_features).cuda()
                opt = torch.optim.Adam(m.parameters(), lr=self.weight_lr)
                loss = torch.tensor(0.0)
                for t in range(self.niter):
                    opt.zero_grad()
                    loss = torch.mean((M.T.detach() - m(I))**2)
                    loss.backward()
                    opt.step()
                log.info("Loss: {}".format(loss.item()))
                m = m.to(device)
                m.bias.data = pt_dict[mn].bias.data


                with torch.no_grad():
                    parent_name = ".".join(mn.split(".")[:-1])
                    child_name = mn.split(".")[-1]
                    for pmn, pm in pretrained.model.named_modules():
                        if pmn == parent_name:
                            pm.add_module(child_name, deepcopy(m))


            elif isinstance(m, GaudiGBLR):
                log.info(mn) 
                if self.ckpt is None:
                    with torch.no_grad():
                        w1 = rearrange(m.lr_weight1, 'nc rpc in_f -> (nc rpc) in_f')
                        w2 = rearrange(m.lr_weight2, 'nc out_f rpc -> out_f (nc rpc)')
                        M = w2 @ w1
                else:
                    M = pt_dict[mn].weight.data
                m.bias.data = pt_dict[mn].bias.data

                budget_in_ratio = self.budget
                if self.ckpt is None:
                    budget = int(M.size(0)*M.size(1)*budget_in_ratio / (M.size(0)+M.size(1))) 
                    structure_lr_base = self.structure_lr_base 
                    widths = m.widths.data.clone().detach()
                    locations = m.locations.data.clone().detach()
                    find_mask = False

                else:
                    budget = int(budget_in_ratio * min(*M.size()))
                    structure_lr_base = self.structure_lr_base
                    widths = locations = None
                    find_mask = False
                M = M.cuda()
                w,l,U,Vt = find_olb(M=M, budget=budget*(M.size(0)+M.size(1)),
                                    thres_row_list=self.thres_row_list,
                                    thres_col_list=self.thres_col_list,
                                    weight_lr=self.weight_lr,
                                    structure_lr_base=structure_lr_base,
                                    verbose=self.verbose,
                                    niter=self.niter,
                                    one_by_one=self.one_by_one,
                                    sched_params={'start_factor': 1.0, 'end_factor': 0.01},
                                    opnorm_target=self.opnorm_target,
                                    use_sigma=self.use_sigma,
                                    find_mask=find_mask,
                                    widths=widths,
                                    locations=locations,
                                    )
                w = w.flip(0)
                l = l.flip(0)
                m.lr_weight1.data = Vt.cpu().data
                m.lr_weight2.data = U.cpu().data
                m.widths.data = w.cpu().data
                m.locations.data = l.cpu().data

                with torch.no_grad():
                    parent_name = ".".join(mn.split(".")[:-1])
                    child_name = mn.split(".")[-1]
                    for pmn, pm in pretrained.model.named_modules():
                        if pmn == parent_name:
                            pm.add_module(child_name, deepcopy(m))

        pl_module.model = pretrained.model






