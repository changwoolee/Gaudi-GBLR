# Adapted from https://github.com/rwightman/pytorch-image-models/blob/master/benchmark.py
import torch
from src.models.layers.fouriermask import FourierMaskLR, FourierMaskConv2d, FourierMaskConv2dIntegrated

try:
    from deepspeed.profiling.flops_profiler import get_model_profile
    has_deepspeed_profiling = True
except ImportError as e:
    has_deepspeed_profiling = False

try:
    from fvcore.nn import FlopCountAnalysis, flop_count, flop_count_str, flop_count_table
    from fvcore.nn import ActivationCountAnalysis
    has_fvcore_profiling = True
except ImportError as e:
    FlopCountAnalysis = None
    ActivationCountAnalysis = None
    has_fvcore_profiling = False


def profile_deepspeed(model, input_size=(3, 224, 224), batch_size=1, detailed=False):
    macs, _ = get_model_profile(
        model=model,
        input_res=(batch_size,) + input_size,  # input shape or input to the input_constructor
        input_constructor=None,  # if specified, a constructor taking input_res is used as input to the model
        print_profile=detailed,  # prints the model graph with the measured profile attached to each module
        detailed=detailed,  # print the detailed profile
        warm_up=10,  # the number of warm-ups before measuring the time of each module
        as_string=False,  # print raw numbers (e.g. 1000) or as human-readable strings (e.g. 1k)
        output_file=None,  # path to the output file. If None, the profiler prints to stdout.
        ignore_modules=None)  # the list of modules to ignore in the profiling
    return macs, 0  # no activation count in DS


def profile_fvcore(model, input_size=(3, 224, 224), input_dtype=torch.float32, max_depth=4,
                   batch_size=1, detailed=False, force_cpu=False):
    if force_cpu:
        model = model.to('cpu')
    device, dtype = next(model.parameters()).device, next(model.parameters()).dtype
    example_input = torch.ones((batch_size,) + input_size, device=device, dtype=input_dtype)
    fca = FlopCountAnalysis(model, example_input)
    aca = ActivationCountAnalysis(model, example_input)
    if detailed:
        print(flop_count_table(fca, max_depth=max_depth))
    return fca, fca.total(), aca, aca.total()


def profile_fvcore_sinc_gaussian(model, input_size=(3, 224, 224), input_dtype=torch.float32, 
                                 max_depth=4, batch_size=1, detailed=False, force_cpu=False,
                                 baseline_complexity=None,
                                 ):

    if force_cpu:
        model = model.to('cpu')
    device, dtype = next(model.parameters()).device, next(model.parameters()).dtype
    example_input = torch.ones((batch_size,) + input_size, device=device, dtype=input_dtype)
    fca = FlopCountAnalysis(model, example_input)
    fca.unsupported_ops_warnings(False)
    fca.uncalled_modules_warnings(False)
    aca = ActivationCountAnalysis(model, example_input)
    aca.unsupported_ops_warnings(False)
    aca.uncalled_modules_warnings(False)
    if baseline_complexity is None:
        baseline_complexity = fca.total()

    num_features = model.model.num_features
    try:
        num_tokens = model.model.stem.num_patches
    except AttributeError:
        num_tokens = model.model.patch_embed.num_patches


    fca_dict = fca.by_module()

    flops_count = 0
    for mn, m in model.named_modules():
        if isinstance(m, FourierMaskLR):
            if m.in_features == num_features or m.out_features == num_features:
                flop = num_tokens * m.get_num_params().item()
            elif m.in_features == num_tokens or m.out_features == num_tokens:
                flop = num_features * m.get_num_params().item()
            fca_dict[mn] = flop
            if 'fc2' in mn:
                fc1_mn = mn[:-4] + '.fc1'
                if fca_dict[fc1_mn] < 1e-9:
                    fca_dict[mn] = 0
                elif flop < 1e-9:
                    fca_dict[fc1_mn] = 0
            if 'v_proj' in mn and fca_dict[mn] < 1e-9:
                q_mn = mn[:-6] + 'q_proj'
                k_mn = mn[:-6] + 'k_proj'
                fca_dict[q_mn] = 0
                fca_dict[k_mn] = 0




                
    leaf_nodes = dict()
    for mn, m in model.named_modules():
        if len(list(m.children())) == 0:
            flops_count += fca_dict[mn]
            leaf_nodes[mn] = fca_dict[mn]


    if detailed:
        print("Sinc Gaussian FLOPS: {}, {}%".format(flops_count, flops_count / baseline_complexity * 100))
    #print(leaf_nodes)
    return fca, flops_count, aca, aca.total(), flops_count / baseline_complexity




def profile_fvcore_gaudi_conv(model, input_size=(3, 224, 224), input_dtype=torch.float32, 
                                 max_depth=4, batch_size=1, detailed=False, force_cpu=False,
                                 baseline_complexity=None,
                                 ):

    if force_cpu:
        model = model.to('cpu')
    device, dtype = next(model.parameters()).device, next(model.parameters()).dtype
    example_input = torch.ones((batch_size,) + input_size, device=device, dtype=input_dtype)
    fca = FlopCountAnalysis(model, example_input)
    fca.unsupported_ops_warnings(False)
    fca.uncalled_modules_warnings(False)
    aca = ActivationCountAnalysis(model, example_input)
    aca.unsupported_ops_warnings(False)
    aca.uncalled_modules_warnings(False)
    if baseline_complexity is None:
        baseline_complexity = fca.total()

    fca_dict = fca.by_module()

    flops_count = 0
    for mn, m in model.named_modules():
        if isinstance(m, (FourierMaskConv2d)):
            #flop = fca_dict[mn]
            #flop = flop - m.out_channels * m.in_channels ** 2 * m.kernel_size[0] * m.kernel_size[1] 
            #output_size = flop // (m.out_channels * m.in_channels * m.kernel_size[0] * m.kernel_size[1])
            #fca_dict[mn] = output_size * m.gaudi_module.get_num_params()
            #kernel_size = m.conv2d_layer.kernel_size
            flops = 0
            num_modules = m.num_modules if not m.unified_mask else 1
            for i in range(m.num_modules):
                flops += m.gaudi_modules[i].get_num_params()
            fca_dict[mn] = m.input_shape[0] * m.input_shape[1] * flops
        elif isinstance(m, (FourierMaskConv2dIntegrated)):
            flops = m.gaudi_module.get_num_params()
            fca_dict[mn] = m.input_shape[0] * m.input_shape[1] * flops


                
    leaf_nodes = dict()
    for mn, m in model.named_modules():
        if (len(list(m.children())) == 0 and not isinstance(m, FourierMaskLR)) or isinstance(m, (FourierMaskConv2d, FourierMaskConv2dIntegrated)):
            flops_count += fca_dict[mn]
            leaf_nodes[mn] = fca_dict[mn]


    if detailed:
        print("Sinc Gaussian FLOPS: {}, {}%".format(flops_count, flops_count / baseline_complexity * 100))
    #print(leaf_nodes)
    return fca, flops_count, aca, aca.total(), flops_count / baseline_complexity

