import copy
from compress.quantization.calibrate import calibrate
from compress.quantization.ptq import (
    QuantizedLinear,
    QuantizedConv2d,
    KMeansQuantizedConv2d,
    KMeansQuantizedLinear,
)
from compress.quantization.qat import (
    QATConv2d,
    QATLinear,
    LSQConv2d,
    LSQLinear,
    snap_loss_model_activations,
    snap_loss_model_params,
    SnapRegularizer,
    PACTReLU,
    get_regularizer_for_pact,
    MutualInfoRegularizer,
    AutoBitAllocationConv2d,
    AutoBitAllocationLinear,
    FusedQATConv2dBatchNorm2d,
    FusedLSQConv2dBatchNorm2d,
)

import torch.nn.functional as F

from compress.quantization.common import (
    IntAffineQuantizationSpec,
    IntAffineQuantizationInfo,
    IntAffineQuantizationMode,
    ste_floor,
    PerTensor,
    ConvWeightsPerOutChannel,
    LinearPerColumn,
    LinearPerRow,
)
from compress.common import (
    gather_submodules,
    default_should_do,
    cls_passlist_should_do,
    combine_should_do,
    keys_passlist_should_do,
)

from compress.layer_fusion import fuse_batch_norm_inference, fuse_conv_bn
from torch import nn
from tqdm import tqdm
import torch
import gc
from typing import Dict, Literal
import functools


def merge_dicts(dict1: Dict, dict2: Dict) -> Dict:
    """
    Merges two dictionaries. If a key is present in both dictionaries, the value from dict2 is used.
    """
    merged_dict = dict1.copy()
    merged_dict.update(dict2)
    return merged_dict


def get_quant_dict(
    model,
    layer_type: str,
    input_spec: IntAffineQuantizationSpec,
    weight_spec: IntAffineQuantizationSpec,
):
    assert isinstance(layer_type, str), "layer_type must be a string"
    assert isinstance(
        input_spec, IntAffineQuantizationSpec
    ), "input_specs must be a IntAffineQuantizationSpec"
    assert isinstance(
        weight_spec, IntAffineQuantizationSpec
    ), "weight_specs must be a IntAffineQuantizationSpec"

    # gather all layers of that type
    cl = (
        [nn.Linear, nn.LazyLinear]
        if layer_type == "linear"
        else [nn.Conv2d, nn.LazyConv2d] if layer_type == "conv2d" else None
    )
    assert cl is not None, f"layer_type {layer_type} not supported"

    # create a dict with the layer type as key and the input and weight specs as values
    mods = gather_submodules(
        model,
        should_do=cls_passlist_should_do(cl),
    )

    # print([name for name, module in mods])

    return {
        name: {
            "input": copy.deepcopy(input_spec),
            "weight": copy.deepcopy(weight_spec),
        }
        for name, module in mods
    }


resnet20_fuse_pairs = [
    # stem
    ("conv1", "bn1"),
    # layer1 – BasicBlock(0, 1, 2)
    ("layer1.0.conv1", "layer1.0.bn1"),
    ("layer1.0.conv2", "layer1.0.bn2"),
    ("layer1.1.conv1", "layer1.1.bn1"),
    ("layer1.1.conv2", "layer1.1.bn2"),
    ("layer1.2.conv1", "layer1.2.bn1"),
    ("layer1.2.conv2", "layer1.2.bn2"),
    # layer2 – BasicBlock(0, 1, 2)
    ("layer2.0.conv1", "layer2.0.bn1"),
    ("layer2.0.conv2", "layer2.0.bn2"),
    ("layer2.0.shortcut.0", "layer2.0.shortcut.1"),  # down‑sample path
    ("layer2.1.conv1", "layer2.1.bn1"),
    ("layer2.1.conv2", "layer2.1.bn2"),
    ("layer2.2.conv1", "layer2.2.bn1"),
    ("layer2.2.conv2", "layer2.2.bn2"),
    # layer3 – BasicBlock(0, 1, 2)
    ("layer3.0.conv1", "layer3.0.bn1"),
    ("layer3.0.conv2", "layer3.0.bn2"),
    ("layer3.0.shortcut.0", "layer3.0.shortcut.1"),  # down‑sample path
    ("layer3.1.conv1", "layer3.1.bn1"),
    ("layer3.1.conv2", "layer3.1.bn2"),
    ("layer3.2.conv1", "layer3.2.bn1"),
    ("layer3.2.conv2", "layer3.2.bn2"),
]


def get_fuse_bn_keys(model_name: str):
    if model_name == "resnet20":
        return resnet20_fuse_pairs
    else:
        print("Model not supported for BN fusion")
        return []


def to_quantized_online(
    model: nn.Module,
    specs: Dict[str, Dict[Literal["input", "weight"], IntAffineQuantizationSpec]],
    inplace=True,
    fuse_bn_keys=None,
    **kwargs,
):
    if not inplace:
        model = copy.deepcopy(model)
    if fuse_bn_keys is not None:
        fuse_conv_bn(model, fuse_bn_keys, fuse_impl=fuse_batch_norm_inference)

    modules_to_replace = gather_submodules(
        model,
        should_do=keys_passlist_should_do(
            specs.keys(),
        ),
    )
    for name, module in tqdm(modules_to_replace, desc="Replacing modules"):
        parent_module = model
        *parent_path, attr_name = name.split(".")
        for part in parent_path:
            parent_module = getattr(parent_module, part)

        setattr(
            parent_module,
            attr_name,
            (
                QuantizedLinear(specs[name]["weight"], specs[name]["input"], module)
                if isinstance(module, (nn.Linear, nn.LazyLinear))
                else (
                    QuantizedConv2d(specs[name]["weight"], specs[name]["input"], module)
                    if isinstance(module, (nn.Conv2d, nn.LazyConv2d))
                    else module
                )
            ),
        )
        modules_to_replace[modules_to_replace.index((name, module))] = None
        del module
        gc.collect()
        torch.cuda.empty_cache()

    del modules_to_replace
    gc.collect()
    torch.cuda.empty_cache()
    return model


def fuse_conv_bn_qat(
    conv: nn.Conv2d,
    bn: nn.BatchNorm2d,
    conv_name: str,
    bn_name: str,
    specs: Dict[str, Dict[Literal["input", "weight"], IntAffineQuantizationSpec]],
):
    if conv_name not in specs:
        raise KeyError(f"Conv layer {conv_name} not found in model")

    weight_spec = specs[conv_name]["weight"]
    input_spec = specs[conv_name]["input"]

    return FusedQATConv2dBatchNorm2d(
        weight_spec,
        input_spec,
        conv,
        bn,
    )


def fuse_conv_bn_lsq(
    conv: nn.Conv2d,
    bn: nn.BatchNorm2d,
    conv_name: str,
    bn_name: str,
    specs: Dict[str, Dict[Literal["input", "weight"], IntAffineQuantizationSpec]],
    data_batch=None,
    online=False,
):
    if conv_name not in specs:
        raise KeyError(f"Conv layer {conv_name} not found in model")

    weight_spec = specs[conv_name]["weight"]
    input_spec = specs[conv_name]["input"]

    return FusedLSQConv2dBatchNorm2d(
        weight_spec,
        input_spec,
        conv,
        bn,
        data_batch=data_batch,
        online=online,
    )


def prepare_for_qat(
    model: nn.Module,
    specs: Dict[str, Dict[Literal["input", "weight"], IntAffineQuantizationSpec]],
    use_PACT=False,
    inplace=True,
    use_lsq=False,
    method_args={},
    fuse_bn_keys=None,
    **kwargs,
):

    if not inplace:
        model = copy.deepcopy(model)
    if use_lsq:
        assert "data_batch" in kwargs, "data_batch must be provided if use_lsq=True"
        model.eval()
        if fuse_bn_keys is not None:
            _model_acts = fuse_conv_bn(
                model,
                fuse_bn_keys,
                fuse_impl=fuse_batch_norm_inference,
                inplace=False,
            )
        else:
            _model_acts = model
        activations = get_activations(_model_acts, kwargs["data_batch"], specs)
    else:
        activations = None

    if fuse_bn_keys is not None:
        if use_lsq:
            fuse_conv_bn(
                model,
                fuse_bn_keys,
                fuse_impl=functools.partial(
                    fuse_conv_bn_lsq,
                    specs=specs,
                    data_batch=kwargs["data_batch"],
                    online=method_args.get("online", False),
                ),
            )
        else:
            fuse_conv_bn(
                model,
                fuse_bn_keys,
                fuse_impl=functools.partial(
                    fuse_conv_bn_qat,
                    specs=specs,
                ),
            )
    modules_to_replace = gather_submodules(
        model,
        should_do=keys_passlist_should_do(
            specs.keys(),
        ),
    )
    for name, module in tqdm(modules_to_replace, desc="Replacing modules"):
        parent_module = model
        *parent_path, attr_name = name.split(".")
        for part in parent_path:
            parent_module = getattr(parent_module, part)
        mod = module
        if not use_lsq:
            if isinstance(module, (nn.Linear, nn.LazyLinear)):
                mod = QATLinear(specs[name]["weight"], specs[name]["input"], module)
            elif isinstance(module, (nn.Conv2d, nn.LazyConv2d)):
                mod = QATConv2d(specs[name]["weight"], specs[name]["input"], module)
        elif use_lsq:
            online = method_args.get("online", False)
            if isinstance(module, (nn.Linear, nn.LazyLinear)):
                mod = LSQLinear(
                    specs[name]["weight"],
                    specs[name]["input"],
                    module,
                    activations[name],
                    online=online,
                )
            elif isinstance(module, (nn.Conv2d, nn.LazyConv2d)):
                mod = LSQConv2d(
                    specs[name]["weight"],
                    specs[name]["input"],
                    module,
                    activations[name],
                    online=online,
                )
        if use_PACT and isinstance(module, (nn.ReLU, nn.ReLU6)):
            mod = PACTReLU()
        setattr(
            parent_module,
            attr_name,
            (mod),
        )

    return model


def requantize_lsq(
    model: nn.Module,
    specs: Dict[str, Dict[Literal["input", "weight"], IntAffineQuantizationSpec]],
    inplace=True,
    **kwargs,
):
    if not inplace:
        model = copy.deepcopy(model)
    modules_to_replace = gather_submodules(
        model,
        should_do=keys_passlist_should_do(
            specs.keys(),
        ),
    )
    for name, module in tqdm(modules_to_replace, desc="Replacing modules"):
        parent_module = model
        *parent_path, attr_name = name.split(".")
        for part in parent_path:
            parent_module = getattr(parent_module, part)

        if not isinstance(module, (LSQConv2d, LSQLinear, FusedLSQConv2dBatchNorm2d)):
            continue

        nbits_acts = module.weight_spec.nbits
        nbits_weights = module.input_spec.nbits

        new_nbits_acts = specs[name]["input"].nbits
        new_nbits_weights = specs[name]["weight"].nbits

        scale_weights = module.weight_info.scale
        scale_acts = module.input_info.scale

        reduce_ratio_weights = float(new_nbits_weights) / float(nbits_weights)
        reduce_ratio_acts = float(new_nbits_acts) / float(nbits_acts)

        new_scale_weights = scale_weights * reduce_ratio_weights
        new_scale_acts = scale_acts * reduce_ratio_acts

        # modify inplace
        module.weight_info.scale = nn.Parameter(new_scale_weights, requires_grad=True)
        module.input_info.scale = nn.Parameter(new_scale_acts, requires_grad=True)
        module.weight_spec = specs[name]["weight"]
        module.input_spec = specs[name]["input"]

    return model


def get_activations(model, data_loader, spec, move_to_cpu=False):
    if isinstance(data_loader, torch.Tensor):
        data_loader = [data_loader]
    activations = {}
    hooks = []
    for name, module in gather_submodules(
        model, should_do=keys_passlist_should_do(spec.keys())
    ):
        activations[name] = []

        def hook_fn(activations):
            def _hook_fn(module, input, output):
                activations.append(input[0].detach())

            return _hook_fn

        hooks.append(module.register_forward_hook(hook_fn(activations[name])))

    for element in data_loader:
        if isinstance(element, (tuple, list)):
            element = element[0]
        element = element.to(next(model.parameters()).device)
        model(element)

    for hook in hooks:
        hook.remove()

    # cat activations
    for name in activations:
        # print(activations)
        activations[name] = torch.cat(activations[name], dim=0).cuda()
    return activations


def to_quantized_offline(
    model: nn.Module,
    specs: Dict[str, Dict[Literal["input", "weight"], IntAffineQuantizationSpec]],
    inplace=True,
    data_loader=None,
    activations=None,
    fuse_bn_keys=None,
    **kwargs,
):

    if not inplace:
        model = copy.deepcopy(model)
    # only one of data_loader and activations should be provided
    assert (data_loader is None) != (
        activations is None
    ), "Either data_loader or activations should be provided"

    if fuse_bn_keys is not None:
        fuse_conv_bn(
            model,
            fuse_bn_keys,
            fuse_impl=fuse_batch_norm_inference,
        )
    if activations is None:
        # first, estimate the quantization parameters with hooks
        layer_and_input_acts = get_activations(model, data_loader, specs)

    else:
        layer_and_input_acts = activations

    input_infos = {}
    for name, input_act in layer_and_input_acts.items():
        input_infos[name] = calibrate(
            input_act,
            specs[name]["input"],
        )

    modules_to_replace = gather_submodules(
        model, should_do=keys_passlist_should_do(specs.keys())
    )
    for name, module in tqdm(modules_to_replace, desc="Replacing modules"):
        parent_module = model
        *parent_path, attr_name = name.split(".")
        for part in parent_path:
            parent_module = getattr(parent_module, part)

        setattr(
            parent_module,
            attr_name,
            (
                QuantizedLinear(specs[name]["weight"], input_infos[name], module)
                if isinstance(module, nn.Linear)
                else QuantizedConv2d(specs[name]["weight"], input_infos[name], module)
            ),
        )

    return model


def merge_qat_lsq_into_offline_quantized_model(model: nn.Module, inplace=True):
    if not inplace:
        model = copy.deepcopy(model)
    modules_to_replace = gather_submodules(
        model,
        should_do=default_should_do,
    )
    for name, module in tqdm(modules_to_replace, desc="Replacing modules"):
        parent_module = model
        *parent_path, attr_name = name.split(".")
        for part in parent_path:
            parent_module = getattr(parent_module, part)

        setattr(
            parent_module,
            attr_name,
            (
                module.to_quant_linear()
                if hasattr(module, "to_quant_linear")
                else (
                    module.to_quant_conv2d()
                    if hasattr(module, "to_quant_conv2d")
                    else module
                )
            ),
        )

    return model


def merge_qat_model(model: nn.Module, inplace=True):
    if not inplace:
        model = copy.deepcopy(model)
    modules_to_replace = gather_submodules(
        model,
        should_do=cls_passlist_should_do((QATLinear, QATConv2d, LSQConv2d, LSQLinear)),
    )
    for name, module in tqdm(modules_to_replace, desc="Replacing modules"):
        parent_module = model
        *parent_path, attr_name = name.split(".")
        for part in parent_path:
            parent_module = getattr(parent_module, part)

        setattr(
            parent_module,
            attr_name,
            (
                module.to_linear()
                if hasattr(module, "to_linear")
                else module.to_conv2d() if hasattr(module, "to_conv2d") else module
            ),
        )

    return model


def to_quantized_kmeans(
    model: nn.Module,
    input_specs: IntAffineQuantizationSpec,
    weight_specs: IntAffineQuantizationSpec,
    inplace=True,
    should_do=default_should_do,
    **kwargs,
):
    if not inplace:
        model = copy.deepcopy(model)
    modules_to_replace = gather_submodules(
        model,
        should_do=should_do,
    )
    for name, module in tqdm(modules_to_replace, desc="Replacing modules"):
        parent_module = model
        *parent_path, attr_name = name.split(".")
        for part in parent_path:
            parent_module = getattr(parent_module, part)

        setattr(
            parent_module,
            attr_name,
            (
                KMeansQuantizedLinear(
                    module, weight_specs["linear"], input_specs["linear"]
                )
                if isinstance(module, nn.Linear)
                else KMeansQuantizedConv2d(
                    module, weight_specs["conv2d"], input_specs["conv2d"]
                )
            ),
        )

    return model
