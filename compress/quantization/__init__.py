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
    LinearWeightsPerOutChannel,
)
from compress.utils import (
    gather_submodules,
    default_should_do,
    cls_passlist_should_do,
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
        activations = get_activations_vision(_model_acts, kwargs["data_batch"], specs)
    else:
        activations = None

    if fuse_bn_keys is not None:
        if use_lsq:
            fuse_conv_bn(
                model,
                fuse_bn_keys,
                fuse_impl=lambda conv, bn, conv_name, bn_name: fuse_conv_bn_lsq(
                    conv=conv,
                    bn=bn,
                    conv_name=conv_name,
                    bn_name=bn_name,
                    specs=specs,
                    data_batch=activations[conv_name],
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
        online = method_args.get("online", False)
        if not use_lsq:
            if isinstance(module, (nn.Linear, nn.LazyLinear)):
                mod = QATLinear(
                    specs[name]["weight"], specs[name]["input"], module, online=online
                )
            elif isinstance(module, (nn.Conv2d, nn.LazyConv2d)):
                mod = QATConv2d(
                    specs[name]["weight"], specs[name]["input"], module, online=online
                )
        elif use_lsq:
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


def requantize_qat(
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
    if kwargs.get("data_batch") is not None:
        data_batch = kwargs["data_batch"]
        activations = get_activations_vision(model, data_batch, specs)
    for name, module in tqdm(modules_to_replace, desc="Replacing modules"):
        parent_module = model
        *parent_path, attr_name = name.split(".")
        for part in parent_path:
            parent_module = getattr(parent_module, part)

        if not isinstance(
            module,
            (
                LSQConv2d,
                LSQLinear,
                FusedLSQConv2dBatchNorm2d,
                QATConv2d,
                QATLinear,
                FusedQATConv2dBatchNorm2d,
            ),
        ):
            continue

        if hasattr(module, "input_spec"):
            module.input_spec = specs[name]["input"]
        if hasattr(module, "weight_spec"):
            module.weight_spec = specs[name]["weight"]
        if hasattr(module, "weight_info"):
            # calibrate
            module.weight_info = calibrate(
                module.weight,
                specs[name]["weight"],
            )
        if hasattr(module, "input_info"):
            assert kwargs.get("data_batch") is not None, "data_batch must be provided"

            # calibrate
            module.input_info = calibrate(
                activations[name],
                specs[name]["input"],
            )

    return model


def get_activations_vision(model, data_loader, keys, move_to_cpu=False):
    if isinstance(data_loader, (torch.Tensor, tuple)):
        data_loader = [data_loader]
    activations = {}
    hooks = []
    for name, module in gather_submodules(
        model, should_do=keys_passlist_should_do(keys)
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
        layer_and_input_acts = get_activations_vision(model, data_loader, specs)

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


def get_activations_transformers(model, dataloader, specs):
    activations = {}
    hooks = []

    def save_activation(name):
        def hook(module, input, output):
            # Store detached CPU tensor of output activations
            activations[name].append(output.detach().cpu())

        return hook

    # Register forward hooks on all linear modules
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and name in specs.keys():
            activations[name] = []
            handles = module.register_forward_hook(save_activation(name))
            hooks.append(handles)

    # Run through the calibration dataloader to collect activations
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            # Move inputs to the model's device, skip labels
            inputs = {
                k: v.to(next(model.parameters()).device)
                for k, v in batch.items()
                if k != "labels"
            }
            print(inputs.keys())
            _ = model(**inputs)

    # Remove hooks to clean up
    for handle in hooks:
        handle.remove()

    # Concatenate activation lists into single tensors per layer
    for name in activations:
        activations[name] = torch.cat(activations[name], dim=0)

    return activations


def separate_params(model: nn.Module):
    quant_params = []
    for mod in model.modules():
        if isinstance(mod, IntAffineQuantizationInfo):
            quant_params.extend(
                p for p in mod.parameters(recurse=False) if p.requires_grad
            )

    quant_set = set(quant_params)
    others = [p for p in model.parameters() if p.requires_grad and p not in quant_set]
    return {"quant_params": quant_params, "others": others}
