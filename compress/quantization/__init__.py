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
)

import torch.nn.functional as F

from compress.quantization.common import (
    IntAffineQuantizationSpec,
    IntAffineQuantizationInfo,
    IntAffineQuantizationMode,
    ste_floor,
)
from compress.common import (
    gather_submodules,
    default_should_do,
    cls_passlist_should_do,
    combine_should_do,
    keys_passlist_should_do,
)
from torch import nn
from tqdm import tqdm
import torch
import gc
from typing import Dict, Literal


def assert_all_in_classes(
    dict: Dict[str, IntAffineQuantizationSpec],
    classes: tuple,
    message: str = "All keys in the dict must be in the classes",
):
    for key in dict:
        assert isinstance(key, tuple(classes)), f"{key} is not in {classes}. {message}"
    return True


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
            "input": input_spec,
            "weight": weight_spec,
        }
        for name, module in mods
    }


def to_quantized_online(
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


def prepare_for_qat(
    model: nn.Module,
    specs: Dict[str, Dict[Literal["input", "weight"], IntAffineQuantizationSpec]],
    use_PACT=False,
    inplace=True,
    use_lsq=False,
    **kwargs,
):

    if not inplace:
        model = copy.deepcopy(model)
    if use_lsq:
        assert "data_batch" in kwargs, "data_batch must be provided if use_lsq=True"
        activations = get_activations(model, kwargs["data_batch"])
    else:
        activations = None
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
            if isinstance(module, (nn.Linear, nn.LazyLinear)):
                mod = LSQLinear(
                    specs[name]["weight"],
                    specs[name]["input"],
                    module,
                    activations[module],
                )
            elif isinstance(module, (nn.Conv2d, nn.LazyConv2d)):
                mod = LSQConv2d(
                    specs[name]["weight"],
                    specs[name]["input"],
                    module,
                    activations[module],
                )
        if use_PACT and isinstance(module, nn.ReLU):
            mod = PACTReLU()
        setattr(
            parent_module,
            attr_name,
            (mod),
        )

    return model


def get_activations(model, data_loader):
    if isinstance(data_loader, torch.Tensor):
        data_loader = [data_loader]
    activations = {}
    hooks = []
    for _, module in gather_submodules(model, should_do=default_should_do):
        activations[module] = []

        def hook_fn(activations):
            def _hook_fn(module, input, output):
                activations.append(output.detach())

            return _hook_fn

        hooks.append(module.register_forward_hook(hook_fn(activations[module])))

    for element in data_loader:
        element = element.to(next(model.parameters()).device)
        model(element)

    for hook in hooks:
        hook.remove()

    # cat activations
    for module in activations:
        activations[module] = torch.cat(activations[module], dim=0)
    return activations


def to_quantized_offline(
    model: nn.Module,
    specs: Dict[str, Dict[Literal["input", "weight"], IntAffineQuantizationSpec]],
    inplace=True,
    should_do=default_should_do,
    data_loader=None,
    activations=None,
    **kwargs,
):
    # only one of data_loader and activations should be provided
    assert (data_loader is None) != (
        activations is None
    ), "Either data_loader or activations should be provided"

    if activations is None:
        # first, estimate the quantization parameters with hooks
        layer_and_input_acts = get_activations(model, data_loader)

    else:
        layer_and_input_acts = activations

    input_infos = {}
    for module, input_act in layer_and_input_acts.items():
        input_infos[module] = calibrate(
            input_act,
            specs[next(iter(specs))]["input"],
        )

    if not inplace:
        model_initializer = kwargs.pop("model_initializer", None)
        assert (
            model_initializer is not None
        ), "model_initializer must be provided if inplace=False"
        model_ = model_initializer()
        model_.load_state_dict(model.state_dict())
        model = model_
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
                QuantizedLinear(specs[name]["weight"], input_infos[module], module)
                if isinstance(module, nn.Linear)
                else QuantizedConv2d(specs[name]["weight"], input_infos[module], module)
            ),
        )

    return model


def prepare_for_qat_lsq(
    model: nn.Module,
    specs: Dict[str, Dict[Literal["input", "weight"], IntAffineQuantizationSpec]],
    data_batch: torch.Tensor,
    inplace=True,
    **kwargs,
):
    if not inplace:
        model_initializer = kwargs.pop("model_initializer", None)
        assert (
            model_initializer is not None
        ), "model_initializer must be provided if inplace=False"
        model_ = model_initializer()
        model_.load_state_dict(model.state_dict())
        model = model_
    activations = get_activations(model, data_batch)
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
                LSQLinear(
                    specs[name]["weight"],
                    specs[name]["input"],
                    module,
                    activations[module],
                )
                if isinstance(module, nn.Linear)
                else LSQConv2d(
                    specs[name]["weight"],
                    specs[name]["input"],
                    module,
                    activations[module],
                )
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


def fake_quantize_floor(x, info: IntAffineQuantizationInfo):
    assert info.zero_point is None
    return ste_floor(x / info.scale).clamp(info.qmin, info.qmax) * info.scale


def adaround_for_layer(model, layer, input_specs, weight_specs, data_loader):
    model.train()
    layer_w = layer.weight
    V = nn.Parameter(torch.randn_like(layer_w).requires_grad_(True))
    optim = torch.optim.Adam([V], lr=0.001)

    def rectified_sigmoid(V):
        x = torch.sigmoid(V)
        x = x * 1.2 + (-0.1)
        return torch.clamp(x, 0.0, 1.0)

    def regularize(V, beta=1.0):
        hV = rectified_sigmoid(V)
        diff = 2.0 * hV - 1.0
        diff_abs_pow = torch.abs(diff).pow(beta)
        val = 1.0 - diff_abs_pow
        return val.sum()

    def beta_schedule(epoch):
        return max(1, epoch - 3) / 50.0

    info = (
        calibrate(layer.weight, input_specs["linear"])
        if isinstance(layer, nn.Linear)
        else calibrate(layer.weight, input_specs["conv2d"])
    )
    epoch = 0

    def _hook_fn(module, input, output):
        input = input[0]
        if module == layer:
            if isinstance(module, nn.Linear):
                curr_out_not_quant = F.linear(input, module.weight, module.bias)
                quant_w = fake_quantize_floor(
                    module.weight + rectified_sigmoid(V), info
                )
                curr_out_quant = F.linear(input, quant_w, module.bias)
            else:
                assert isinstance(module, nn.Conv2d)
                curr_out_not_quant = F.conv2d(
                    input,
                    module.weight,
                    module.bias,
                    module.stride,
                    module.padding,
                    module.dilation,
                    module.groups,
                )
                quant_w = fake_quantize_floor(
                    module.weight + rectified_sigmoid(V), info
                )
                curr_out_quant = F.conv2d(
                    input,
                    quant_w,
                    module.bias,
                    module.stride,
                    module.padding,
                    module.dilation,
                    module.groups,
                )

            def mse(x, y):
                return (x - y).pow(2).mean()

            loss = mse(
                curr_out_not_quant.reshape(-1), curr_out_quant.reshape(-1)
            ) + regularize(V, beta_schedule(epoch))
            optim.zero_grad()
            layer_w.grad = None
            V.grad = None
            loss.backward(inputs=[V])
            optim.step()
            raise ValueError("Stop here")

        else:
            return output

    hook = layer.register_forward_hook(_hook_fn)

    for i in range(50):
        for element in data_loader:
            try:
                model(element)
            except ValueError:
                continue
        epoch += 1

    hook.remove()
    ret = layer.weight + rectified_sigmoid(V)

    """if bool(torch.isnan(rectified_sigmoid(V)).sum() != 0):
        print(f"Found NaN in the weight of {layer}. Total NaNs: {torch.isnan(ret).sum()} out of {ret.numel()}")
        print(V)
    """
    is_nan_idxs = torch.isnan(ret)
    ret[is_nan_idxs] = layer.weight[is_nan_idxs]
    return ret


def to_quantized_adaround(
    model: nn.Module,
    input_specs: IntAffineQuantizationSpec,
    weight_specs: IntAffineQuantizationSpec,
    data_loader,
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
        w = nn.Parameter(
            adaround_for_layer(model, module, input_specs, weight_specs, data_loader)
        )
        module.weight = nn.Parameter(w)
        setattr(
            parent_module,
            attr_name,
            (
                QuantizedLinear(weight_specs["linear"], input_specs["linear"], module)
                if isinstance(module, nn.Linear)
                else QuantizedConv2d(
                    weight_specs["conv2d"], input_specs["conv2d"], module
                )
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
