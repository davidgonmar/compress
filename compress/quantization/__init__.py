import copy
from compress.quantization.calibrate import calibrate
from compress.quantization.ptq_ops import (
    QuantizedLinear,
    QuantizedConv2d,
    KMeansQuantizedConv2d,
    KMeansQuantizedLinear,
)
from compress.quantization.qat_ops import (
    QATConv2d,
    QATLinear,
    LSQConv2d,
    LSQLinear,
    snap_loss_model_activations,
    snap_loss_model_params,
    SnapRegularizer,
)

import torch.nn.functional as F

from compress.quantization.util import (
    IntQuantizationSpec,
    IntQuantizationInfo,
    ste_floor,
)
from compress.common import gather_submodules, default_should_do
from torch import nn
from tqdm import tqdm
import torch
import gc


def to_quantized_online(
    model: nn.Module,
    input_specs: IntQuantizationSpec,
    weight_specs: IntQuantizationSpec,
    inplace=True,
    should_do=default_should_do,
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
    modules_to_replace = gather_submodules(model, should_do=should_do, prefix="")
    for name, module in tqdm(modules_to_replace, desc="Replacing modules"):
        parent_module = model
        *parent_path, attr_name = name.split(".")
        for part in parent_path:
            parent_module = getattr(parent_module, part)

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
        modules_to_replace[modules_to_replace.index((name, module))] = None
        del module
        gc.collect()
        torch.cuda.empty_cache()

    del modules_to_replace
    gc.collect()
    torch.cuda.empty_cache()
    return model


def get_activations(model, data_loader):
    if isinstance(data_loader, torch.Tensor):
        data_loader = [data_loader]
    activations = {}
    hooks = []
    for _, module in gather_submodules(model, should_do=default_should_do, prefix=""):
        activations[module] = []

        def hook_fn(activations):
            def _hook_fn(module, input, output):
                activations.append(output.detach())

            return _hook_fn

        hooks.append(module.register_forward_hook(hook_fn(activations[module])))

    for element in data_loader:
        model(element)

    for hook in hooks:
        hook.remove()

    # cat activations
    for module in activations:
        activations[module] = torch.cat(activations[module], dim=0)
    return activations


def to_quantized_offline(
    model: nn.Module,
    input_specs: IntQuantizationSpec,
    weight_specs: IntQuantizationSpec,
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
            input_specs["linear" if isinstance(module, nn.Linear) else "conv2d"],
        )

    if not inplace:
        model_initializer = kwargs.pop("model_initializer", None)
        assert (
            model_initializer is not None
        ), "model_initializer must be provided if inplace=False"
        model_ = model_initializer()
        model_.load_state_dict(model.state_dict())
        model = model_
    modules_to_replace = gather_submodules(model, should_do=should_do, prefix="")
    for name, module in tqdm(modules_to_replace, desc="Replacing modules"):
        parent_module = model
        *parent_path, attr_name = name.split(".")
        for part in parent_path:
            parent_module = getattr(parent_module, part)

        setattr(
            parent_module,
            attr_name,
            (
                QuantizedLinear(weight_specs["linear"], input_infos[module], module)
                if isinstance(module, nn.Linear)
                else QuantizedConv2d(
                    weight_specs["conv2d"], input_infos[module], module
                )
            ),
        )

    return model


def prepare_for_qat(
    model: nn.Module,
    input_specs: IntQuantizationSpec,
    weight_specs: IntQuantizationSpec,
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
    modules_to_replace = gather_submodules(
        model, should_do=default_should_do, prefix=""
    )
    for name, module in tqdm(modules_to_replace, desc="Replacing modules"):
        parent_module = model
        *parent_path, attr_name = name.split(".")
        for part in parent_path:
            parent_module = getattr(parent_module, part)

        if isinstance(module, nn.Linear):
            if name in weight_specs:
                assert name in input_specs, f"Input spec for {name} not found"
                mod = QATLinear(weight_specs[name], input_specs[name], module)
            else:
                mod = QATLinear(weight_specs["linear"], input_specs["linear"], module)
        elif isinstance(module, nn.Conv2d):
            if name in weight_specs:
                assert name in input_specs, f"Input spec for {name} not found"
                mod = QATConv2d(weight_specs[name], input_specs[name], module)
            else:
                mod = QATConv2d(weight_specs["conv2d"], input_specs["conv2d"], module)
        else:
            continue
        setattr(
            parent_module,
            attr_name,
            (mod),
        )

    return model


def prepare_for_qat_lsq(
    model: nn.Module,
    input_specs: IntQuantizationSpec,
    weight_specs: IntQuantizationSpec,
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
        model, should_do=default_should_do, prefix=""
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
                    weight_specs["linear"],
                    input_specs["linear"],
                    module,
                    activations[module],
                )
                if isinstance(module, nn.Linear)
                else LSQConv2d(
                    weight_specs["conv2d"],
                    input_specs["conv2d"],
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
        model, should_do=default_should_do, prefix=""
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
        model, should_do=default_should_do, prefix=""
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


def fake_quantize_floor(x, info: IntQuantizationInfo):
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
    input_specs: IntQuantizationSpec,
    weight_specs: IntQuantizationSpec,
    data_loader,
    inplace=True,
    should_do=default_should_do,
    **kwargs,
):
    if not inplace:
        model = copy.deepcopy(model)
    modules_to_replace = gather_submodules(model, should_do=should_do, prefix="")
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
    input_specs: IntQuantizationSpec,
    weight_specs: IntQuantizationSpec,
    inplace=True,
    should_do=default_should_do,
    **kwargs,
):
    if not inplace:
        model = copy.deepcopy(model)
    modules_to_replace = gather_submodules(model, should_do=should_do, prefix="")
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
