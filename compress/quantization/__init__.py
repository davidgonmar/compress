import copy
from itertools import product
from typing import List, Tuple
from compress.quantization.calibrate import calibrate
from compress.quantization.quantized_ops import (
    QuantizedLinear,
    QuantizedConv2d,
    FakeQuantizedLinear,
    FakeQuantizedConv2d,
)
from compress.quantization.util import IntQuantizationSpec
from compress.common import gather_submodules, default_should_do
from torch import nn
from tqdm import tqdm
import torch


def to_quantized_online(
    model: nn.Module,
    input_specs: IntQuantizationSpec,
    weight_specs: IntQuantizationSpec,
    inplace=True,
    should_do=default_should_do,
    **kwargs,
):
    modules_to_replace = gather_submodules(model, should_do=should_do, prefix="")
    if not inplace:
        model_initializer = kwargs.pop("model_initializer", None)
        assert (
            model_initializer is not None
        ), "model_initializer must be provided if inplace=False"
        model_ = model_initializer()
        model_.load_state_dict(model.state_dict())
        model = model_

    for name, module in tqdm(modules_to_replace, desc="Replacing modules"):
        parent_module = model
        *parent_path, attr_name = name.split(".")
        for part in parent_path:
            parent_module = getattr(parent_module, part)

        setattr(
            parent_module,
            attr_name,
            QuantizedLinear(weight_specs["linear"], input_specs["linear"], module)
            if isinstance(module, nn.Linear)
            else QuantizedConv2d(weight_specs["conv2d"], input_specs["conv2d"], module),
        )

    return model


def get_activations(model, data_loader):
    activations = {}
    hooks = []
    for _, module in gather_submodules(model, should_do=default_should_do, prefix=""):
        activations[module] = []

        def hook_fn(activations):
            def _hook_fn(module, input, output):
                activations.append(output.detach().cpu())

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

    modules_to_replace = gather_submodules(model, should_do=should_do, prefix="")
    if not inplace:
        model_initializer = kwargs.pop("model_initializer", None)
        assert (
            model_initializer is not None
        ), "model_initializer must be provided if inplace=False"
        model_ = model_initializer()
        model_.load_state_dict(model.state_dict())
        model = model_

    for name, module in tqdm(modules_to_replace, desc="Replacing modules"):
        parent_module = model
        *parent_path, attr_name = name.split(".")
        for part in parent_path:
            parent_module = getattr(parent_module, part)

        setattr(
            parent_module,
            attr_name,
            QuantizedLinear(weight_specs["linear"], input_infos[module], module)
            if isinstance(module, nn.Linear)
            else QuantizedConv2d(weight_specs["conv2d"], input_infos[module], module),
        )

    return model


def prepare_for_qat(
    model: nn.Module,
    input_specs: IntQuantizationSpec,
    weight_specs: IntQuantizationSpec,
    inplace=True,
    **kwargs,
):
    modules_to_replace = gather_submodules(
        model, should_do=default_should_do, prefix=""
    )
    if not inplace:
        model_initializer = kwargs.pop("model_initializer", None)
        assert (
            model_initializer is not None
        ), "model_initializer must be provided if inplace=False"
        model_ = model_initializer()
        model_.load_state_dict(model.state_dict())
        model = model_

    for name, module in tqdm(modules_to_replace, desc="Replacing modules"):
        parent_module = model
        *parent_path, attr_name = name.split(".")
        for part in parent_path:
            parent_module = getattr(parent_module, part)

        setattr(
            parent_module,
            attr_name,
            FakeQuantizedLinear(weight_specs["linear"], input_specs["linear"], module)
            if isinstance(module, nn.Linear)
            else FakeQuantizedConv2d(
                weight_specs["conv2d"], input_specs["conv2d"], module
            ),
        )

    return model


def merge_qat_model(model: nn.Module, inplace=True):
    modules_to_replace = gather_submodules(
        model, should_do=default_should_do, prefix=""
    )
    if not inplace:
        model = copy.deepcopy(model)

    for name, module in tqdm(modules_to_replace, desc="Replacing modules"):
        parent_module = model
        *parent_path, attr_name = name.split(".")
        for part in parent_path:
            parent_module = getattr(parent_module, part)

        setattr(
            parent_module,
            attr_name,
            module.to_linear()
            if isinstance(module, FakeQuantizedLinear)
            else module.to_conv2d(),
        )

    return model
