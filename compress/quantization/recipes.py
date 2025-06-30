from compress.quantization import (
    IntAffineQuantizationSpec,
    IntAffineQuantizationMode,
)
from compress.quantization.common import (
    PerTensor,
    ConvWeightsPerOutChannel,
    LinearWeightsPerOutChannel,
)
import torchvision
import torch
from compress.experiments.cifar_resnet import resnet20
import torch.nn as nn


# These are based on the torchvision models

# ======================= MOBILENET V2 ==========================


def get_after_relu_mobilenetv2():
    after_relu = set()

    # layer 1
    after_relu.add("features.1.conv.0.0")
    after_relu.add("features.1.conv.1")

    for i in range(2, 18):
        after_relu.add(f"features.{i}.conv.1.0")
        after_relu.add(f"features.{i}.conv.2")

    after_relu.add("classifier.1")
    return after_relu


def get_non_after_relu_mobilenetv2():
    non_after_relu = set()

    # layer 1
    non_after_relu.add("features.0.0")

    for i in range(2, 18):
        non_after_relu.add(f"features.{i}.conv.0.0")

    non_after_relu.add("features.18.0")
    return non_after_relu


def get_all_mobilenetv2_layers():
    return get_after_relu_mobilenetv2() | get_non_after_relu_mobilenetv2()


def mobilenetv2_recipe_quant(
    bits_activation: int,
    bits_weight: int,
    clip_percentile: float,
    leave_edge_layers_8_bits: bool,
    symmetric: bool,
):

    # careful in terms of choosing symmetric or asymmetric
    if symmetric:
        # if is after relu -> use unsigned quantization
        # if is not after relu -> use signed quantization
        after_relu = get_after_relu_mobilenetv2()
        non_after_relu = get_non_after_relu_mobilenetv2()

        weight_specs = {}

        for k in after_relu:
            weight_specs[k] = IntAffineQuantizationSpec(
                nbits=bits_weight,
                signed=True,
                quant_mode=IntAffineQuantizationMode.SYMMETRIC,
                percentile=clip_percentile,
            )
        for k in non_after_relu:
            weight_specs[k] = IntAffineQuantizationSpec(
                nbits=bits_weight,
                signed=True,
                quant_mode=IntAffineQuantizationMode.SYMMETRIC,
                percentile=clip_percentile,
            )

        input_specs = {}
        for k in after_relu:
            input_specs[k] = IntAffineQuantizationSpec(
                nbits=bits_activation,
                signed=False,
                quant_mode=IntAffineQuantizationMode.SYMMETRIC,
                percentile=clip_percentile,
            )
        for k in non_after_relu:
            input_specs[k] = IntAffineQuantizationSpec(
                nbits=bits_activation,
                signed=True,
                quant_mode=IntAffineQuantizationMode.SYMMETRIC,
                percentile=clip_percentile,
            )

        if leave_edge_layers_8_bits:
            for k in ("classifier.1", "features.0.0"):
                input_specs[k] = IntAffineQuantizationSpec(
                    nbits=8,
                    signed=k == "features.0.0",
                    quant_mode=IntAffineQuantizationMode.SYMMETRIC,
                    percentile=clip_percentile,
                )
                weight_specs[k] = IntAffineQuantizationSpec(
                    nbits=8,
                    signed=True,
                    quant_mode=IntAffineQuantizationMode.SYMMETRIC,
                    percentile=clip_percentile,
                )

        quant_specs = {}
        for k in weight_specs.keys():
            quant_specs[k] = {
                "input": input_specs[k],
                "weight": weight_specs[k],
            }

        return quant_specs
    else:
        # otherwise it does not matter and we can just do all signed
        weight_specs = {}
        input_specs = {}
        for k in get_all_mobilenetv2_layers():
            weight_specs[k] = IntAffineQuantizationSpec(
                nbits=bits_weight,
                signed=True,
                quant_mode=IntAffineQuantizationMode.ASYMMETRIC,
                percentile=clip_percentile,
            )
            input_specs[k] = IntAffineQuantizationSpec(
                nbits=bits_activation,
                signed=True,
                quant_mode=IntAffineQuantizationMode.ASYMMETRIC,
                percentile=clip_percentile,
            )
        if leave_edge_layers_8_bits:
            for k in ("classifier.1", "features.0.0"):
                input_specs[k] = IntAffineQuantizationSpec(
                    nbits=8,
                    signed=True,
                    quant_mode=IntAffineQuantizationMode.ASYMMETRIC,
                    percentile=clip_percentile,
                )
                weight_specs[k] = IntAffineQuantizationSpec(
                    nbits=8,
                    signed=True,
                    quant_mode=IntAffineQuantizationMode.ASYMMETRIC,
                    percentile=clip_percentile,
                )
        quant_specs = {}
        for k in weight_specs.keys():
            quant_specs[k] = {
                "input": input_specs[k],
                "weight": weight_specs[k],
            }
        return quant_specs


def get_all_resnet18_layers():
    with torch.device("meta"):
        model = torchvision.models.resnet18(weights=None)
        names = []
        for name, mod in model.named_modules():
            if isinstance(mod, torch.nn.Conv2d):
                names.append(name)
        return names


def get_resnet18_recipe_quant(
    bits_activation: int,
    bits_weight: int,
    clip_percentile: float,
    leave_edge_layers_8_bits: bool,
    symmetric: bool,
):

    if symmetric:
        # if is after relu -> use unsigned quantization
        # if is not after relu -> use signed quantization
        after_relu = set()
        non_after_relu = set()

        # layer 1
        non_after_relu.add("conv1")

        for l in range(1, 5):
            after_relu.add(f"layer{l}.0.conv1")
            after_relu.add(f"layer{l}.0.conv2")
            after_relu.add(f"layer{l}.1.conv1")
            after_relu.add(f"layer{l}.1.conv2")

            # downsample
            if l in (2, 3, 4):
                after_relu.add(f"layer{l}.0.downsample.0")

        after_relu.add("fc")

        weight_specs = {}
        for k in after_relu:
            weight_specs[k] = IntAffineQuantizationSpec(
                nbits=bits_weight,
                signed=True,
                quant_mode=IntAffineQuantizationMode.SYMMETRIC,
                percentile=clip_percentile,
            )

        for k in non_after_relu:
            weight_specs[k] = IntAffineQuantizationSpec(
                nbits=bits_weight,
                signed=True,
                quant_mode=IntAffineQuantizationMode.SYMMETRIC,
                percentile=clip_percentile,
            )

        input_specs = {}

        for k in after_relu:
            input_specs[k] = IntAffineQuantizationSpec(
                nbits=bits_activation,
                signed=False,
                quant_mode=IntAffineQuantizationMode.SYMMETRIC,
                percentile=clip_percentile,
            )

        for k in non_after_relu:
            input_specs[k] = IntAffineQuantizationSpec(
                nbits=bits_activation,
                signed=True,
                quant_mode=IntAffineQuantizationMode.SYMMETRIC,
                percentile=clip_percentile,
            )

        if leave_edge_layers_8_bits:
            for k in ("fc", "conv1"):
                input_specs[k] = IntAffineQuantizationSpec(
                    nbits=8,
                    signed=k == "conv1",
                    quant_mode=IntAffineQuantizationMode.SYMMETRIC,
                    percentile=clip_percentile,
                )
                weight_specs[k] = IntAffineQuantizationSpec(
                    nbits=8,
                    signed=True,
                    quant_mode=IntAffineQuantizationMode.SYMMETRIC,
                    percentile=clip_percentile,
                )

        quant_specs = {}
        for k in weight_specs.keys():
            quant_specs[k] = {
                "input": input_specs[k],
                "weight": weight_specs[k],
            }

        return quant_specs
    else:
        # otherwise it does not matter and we can just do all signed
        weight_specs = {}
        input_specs = {}
        for k in get_all_resnet18_layers():
            weight_specs[k] = IntAffineQuantizationSpec(
                nbits=bits_weight,
                signed=True,
                quant_mode=IntAffineQuantizationMode.ASYMMETRIC,
                percentile=clip_percentile,
            )
            input_specs[k] = IntAffineQuantizationSpec(
                nbits=bits_activation,
                signed=True,
                quant_mode=IntAffineQuantizationMode.ASYMMETRIC,
                percentile=clip_percentile,
            )
        if leave_edge_layers_8_bits:
            for k in ("fc", "conv1"):
                input_specs[k] = IntAffineQuantizationSpec(
                    nbits=8,
                    signed=True,
                    quant_mode=IntAffineQuantizationMode.ASYMMETRIC,
                    percentile=clip_percentile,
                )
                weight_specs[k] = IntAffineQuantizationSpec(
                    nbits=8,
                    signed=True,
                    quant_mode=IntAffineQuantizationMode.ASYMMETRIC,
                    percentile=clip_percentile,
                )
        quant_specs = {}
        for k in weight_specs.keys():
            quant_specs[k] = {
                "input": input_specs[k],
                "weight": weight_specs[k],
            }
        return quant_specs


def get_all_resnet20_layers():
    with torch.device("meta"):
        model = resnet20()
        names = []
        for name, mod in model.named_modules():
            if isinstance(mod, torch.nn.Conv2d):
                names.append(name)
            elif isinstance(mod, torch.nn.Linear):
                names.append(name)
        return names


def get_resnet20_recipe_quant(
    bits_activation: int,
    bits_weight: int,
    clip_percentile: float,
    leave_edge_layers_8_bits: bool,
    symmetric: bool,
    quant_mode_override: IntAffineQuantizationMode | None = None,
    conv_weight_grouper=ConvWeightsPerOutChannel(),
    conv_activation_grouper=PerTensor(),
    linear_weight_grouper=LinearWeightsPerOutChannel(),
    linear_activation_grouper=PerTensor(),
):

    def _get_grouper(name, is_w):
        if "linear" in name:
            return linear_weight_grouper if is_w else linear_activation_grouper
        return conv_weight_grouper if is_w else conv_activation_grouper

    if symmetric:
        # if is after relu -> use unsigned quantization
        # if is not after relu -> use signed quantization
        after_relu = set()
        non_after_relu = set()

        # layer 1
        non_after_relu.add("conv1")

        for l in range(1, 4):
            after_relu.add(f"layer{l}.0.conv1")
            after_relu.add(f"layer{l}.0.conv2")
            after_relu.add(f"layer{l}.1.conv1")
            after_relu.add(f"layer{l}.1.conv2")
            after_relu.add(f"layer{l}.2.conv1")
            after_relu.add(f"layer{l}.2.conv2")
            # downsample
            if l in (2, 3):
                after_relu.add(f"layer{l}.0.shortcut.0")

        after_relu.add("linear")

        weight_specs = {}

        for k in after_relu:
            weight_specs[k] = IntAffineQuantizationSpec(
                nbits=bits_weight,
                signed=True,
                quant_mode=quant_mode_override or IntAffineQuantizationMode.SYMMETRIC,
                percentile=clip_percentile,
                grouper=_get_grouper(k, True),
            )

        for k in non_after_relu:
            weight_specs[k] = IntAffineQuantizationSpec(
                nbits=bits_weight,
                signed=True,
                quant_mode=quant_mode_override or IntAffineQuantizationMode.SYMMETRIC,
                percentile=clip_percentile,
                grouper=_get_grouper(k, True),
            )

        input_specs = {}

        for k in after_relu:
            input_specs[k] = IntAffineQuantizationSpec(
                nbits=bits_activation,
                signed=False,
                quant_mode=quant_mode_override or IntAffineQuantizationMode.SYMMETRIC,
                percentile=clip_percentile,
                grouper=_get_grouper(k, False),
            )

        for k in non_after_relu:
            input_specs[k] = IntAffineQuantizationSpec(
                nbits=bits_activation,
                signed=True,
                quant_mode=quant_mode_override or IntAffineQuantizationMode.SYMMETRIC,
                percentile=clip_percentile,
                grouper=_get_grouper(k, False),
            )

        if leave_edge_layers_8_bits:
            for k in ("linear", "conv1"):
                input_specs[k] = IntAffineQuantizationSpec(
                    nbits=8,
                    signed=k == "conv1",
                    quant_mode=quant_mode_override or IntAffineQuantizationMode.SYMMETRIC,
                    percentile=clip_percentile,
                    grouper=_get_grouper(k, False),
                )

                weight_specs[k] = IntAffineQuantizationSpec(
                    nbits=8,
                    signed=True,
                    quant_mode=quant_mode_override or IntAffineQuantizationMode.SYMMETRIC,
                    percentile=clip_percentile,
                    grouper=_get_grouper(k, True),
                )
        quant_specs = {}
        for k in weight_specs.keys():
            quant_specs[k] = {
                "input": input_specs[k],
                "weight": weight_specs[k],
            }

        return quant_specs
    else:
        # otherwise it does not matter and we can just do all signed
        weight_specs = {}
        input_specs = {}
        for k in get_all_resnet20_layers():
            weight_specs[k] = IntAffineQuantizationSpec(
                nbits=bits_weight,
                signed=True,
                quant_mode=quant_mode_override or IntAffineQuantizationMode.ASYMMETRIC,
                percentile=clip_percentile,
                grouper=_get_grouper(k, True),
            )
            input_specs[k] = IntAffineQuantizationSpec(
                nbits=bits_activation,
                signed=True,
                quant_mode=quant_mode_override or IntAffineQuantizationMode.ASYMMETRIC,
                percentile=clip_percentile,
                grouper=_get_grouper(k, False),
            )

        if leave_edge_layers_8_bits:
            for k in ("fc", "conv1"):
                input_specs[k] = IntAffineQuantizationSpec(
                    nbits=8,
                    signed=True,
                    quant_mode=quant_mode_override or IntAffineQuantizationMode.ASYMMETRIC,
                    percentile=clip_percentile,
                    grouper=_get_grouper(k, False),
                )
                weight_specs[k] = IntAffineQuantizationSpec(
                    nbits=8,
                    signed=True,
                    quant_mode=quant_mode_override or IntAffineQuantizationMode.ASYMMETRIC,
                    percentile=clip_percentile,
                    grouper=_get_grouper(k, True),
                )

        quant_specs = {}
        for k in weight_specs.keys():
            quant_specs[k] = {
                "input": input_specs[k],
                "weight": weight_specs[k],
            }

        return quant_specs


def get_recipe_quant(model_name: str):
    if model_name == "mobilenet_v2":
        return mobilenetv2_recipe_quant
    elif model_name == "resnet18":
        return get_resnet18_recipe_quant
    elif model_name == "resnet20":
        return get_resnet20_recipe_quant
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def get_quant_keys(model_name: str):
    # get recipe then recipe.keys()
    recp = get_recipe_quant(model_name)(
        bits_activation=8,
        bits_weight=8,
        clip_percentile=99.5,
        leave_edge_layers_8_bits=False,
        symmetric=False,
    )

    return recp.keys()


def get_generic_recipe_quant(
    model,
    bits_activation: int,
    bits_weight: int,
    clip_percentile: float,
    edge_layers: list[str] | None = None,
    leave_edge_layers_8_bits: bool = False,
    symmetric: bool = True,
    conv_weight_grouper=None,
    conv_activation_grouper=None,
    linear_weight_grouper=None,
    linear_activation_grouper=None,
):
    edge_layers = edge_layers or []
    conv_weight_grouper = conv_weight_grouper or ConvWeightsPerOutChannel()
    conv_activation_grouper = conv_activation_grouper or PerTensor()
    linear_weight_grouper = linear_weight_grouper or LinearWeightsPerOutChannel()
    linear_activation_grouper = linear_activation_grouper or PerTensor()
    mode = (
        IntAffineQuantizationMode.SYMMETRIC
        if symmetric
        else IntAffineQuantizationMode.ASYMMETRIC
    )

    conv_names, linear_names = [], []
    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.LazyConv2d)):
            conv_names.append(name)
        elif isinstance(m, nn.Linear):
            linear_names.append(name)
    first_conv = conv_names[0] if leave_edge_layers_8_bits and conv_names else None
    last_linear = (
        linear_names[-1] if leave_edge_layers_8_bits and linear_names else None
    )

    specs = {}
    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv2d, nn.LazyConv2d, nn.Linear)):
            is_conv = isinstance(m, (nn.Conv2d, nn.LazyConv2d))
            # decide bits
            use_8 = name in edge_layers or name in (first_conv, last_linear)
            w_bits = 8 if use_8 else bits_weight
            a_bits = 8 if use_8 else bits_activation

            specs[name] = {
                "weight": IntAffineQuantizationSpec(
                    nbits=w_bits,
                    signed=True,
                    quant_mode=mode,
                    percentile=clip_percentile,
                    grouper=conv_weight_grouper if is_conv else linear_weight_grouper,
                ),
                "input": IntAffineQuantizationSpec(
                    nbits=a_bits,
                    signed=True,
                    quant_mode=mode,
                    percentile=clip_percentile,
                    grouper=(
                        conv_activation_grouper
                        if is_conv
                        else linear_activation_grouper
                    ),
                ),
            }
    return specs
