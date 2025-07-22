from compress.quantization import IntAffineQuantizationSpec, IntAffineQuantizationMode
from compress.quantization.common import (
    PerTensor,
    ConvWeightsPerOutChannel,
    LinearWeightsPerOutChannel,
    AbstractGrouper,
)
from typing import Iterable, Mapping, Optional, Callable, Dict, Any, Union

QuantArgDict = Mapping[str, Any]
QuantArgs = Union[QuantArgDict, Mapping[str, QuantArgDict]]


def get_after_relu_mobilenetv2():
    s = {"features.1.conv.0.0", "features.1.conv.1", "classifier.1"}
    for i in range(2, 18):
        s.update({f"features.{i}.conv.1.0", f"features.{i}.conv.2"})
    return s


def get_non_after_relu_mobilenetv2():
    s = {"features.0.0", "features.18.0"}
    for i in range(2, 18):
        s.add(f"features.{i}.conv.0.0")
    return s


def build_int_affine_recipe(
    *,
    bits_activation: int,
    bits_weight: int,
    layers_after_relu: Iterable[str],
    layers_not_after_relu: Iterable[str],
    edge_layers: Optional[Iterable[str]] = None,
    leave_edge_layers_8_bits: bool = True,
    quant_args: QuantArgs = {
        "quant_mode": IntAffineQuantizationMode.SYMMETRIC,
        "percentile": 99.5,
    },
    conv_weight_grouper: AbstractGrouper = ConvWeightsPerOutChannel(),
    conv_activation_grouper: AbstractGrouper = PerTensor(),
    linear_weight_grouper: AbstractGrouper = LinearWeightsPerOutChannel(),
    linear_activation_grouper: AbstractGrouper = PerTensor(),
    get_layer_type: Callable[[str], str] = lambda n: (
        "linear" if "linear" in n else "conv"
    ),
) -> Dict[str, Dict[str, IntAffineQuantizationSpec]]:
    if any(k in quant_args for k in ("weights", "activations")):
        weight_qargs: QuantArgDict = quant_args.get("weights")
        act_qargs: QuantArgDict = quant_args.get("activations")
        assert (
            weight_qargs is not None and act_qargs is not None
        ), "weights and activations must be provided if one of them is provided"
        if not weight_qargs:
            weight_qargs = act_qargs
        if not act_qargs:
            act_qargs = weight_qargs
    else:
        weight_qargs = act_qargs = quant_args

    modes = (
        IntAffineQuantizationMode.SYMMETRIC,
        IntAffineQuantizationMode.ASYMMETRIC,
        IntAffineQuantizationMode.LSQ_INITIALIZATION,
    )
    req_extra = {
        IntAffineQuantizationMode.SYMMETRIC: ["percentile"],
        IntAffineQuantizationMode.ASYMMETRIC: ["percentile"],
        IntAffineQuantizationMode.LSQ_INITIALIZATION: [],
    }

    def _validate(args: QuantArgDict):
        if "quant_mode" not in args:
            raise ValueError("'quant_mode' missing in quant_args")
        qmode = args["quant_mode"]
        if qmode not in modes:
            raise ValueError(f"Unsupported quant_mode {qmode}")
        for k in req_extra[qmode]:
            if k not in args:
                raise ValueError(
                    f"Missing required argument {k!r} for quant_mode {qmode}"
                )
        return qmode in (
            IntAffineQuantizationMode.SYMMETRIC,
            IntAffineQuantizationMode.LSQ_INITIALIZATION,
        )

    symmetric_a = _validate(act_qargs)

    def _get_grouper(name: str, is_weight: bool):
        if get_layer_type(name) == "linear":
            return linear_weight_grouper if is_weight else linear_activation_grouper
        return conv_weight_grouper if is_weight else conv_activation_grouper

    def _spec(nbits: int, signed: bool, is_weight: bool, layer: str):
        args = weight_qargs if is_weight else act_qargs
        extra = {k: v for k, v in args.items() if k != "quant_mode"}
        return IntAffineQuantizationSpec(
            nbits=nbits,
            signed=signed,
            quant_mode=args["quant_mode"],
            grouper=_get_grouper(layer, is_weight),
            **extra,
        )

    w_specs: Dict[str, IntAffineQuantizationSpec] = {}
    i_specs: Dict[str, IntAffineQuantizationSpec] = {}

    for n in layers_after_relu:
        w_specs[n] = _spec(bits_weight, True, True, n)
        i_specs[n] = _spec(bits_activation, not symmetric_a, False, n)

    for n in layers_not_after_relu:
        w_specs[n] = _spec(bits_weight, True, True, n)
        i_specs[n] = _spec(bits_activation, True, False, n)

    if leave_edge_layers_8_bits and edge_layers:
        for n in edge_layers:
            after = n in layers_after_relu
            i_specs[n] = _spec(8, not (symmetric_a and after), False, n)
            w_specs[n] = _spec(8, True, True, n)

    return {k: {"input": i_specs[k], "weight": w_specs[k]} for k in w_specs}


def mobilenetv2_recipe_quant(
    bits_activation: int,
    bits_weight: int,
    leave_edge_layers_8_bits: bool,
    quant_args: QuantArgs = {
        "quant_mode": IntAffineQuantizationMode.SYMMETRIC,
        "percentile": 99.5,
    },
    conv_weight_grouper: AbstractGrouper = ConvWeightsPerOutChannel(),
    conv_activation_grouper: AbstractGrouper = PerTensor(),
    linear_weight_grouper: AbstractGrouper = LinearWeightsPerOutChannel(),
    linear_activation_grouper: AbstractGrouper = PerTensor(),
):
    return build_int_affine_recipe(
        bits_activation=bits_activation,
        bits_weight=bits_weight,
        layers_after_relu=get_after_relu_mobilenetv2(),
        layers_not_after_relu=get_non_after_relu_mobilenetv2(),
        edge_layers=("features.0.0", "classifier.1"),
        leave_edge_layers_8_bits=leave_edge_layers_8_bits,
        quant_args=quant_args,
        conv_weight_grouper=conv_weight_grouper,
        conv_activation_grouper=conv_activation_grouper,
        linear_weight_grouper=linear_weight_grouper,
        linear_activation_grouper=linear_activation_grouper,
        get_layer_type=lambda n: "linear" if n.startswith("classifier") else "conv",
    )


def _resnet18_after_relu():
    s = {"fc"}
    for l in range(1, 5):
        s.update(
            {
                f"layer{l}.0.conv1",
                f"layer{l}.0.conv2",
                f"layer{l}.1.conv1",
                f"layer{l}.1.conv2",
            }
        )
        if l in (2, 3, 4):
            s.add(f"layer{l}.0.downsample.0")
    return s


def resnet18_recipe_quant(
    bits_activation: int,
    bits_weight: int,
    leave_edge_layers_8_bits: bool,
    quant_args: QuantArgs = {
        "quant_mode": IntAffineQuantizationMode.SYMMETRIC,
        "percentile": 99.5,
    },
    conv_weight_grouper: AbstractGrouper = ConvWeightsPerOutChannel(),
    conv_activation_grouper: AbstractGrouper = PerTensor(),
    linear_weight_grouper: AbstractGrouper = LinearWeightsPerOutChannel(),
    linear_activation_grouper: AbstractGrouper = PerTensor(),
):
    return build_int_affine_recipe(
        bits_activation=bits_activation,
        bits_weight=bits_weight,
        layers_after_relu=_resnet18_after_relu(),
        layers_not_after_relu={"conv1"},
        edge_layers=("conv1", "fc"),
        leave_edge_layers_8_bits=leave_edge_layers_8_bits,
        quant_args=quant_args,
        conv_weight_grouper=conv_weight_grouper,
        conv_activation_grouper=conv_activation_grouper,
        linear_weight_grouper=linear_weight_grouper,
        linear_activation_grouper=linear_activation_grouper,
        get_layer_type=lambda n: "linear" if n == "fc" else "conv",
    )


def _resnet20_after_relu():
    s = {"linear"}
    for l in range(1, 4):
        s.update(
            {
                f"layer{l}.0.conv1",
                f"layer{l}.0.conv2",
                f"layer{l}.1.conv1",
                f"layer{l}.1.conv2",
                f"layer{l}.2.conv1",
                f"layer{l}.2.conv2",
            }
        )
        if l in (2, 3):
            s.add(f"layer{l}.0.shortcut.0")
    return s


def resnet20_recipe_quant(
    bits_activation: int,
    bits_weight: int,
    leave_edge_layers_8_bits: bool,
    quant_args: QuantArgs = {
        "quant_mode": IntAffineQuantizationMode.SYMMETRIC,
        "percentile": 99.5,
    },
    conv_weight_grouper: AbstractGrouper = ConvWeightsPerOutChannel(),
    conv_activation_grouper: AbstractGrouper = PerTensor(),
    linear_weight_grouper: AbstractGrouper = LinearWeightsPerOutChannel(),
    linear_activation_grouper: AbstractGrouper = PerTensor(),
):
    return build_int_affine_recipe(
        bits_activation=bits_activation,
        bits_weight=bits_weight,
        layers_after_relu=_resnet20_after_relu(),
        layers_not_after_relu={"conv1"},
        edge_layers=("conv1", "linear"),
        leave_edge_layers_8_bits=leave_edge_layers_8_bits,
        quant_args=quant_args,
        conv_weight_grouper=conv_weight_grouper,
        conv_activation_grouper=conv_activation_grouper,
        linear_weight_grouper=linear_weight_grouper,
        linear_activation_grouper=linear_activation_grouper,
    )


def get_recipe_quant(model_name: str):
    if model_name == "mobilenet_v2":
        return mobilenetv2_recipe_quant
    if model_name == "resnet18":
        return resnet18_recipe_quant
    if model_name == "resnet20":
        return resnet20_recipe_quant
    raise ValueError(f"Unknown model name: {model_name}")


def get_quant_keys(model_name: str):
    recp = get_recipe_quant(model_name)(
        bits_activation=8,
        bits_weight=8,
        leave_edge_layers_8_bits=False,
        quant_args={
            "quant_mode": IntAffineQuantizationMode.ASYMMETRIC,
            "percentile": 99.5,
        },
    )
    return recp.keys()
