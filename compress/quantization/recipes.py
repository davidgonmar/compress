from compress.quantization import (
    IntAffineQuantizationSpec,
    IntAffineQuantizationMode,
)

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


def mobilenetv2_recipe_symmetric_quant(
    bits_activation: int,
    bits_weight: int,
    clip_percentile: float,
    leave_edge_layers_8_bits: bool,
):
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
