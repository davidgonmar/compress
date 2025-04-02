import torch
from compress.quantization.common import (
    IntAffineQuantizationInfo,
    IntAffineQuantizationSpec,
    IntAffineQuantizationMode,
)


def statistics_aware_weight_binning_scale(x, n_bits):
    # from https://arxiv.org/abs/1807.06964

    nbin = 2**n_bits
    w_flat = x.view(-1)

    E_abs_w = w_flat.abs().mean()  # E[|w|]
    E_w2 = (w_flat**2).mean()  # E[w^2]

    c_map = {
        2: (0.0, +1.0),  # alpha^* = E|w|
        3: (2.587, -1.693),  # alpha^* = 2.587*sqrt(E[w^2]) - 1.693*E|w|
        4: (3.212, -2.178),  # alpha^* = 3.212*sqrt(E[w^2]) - 2.178*E|w|
    }
    if nbin not in c_map:
        raise ValueError(
            f"nbin={nbin} not in {list(c_map.keys())}.  "
            f"Only 2,3,4 bins are supported by this demo function."
        )

    c1, c2 = c_map[nbin]
    alpha_star = c1 * torch.sqrt(E_w2) + c2 * E_abs_w

    return alpha_star


def calibrate(
    x: torch.Tensor,
    spec: IntAffineQuantizationSpec,
):
    assert (
        spec.group_dims is None or len(spec.group_dims) == []
    ), "Group dims not supported"
    if spec.quant_mode in [
        IntAffineQuantizationMode.STATISTICS_AWARE_BINNING_ASYMMETRIC,
        IntAffineQuantizationMode.STATISTICS_AWARE_BINNING_SYMMETRIC,
    ]:
        assert (
            spec.quant_mode
            == IntAffineQuantizationMode.STATISTICS_AWARE_BINNING_SYMMETRIC
        ), "Only symmetric binning is supported"
        scale = statistics_aware_weight_binning_scale(x, spec.nbits)
        zero_point = None
        return IntAffineQuantizationInfo(spec, scale.detach(), zero_point)

    if spec.quant_mode == IntAffineQuantizationMode.ASYMMETRIC:
        # just regular min/max calibration
        xmin = x.amin()
        xmax = x.amax()
        scale = (xmax - xmin) / (spec.qmax - spec.qmin)
        zero_point = torch.round(spec.qmin - xmin / scale).to(x.dtype)
        return IntAffineQuantizationInfo(spec, scale.detach(), zero_point.detach())

    if spec.quant_mode == IntAffineQuantizationMode.SYMMETRIC:
        # just regular amax calibration
        assert spec.signed, "Symmetric quantization only supports signed quantization"
        xmax = x.abs().amax()
        scale = 2 * xmax / (spec.qmax - spec.qmin)
        zero_point = None
        return IntAffineQuantizationInfo(spec, scale.detach(), zero_point)
