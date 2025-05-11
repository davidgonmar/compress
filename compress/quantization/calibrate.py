import torch
from compress.quantization.common import (
    IntAffineQuantizationInfo,
    IntAffineQuantizationSpec,
    IntAffineQuantizationMode,
    PerTensor,
)


# ========================
# Entropy Calibration (reference https://www.cse.iitd.ac.in/~rijurekha/course/tensorrt.pdf)
# Extended for the case of symmetric quantization (negative and positive values) and other bit widths
# ========================
def kl_divergence(p: torch.Tensor, q: torch.Tensor, eps=1e-12):
    mask = p > 0
    return (p[mask] * torch.log((p[mask] + eps) / (q[mask] + eps))).sum()


def merge_to_bins(p, num_bins_out):
    """
    Vectorized version: assign each of the input bins to an output bin,
    then use index_add to sum up the contributions.
    """
    length_in = p.shape[0]
    device = p.device
    # for each index in p, compute its bin: floor(i * num_bins_out / length_in)
    indices = torch.floor(
        torch.arange(length_in, device=device, dtype=torch.float32)
        * num_bins_out
        / length_in
    ).long()
    merged = torch.zeros(num_bins_out, dtype=p.dtype, device=device)
    merged = merged.index_add(0, indices, p)
    return merged


def expand_from_bins(q, length_out):
    """
    Vectorized version: for each output index, determine from which bin of q
    its value comes and divide by the number of outputs per bin.
    """
    length_in = q.shape[0]
    device = q.device
    # for each output index, assign a bin by floor(i * length_in / length_out)
    bin_indices = torch.floor(
        torch.arange(length_out, device=device, dtype=torch.float32)
        * length_in
        / length_out
    ).long()
    # count how many times each bin appears
    counts = torch.bincount(bin_indices, minlength=length_in)
    expanded = q[bin_indices] / counts[bin_indices].to(q.dtype)
    return expanded


def entropy_calibration(hist_fp32, target_nlevels):
    """
    Iterates over candidate thresholds (i from 128 to 2048 bins)
    and finds the threshold that minimizes the KL divergence.
    """
    hist_fp32 = hist_fp32.float().cpu()
    num_bins = hist_fp32.shape[0]

    best_i = None
    min_kl_value = float("inf")
    # print(hist_fp32.tolist())
    for i in range(target_nlevels, num_bins + 1):
        P = hist_fp32[:i].clone()
        P_T = merge_to_bins(P, target_nlevels)
        P_T /= P_T.sum() + 1e-12
        Q = expand_from_bins(P_T, i)
        Q /= Q.sum() + 1e-12

        if i < num_bins:
            P[-1] += hist_fp32[i:].sum()

        P /= P.sum() + 1e-12

        kl_val = kl_divergence(P, Q)

        if kl_val < min_kl_value:
            min_kl_value = kl_val
            best_i = i

    return best_i, min_kl_value


# approximate quantile (faster and does not throw error if tensor is big)
def quantile(tensor, q, dim=None, keepdim=False):
    assert 0 <= q <= 1, "\n\nquantile value should be a float between 0 and 1.\n\n"
    if dim is None:
        tensor = tensor.flatten()
        dim = 0
    sorted_tensor, _ = torch.sort(tensor, dim=dim)
    num_elements = sorted_tensor.size(dim)
    index = q * (num_elements - 1)
    lower_index = int(index)
    upper_index = min(lower_index + 1, num_elements - 1)
    lower_value = sorted_tensor.select(dim, lower_index)
    upper_value = sorted_tensor.select(dim, upper_index)
    weight = index - lower_index
    quantile_value = (1 - weight) * lower_value + weight * upper_value
    return quantile_value.unsqueeze(dim) if keepdim else quantile_value


def calibrate(
    x: torch.Tensor,
    spec: IntAffineQuantizationSpec,
):
    if spec.quant_mode in [
        IntAffineQuantizationMode.STATISTICS_AWARE_BINNING_ASYMMETRIC,
        IntAffineQuantizationMode.STATISTICS_AWARE_BINNING_SYMMETRIC,
    ]:
        assert (
            spec.quant_mode
            == IntAffineQuantizationMode.STATISTICS_AWARE_BINNING_SYMMETRIC
        ), "Only symmetric binning is supported"
        nbin = 2**spec.nbits
        w_flat = spec.grouper.group(x)

        E_abs_w = w_flat.abs().mean(0)  # E[|w|]
        E_w2 = (w_flat**2).mean(0)  # E[w^2]

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
        scale = alpha_star
        zero_point = None
        return IntAffineQuantizationInfo(spec, scale.detach(), zero_point)

    if spec.quant_mode == IntAffineQuantizationMode.ENTROPY_SYMMETRIC:
        assert (
            spec.grouper is PerTensor
        ), "Entropy calibration only supports per-tensor quantization atm"
        assert spec.signed, "Entropy calibration only supports signed quantization atm"
        hist_fp32 = (
            x.float()
            .abs()
            .histc(bins=2048, min=x.abs().min().item(), max=x.abs().max().item())
        )
        best_i, min_kl_value = entropy_calibration(
            hist_fp32, target_nlevels=2 ** (spec.nbits - 1)
        )
        # get the percentile of the best i
        min_val = 0  # ideally, for absolute values
        max_val = x.abs().max().item()
        bin_width = (max_val - min_val) / 2048
        threshold = (best_i + 0.5) * bin_width
        scale = 2 * threshold / (spec.qmax - spec.qmin)
        zero_point = None
        return IntAffineQuantizationInfo(
            spec, torch.tensor(scale).to(x.device).detach(), zero_point
        )

    assert "percentile" in spec.mode_args, "percentile not in mode_args"
    percentile = spec.mode_args["percentile"]

    if spec.quant_mode == IntAffineQuantizationMode.ASYMMETRIC:
        xm = spec.grouper.group(x)
        lower_percentile = (1 - percentile) / 2
        upper_percentile = (1 + percentile) / 2
        xmin = quantile(xm, lower_percentile, dim=0)  # shape (n_groups)
        xmax = quantile(xm, upper_percentile, dim=0)
        scale = (xmax - xmin) / (spec.qmax - spec.qmin)
        zero_point = torch.round(spec.qmin - xmin / scale).to(x.dtype)
        return IntAffineQuantizationInfo(spec, scale.detach(), zero_point.detach())

    if spec.quant_mode == IntAffineQuantizationMode.SYMMETRIC:
        xm = spec.grouper.group(x)
        # print(x.max(), x.min())
        xmax = quantile(xm.abs(), percentile, dim=0)
        scale = 2 * xmax / (spec.qmax - spec.qmin)
        zero_point = None

        return IntAffineQuantizationInfo(spec, scale.detach(), zero_point)
