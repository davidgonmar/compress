import torch
from compress.quantization.util import IntQuantizationInfo, IntQuantizationSpec
from compress.utils import dims_sub


def calibrate(
    x: torch.Tensor,
    spec: IntQuantizationSpec,
    symmetric: bool = True,
    return_z_as_int: bool = True,
):
    reduction_dims = (
        spec.group_dims if spec.group_dims is not None else list(range(x.ndim))
    )
    if not symmetric:
        xmin = x.amin(reduction_dims)
        xmax = x.amax(reduction_dims)
        scale = (xmax - xmin) / (spec.qmax - spec.qmin)
        zero_point = torch.round(spec.qmin - xmin / scale).to(x.dtype)

        shape = [1] * x.ndim
        for dim in dims_sub(list(range(x.ndim)), reduction_dims):
            shape[dim] = x.shape[dim]

        scale = scale.reshape(shape)
        zero_point = zero_point.reshape(shape)
        if return_z_as_int:
            zero_point = zero_point.to(torch.int8)
        return IntQuantizationInfo(spec, scale.detach(), zero_point.detach())
    else:
        xmax = x.abs().amax(reduction_dims)
        scale = xmax / spec.qmax
        shape = [1] * x.ndim

        for dim in dims_sub(list(range(x.ndim)), reduction_dims):
            shape[dim] = x.shape[dim]
        scale = scale.reshape(shape)
        return IntQuantizationInfo(spec, scale.detach(), None)
