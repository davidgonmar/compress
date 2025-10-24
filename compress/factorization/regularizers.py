import torch
from compress.functional import (
    l1_l2_ratio,
    squared_l1_l2_ratio,
    DEFAULT_NORMALIZE,
    scad,
)
from typing import List, Callable
from compress.factorization.low_rank_ops import LowRankLinear, LowRankConv2d
from compress.utils import extract_weights
import torch.nn as nn


Reshaper = Callable[[torch.Tensor], torch.Tensor]


class SafeSvals(torch.autograd.Function):
    """
    Computes the singular values of a matrix, and handles the case where some singular values are too close
    (which means that the respective subspaces are close to be swappable) by setting the derivatives to 0.
    """

    @staticmethod
    def forward(ctx, input: torch.Tensor, threshold: float = 1e-6) -> torch.Tensor:
        assert not input.is_complex(), "Complex tensors are not supported"
        assert input.ndim == 2, "Only matrices are supported"
        U, S, Vh = torch.linalg.svd(input, full_matrices=False)
        ctx.save_for_backward(U, S, Vh)
        ctx.threshold = threshold
        return S

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        # grad output of shape (r)
        U, S, Vt = ctx.saved_tensors

        # mask = neighbor (left or right) singular values are too close
        ill = torch.abs(S[:-1] - S[1:]) < ctx.threshold
        padleft = torch.zeros(1, device=ill.device, dtype=ill.dtype)
        padright = torch.zeros(1, device=ill.device, dtype=ill.dtype)
        illleft = torch.cat((padleft, ill), dim=0)
        illright = torch.cat((ill, padright), dim=0)
        ill = illleft | illright

        # dont take into account singular values that are too close
        grad_output = torch.where(ill, torch.zeros_like(grad_output), grad_output)

        # masking is implicit for U and Vt because of the einsum
        return torch.einsum("k,ik,kj->ij", grad_output, U, Vt)


safe_svals = SafeSvals.apply


def singular_values_l1_l2_ratio(
    input: torch.Tensor, normalize=DEFAULT_NORMALIZE
) -> torch.Tensor:
    singular_values = safe_svals(input)
    return l1_l2_ratio(singular_values, normalize)


def singular_values_squared_l1_l2_ratio(
    input: torch.Tensor, normalize=DEFAULT_NORMALIZE
) -> torch.Tensor:
    singular_values = safe_svals(input)
    return squared_l1_l2_ratio(singular_values, normalize)


def singular_values_scad(
    input: torch.Tensor, lambda_val: float, a_val: float, reduction: str = "sum"
) -> torch.Tensor:
    singular_values = safe_svals(input)
    return scad(singular_values, lambda_val, a_val, reduction)


def nuclear_norm(matrix: torch.Tensor) -> torch.Tensor:
    return torch.norm(matrix, p="nuc")


def singular_values_entropy(input: torch.Tensor) -> torch.Tensor:
    singular_values = safe_svals(input)
    smx = torch.nn.functional.softmax(singular_values, dim=-1)
    return -torch.sum(smx * torch.log(smx.clamp_min(1e-12)))


def orthogonal_regularizer(
    matrix: torch.Tensor, normalize_by_numel=True
) -> torch.Tensor:
    numel = matrix.shape[0] * matrix.shape[1]
    ret = (
        torch.norm(
            torch.mm(matrix, matrix.t()) - torch.eye(matrix.shape[0]).to(matrix.device),
            p="fro",
        )
        ** 2
    )
    return ret / numel if normalize_by_numel else ret


# Pairs (fn, sgn) where sgn is 1 if the metric should be minimized, -1 if maximized
_regularizers = {
    "entropy": lambda **kwargs: (lambda x: singular_values_entropy(x), 1.0),
    "l1_l2_ratio": lambda **kwargs: (
        lambda x: singular_values_l1_l2_ratio(
            x, kwargs.get("normalize", DEFAULT_NORMALIZE)
        ),
        -1.0 if kwargs.get("normalize", DEFAULT_NORMALIZE) else 1.0,
    ),
    "scad": lambda **kwargs: (
        lambda x: singular_values_scad(
            x, kwargs["lambda_val"], kwargs["a_val"], kwargs.get("reduction", "sum")
        ),
        1.0,
    ),
    "squared_l1_l2_ratio": lambda **kwargs: (
        lambda x: singular_values_squared_l1_l2_ratio(
            x, kwargs.get("normalize", DEFAULT_NORMALIZE)
        ),
        -1.0 if kwargs.get("normalize", DEFAULT_NORMALIZE) else 1.0,
    ),
    "nuclear_norm": lambda **kwargs: (
        lambda x: nuclear_norm(x),
        1.0,
    ),
    "noop": lambda **kwargs: (lambda x: torch.tensor(0.0), 1.0),
}


# For Conv2d weights, we have shape (OUT, IN, H, W)
# We reshape them into (OUT, IN * H * W) to compute the singular values
class SingularValuesRegularizer:
    def __init__(
        self,
        *,
        metric: str,
        params_and_reshapers: List[tuple[torch.Tensor, Reshaper]],
        weights: float | List[float] = 1.0,
        **kwargs
    ):
        super(SingularValuesRegularizer, self).__init__()
        self.params_and_reshapers = params_and_reshapers

        self.weights = (
            [weights] * len(params_and_reshapers)
            if isinstance(weights, float)
            else weights
        )
        assert len(self.params_and_reshapers) == len(
            self.weights
        ), "Number of params and weights should match, got {} and {}".format(
            len(self.params), len(self.weights)
        )
        self.metric = metric
        self.kwargs = kwargs

        self.fn, self.sgn = _regularizers[metric](**kwargs)

    def __call__(self) -> torch.Tensor:
        return self.sgn * sum(
            weight * self.fn(reshaper(param), **self.kwargs)
            for (param, reshaper), weight in zip(
                self.params_and_reshapers, self.weights
            )
        )


class OrthogonalRegularizer:
    def __init__(
        self,
        params: List[List[tuple[str, torch.nn.Module]]],
        weights: float | List[float] = 1.0,
        normalize_by_numel=True,
    ):
        self.params = params
        self.weights = (
            [weights] * len(params) if isinstance(weights, float) else weights
        )
        assert len(self.params) == len(
            self.weights
        ), "Number of params and weights should match, got {} and {}".format(
            len(self.params), len(self.weights)
        )
        self.normalize_by_numel = normalize_by_numel

    @staticmethod
    def apply_to_low_rank_modules(model, weights=1.0, normalize_by_numel=True):
        params = extract_weights(
            model,
            cls_list=(LowRankLinear, LowRankConv2d),
            additional_check=lambda module: hasattr(
                module, "keep_singular_values_separated"
            )
            and module.keep_singular_values_separated,
            keywords={"w0", "w1"},
            ret_module=True,
        )
        return OrthogonalRegularizer(params, weights, normalize_by_numel)

    def __call__(self) -> torch.Tensor:
        real_params = []
        for param in self.params:
            (name, module), w = param
            kw = "w0" if "w0" in name else "w1" if "w1" in name else None
            assert kw is not None or not isinstance(module, LowRankConv2d), (
                str(kw) + " " + str(module) + " " + str(name)
            )
            if isinstance(module, LowRankConv2d):
                real_params.append(module.get_weights_as_matrices(w, kw))
            else:
                real_params.append(w)
        params = real_params
        return sum(
            weight * orthogonal_regularizer(param, self.normalize_by_numel)
            for param, weight in zip(params, self.weights)
        )


def default_tensor_to_matrix_reshape(tensor: torch.Tensor) -> torch.Tensor:
    assert tensor.dim() == 2, "Expected 2D tensor, got {}".format(tensor.shape)
    return tensor


def conv2d_to_matrix_reshape(tensor: torch.Tensor) -> torch.Tensor:
    assert tensor.dim() == 4
    o, i, h, w = tensor.shape
    return tensor.permute(1, 2, 3, 0).reshape(i * h * w, o)


_module_to_reshaper = {
    (nn.Linear, "weight"): default_tensor_to_matrix_reshape,
    (nn.LazyLinear, "weight"): default_tensor_to_matrix_reshape,
    (nn.Conv2d, "weight"): conv2d_to_matrix_reshape,
}


def extract_weights_and_reshapers(
    model, cls_list, additional_check=lambda *args: True, keywords="weight"
):
    params = extract_weights(
        model, cls_list, additional_check, keywords, ret_module=True
    )
    modules_and_names = [(name, module) for (name, module), param in params]
    reshapers_status = [
        (
            (module.__class__, name.split(".")[-1]),
            (module.__class__, name.split(".")[-1]) in _module_to_reshaper,
        )
        for name, module in modules_and_names
    ]

    if all(status for _, status in reshapers_status):
        print("Found reshapers for all modules.")
    else:
        found_reshapers = [
            module_info for module_info, status in reshapers_status if status
        ]
        not_found_reshapers = [
            module_info for module_info, status in reshapers_status if not status
        ]
        raise ValueError(
            "Cannot find reshaper for all modules. Found reshapers for: {}. Not found for: {}".format(
                found_reshapers, not_found_reshapers
            )
        )

    return [
        (param, _module_to_reshaper[(module.__class__, name.split(".")[-1])])
        for (name, module), param in params
    ]
