from typing import List
import torch
import math
from compress.low_rank_ops import LowRankLinear, LowRankConv2d
from compress.utils import extract_weights
from typing import Callable
import torch.nn as nn
from compress.pruning_strats import (
    PruningGranularity,
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


Reshaper = Callable[[torch.Tensor], torch.Tensor]


def singular_values_entropy(input: torch.Tensor) -> torch.Tensor:
    singular_values = torch.linalg.svdvals(input)
    smx = torch.nn.functional.softmax(singular_values, dim=-1)
    return -torch.sum(smx * torch.log(smx.clamp_min(1e-12)))


DEFAULT_NORMALIZE = True


def hoyer_sparsity(input: torch.Tensor, normalize=DEFAULT_NORMALIZE) -> torch.Tensor:
    # Metric to measure sparsity of the singular values. Taken from https://arxiv.org/abs/cs/0408058.
    # The paper introduces it in its normalized form (from 0 to 1, where 1 is the most sparse), but it can also be used in its non-normalized form (R^+, where 0 is the most sparse).
    n = input.numel()
    input = input.flatten()
    l1_norm = torch.sum(torch.abs(input))
    l2_norm = torch.norm(input, p=2) + 1e-12
    return (
        (math.sqrt(n) - (l1_norm / l2_norm)) / (math.sqrt(n) - 1)
        if normalize
        else (l2_norm / l1_norm)
    )


def squared_hoyer_sparsity(
    input: torch.Tensor, normalize=DEFAULT_NORMALIZE
) -> torch.Tensor:
    return hoyer_sparsity(input, normalize) ** 2


def singular_values_hoyer_sparsity(
    input: torch.Tensor, normalize=DEFAULT_NORMALIZE
) -> torch.Tensor:
    singular_values = torch.linalg.svdvals(input)
    return hoyer_sparsity(singular_values, normalize)


def singular_values_squared_hoyer_sparsity(
    input: torch.Tensor, normalize=DEFAULT_NORMALIZE
) -> torch.Tensor:
    singular_values = torch.linalg.svdvals(input)
    return squared_hoyer_sparsity(singular_values, normalize)


_reductions = {
    "sum": torch.sum,
    "mean": torch.mean,
    "none": lambda x: x,
    None: lambda x: x,
}


def scad(
    input: torch.Tensor, lambda_val: float, a_val: float, reduction: str = "mean"
) -> torch.Tensor:
    # https://andrewcharlesjones.github.io/journal/scad.html
    abs_input = torch.abs(input)
    # Case 1: |x| <= λ
    case1 = lambda_val * abs_input * (abs_input <= lambda_val)

    # Case 2: λ < |x| <= aλ
    case2 = (
        (-(abs_input**2) + 2 * a_val * lambda_val * abs_input - lambda_val**2)
        / (2 * (a_val - 1))
    ) * ((abs_input > lambda_val) & (abs_input <= a_val * lambda_val))

    # Case 3: |x| > aλ
    case3 = ((a_val + 1) * (lambda_val**2) / 2) * (abs_input > a_val * lambda_val)
    penalty = case1 + case2 + case3
    return _reductions[reduction](penalty)


def singular_values_scad(
    input: torch.Tensor, lambda_val: float, a_val: float, reduction: str = "sum"
) -> torch.Tensor:
    singular_values = torch.linalg.svdvals(input)
    return scad(singular_values, lambda_val, a_val, reduction)


def nuclear_norm_regularizer(matrix: torch.Tensor) -> torch.Tensor:
    return torch.norm(matrix, p="nuc")


def approximated_hoyer_sparsity_regularizer(
    matrix: torch.Tensor, normalize: bool
) -> torch.Tensor:
    # Approximation of the Hoyer sparsity metric without computing the singular values
    # the schatten-2 norm is equivalent to the frobenius norm
    # the schatten-1 norm is approximated by sum of row norms

    # So this seems to induce low rank, but idk why (found it by mistake)
    nom = torch.norm(matrix, p=1)
    denom = torch.norm(matrix, p=2)

    return (
        (math.sqrt(matrix.shape[0]) - (nom / denom)) / (math.sqrt(matrix.shape[0]) - 1)
        if normalize
        else nom / denom
    )


# Pairs (fn, sgn) where sgn is -1 if the metric should be minimized, 1 if maximized
_regularizers = {
    "entropy": lambda **kwargs: (lambda x, **kwargs: singular_values_entropy(x), -1.0),
    "hoyer_sparsity": lambda **kwargs: (
        lambda x, **kwargs: singular_values_hoyer_sparsity(
            x, kwargs.get("normalize", DEFAULT_NORMALIZE)
        ),
        -1.0 if kwargs.get("normalize", DEFAULT_NORMALIZE) else 1.0,
    ),
    "scad": lambda **kwargs: (
        lambda x, **kwargs: singular_values_scad(
            x, kwargs["lambda_val"], kwargs["a_val"], kwargs.get("reduction", "sum")
        ),
        -1.0,
    ),
    "squared_hoyer_sparsity": lambda **kwargs: (
        lambda x, **kwargs: singular_values_squared_hoyer_sparsity(
            x, kwargs.get("normalize", DEFAULT_NORMALIZE)
        ),
        -1.0 if kwargs.get("normalize", DEFAULT_NORMALIZE) else 1.0,
    ),
    "nuclear_norm": lambda **kwargs: (
        lambda x, **kwargs: nuclear_norm_regularizer(x),
        1.0,
    ),
    "approximated_hoyer_sparsity": lambda **kwargs: (
        lambda x, **kwargs: approximated_hoyer_sparsity_regularizer(
            x, kwargs.get("normalize", DEFAULT_NORMALIZE)
        ),
        1.0 if kwargs.get("normalize", DEFAULT_NORMALIZE) else -1.0,
    ),
    "noop": lambda **kwargs: (lambda x, **kwargs: torch.tensor(0.0), 1.0),
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


def orthogonal_regularizer(
    matrix: torch.Tensor, normalize_by_rank_squared=True
) -> torch.Tensor:
    mat_rank = min(matrix.shape[0], matrix.shape[1])
    ret = (
        torch.norm(
            torch.mm(matrix, matrix.t()) - torch.eye(matrix.shape[0]).to(matrix.device),
            p="fro",
        )
        ** 2
    )
    return ret / mat_rank if normalize_by_rank_squared else ret


class OrthogonalRegularizer:
    def __init__(
        self,
        params: List[List[tuple[str, torch.nn.Module]]],
        weights: float | List[float] = 1.0,
        normalize_by_rank_squared=True,
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
        self.normalize_by_rank_squared = normalize_by_rank_squared

    @staticmethod
    def apply_to_low_rank_modules(model, weights=1.0, normalize_by_rank_squared=True):
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
        return OrthogonalRegularizer(params, weights, normalize_by_rank_squared)

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
            weight * orthogonal_regularizer(param, self.normalize_by_rank_squared)
            for param, weight in zip(params, self.weights)
        )


# METRICS APPLIED TO A SINGLE GROUP
_params_metrics = {
    "hoyer_sparsity": lambda **kwargs: (
        lambda x, **kwargs: hoyer_sparsity(x, **kwargs),
        -1.0,
    ),
    "scad": lambda **kwargs: (lambda x, **kwargs: scad(x, **kwargs), -1.0),
    "squared_hoyer_sparsity": lambda **kwargs: (
        lambda x, **kwargs: squared_hoyer_sparsity(x, **kwargs),
        -1.0,
    ),
    "noop": lambda **kwargs: (lambda x, **kwargs: torch.tensor(0.0), 1.0),
}


def extract_weights_and_pruning_granularities(
    model, cls_list, cfg, additional_check=lambda *args: True, keywords="weight"
):
    params = extract_weights(
        model, cls_list, additional_check, keywords, ret_module=True
    )
    modules_and_names = [(name, module) for (name, module), param in params]
    granul_status = [
        (
            (module.__class__, name.split(".")[-1]),
            (module.__class__, name.split(".")[-1]) in cfg,
        )
        for name, module in modules_and_names
    ]

    if all(status for _, status in granul_status):
        print("Found pruning granularities for all modules.")
    else:
        found = [module_info for module_info, status in granul_status if status]
        not_found = [module_info for module_info, status in granul_status if not status]
        raise ValueError(
            "Cannot find pruning granularities for all modules. Found pruning granularities for: {}. Not found for: {}".format(
                found, not_found
            )
        )

    return [
        (param, cfg[(module.__class__, name.split(".")[-1])]())
        for (name, module), param in params
    ]


class SparsityRegularizer:
    def __init__(
        self,
        metric: str,
        params_and_pruning_granularities: List[tuple[torch.Tensor, PruningGranularity]],
        weights: float | List[float] = 1.0,
        mode: str = "inside_group",
        **kwargs
    ):
        self.params_and_pruning_granularities = params_and_pruning_granularities
        self.weights = (
            [weights] * len(params_and_pruning_granularities)
            if isinstance(weights, float)
            else weights
        )
        assert len(self.params_and_pruning_granularities) == len(
            self.weights
        ), "Number of params and weights should match, got {} and {}".format(
            len(self.params), len(self.weights)
        )
        self.metric = metric
        self.kwargs = kwargs

        self.fn, self.sgn = _params_metrics[metric](**kwargs)
        self.mode = mode
        # granul.transform maps the weights to a tensor of shape (n_groups, m_elements_per_group), so
        # the metric can be obtained by doing mean over dim=1.

    # tries to prune entire groups
    def call_per_group(self) -> torch.Tensor:
        return self.sgn * sum(
            weight
            * self.fn(granul.transform(param).sum(1, keepdim=False), **self.kwargs)
            for (param, granul), weight in zip(
                self.params_and_pruning_granularities, self.weights
            )
        )

    # inside each group, applies the sparsity metric locally
    def call_inside_group(self) -> torch.Tensor:
        return self.sgn * sum(
            weight
            * torch.vmap(lambda x: self.fn(x, **self.kwargs))(
                granul.transform(param)
            ).mean()
            for (param, granul), weight in zip(
                self.params_and_pruning_granularities, self.weights
            )
        )

    def __call__(self) -> torch.Tensor:
        if self.mode == "inside_group":
            return self.call_inside_group()
        elif self.mode == "per_group":
            return self.call_per_group()


def update_weights(reg, weights):
    weights = (
        [weights]
        * len(
            reg.params_and_pruning_granularities
            if hasattr(reg, "params_and_pruning_granularities")
            else reg.params if hasattr(reg, "params") else reg.params_and_reshapers
        )
        if isinstance(weights, float)
        else weights
    )
    if isinstance(reg, SingularValuesRegularizer):
        reg.weights = weights
    elif isinstance(reg, OrthogonalRegularizer):
        reg.weights = weights
    elif isinstance(reg, SparsityRegularizer):
        reg.weights = weights
    else:
        raise ValueError("Unknown regularizer type: {}".format(type(reg)))
    return reg
