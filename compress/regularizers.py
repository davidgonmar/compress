from typing import List
import torch
import math


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


def singular_values_hoyer_sparsity(
    input: torch.Tensor, normalize=DEFAULT_NORMALIZE
) -> torch.Tensor:
    singular_values = torch.linalg.svdvals(input)
    return hoyer_sparsity(singular_values, normalize)


_reductions = {
    "sum": torch.sum,
    "mean": torch.mean,
    "none": lambda x: x,
    None: lambda x: x,
}


def scad(
    input: torch.Tensor, lambda_val: float, a: float, reduction: str = "sum"
) -> torch.Tensor:
    # https://andrewcharlesjones.github.io/journal/scad.html
    abs_input = torch.abs(input)
    # Case 1: |x| <= λ
    case1 = lambda_val * abs_input * (abs_input <= lambda_val)

    # Case 2: λ < |x| <= aλ
    case2 = (
        (-(abs_input**2) + 2 * a * lambda_val * abs_input - lambda_val**2)
        / (2 * (a - 1))
    ) * ((abs_input > lambda_val) & (abs_input <= a * lambda_val))

    # Case 3: |x| > aλ
    case3 = ((a + 1) * (lambda_val**2) / 2) * (abs_input > a * lambda_val)
    penalty = case1 + case2 + case3
    return _reductions[reduction](penalty)


def singular_values_scad(
    input: torch.Tensor, lambda_val: float, a_val: float, reduction: str = "sum"
) -> torch.Tensor:
    singular_values = torch.linalg.svdvals(input)
    return scad(singular_values, lambda_val, a_val, reduction)


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
    "noop": lambda **kwargs: (lambda x, **kwargs: 0.0, 1.0),
}


# For Conv2d weights, we have shape (OUT, IN, H, W)
# We reshape them into (OUT, IN * H * W) to compute the singular values
class SingularValuesRegularizer:
    def __init__(
        self,
        *,
        metric: str,
        params: List[torch.Tensor],
        weights: float | List[float] = 1.0,
        **kwargs
    ):
        super(SingularValuesRegularizer, self).__init__()
        self.params = params
        self.weights = (
            [weights] * len(params) if isinstance(weights, float) else weights
        )
        assert len(self.params) == len(
            self.weights
        ), "Number of params and weights should match, got {} and {}".format(
            len(self.params), len(self.weights)
        )
        self.metric = metric
        self.kwargs = kwargs

        self.fn, self.sgn = _regularizers[metric](**kwargs)

    def __call__(self) -> torch.Tensor:
        return self.sgn * sum(
            weight * self.fn(param.reshape(param.shape[0], -1), **self.kwargs)
            for param, weight in zip(self.params, self.weights)
        )
