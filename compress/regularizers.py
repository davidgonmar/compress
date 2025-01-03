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


# Pairs (fn, sgn) where sgn is -1 if the metric should be minimized, 1 if maximized
_regularizers = {
    "entropy": lambda **kwargs: (singular_values_entropy, 1.0),
    "hoyer_sparsity": lambda **kwargs: (
        singular_values_hoyer_sparsity,
        -1.0 if kwargs.get("normalize", DEFAULT_NORMALIZE) else 1.0,
    ),
    "noop": lambda **kwargs: (lambda x, **kwargs: 0.0, 1.0),
}


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
            weight * self.fn(param, **self.kwargs)
            for param, weight in zip(self.params, self.weights)
        )
