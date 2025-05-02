import math
import torch


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
