from typing import List
import torch
import math

def singular_values_entropy(input: torch.Tensor) -> torch.Tensor:
    singular_values = torch.linalg.svdvals(input)
    smx = torch.nn.functional.softmax(singular_values, dim=-1)
    return -torch.sum(smx * torch.log(smx.clamp_min(1e-12)))
    

class SingularValuesEntropyRegularizer:
    def __init__(self, params: List[torch.Tensor], weights: float | List[float] = 1.0):
        super(SingularValuesEntropyRegularizer, self).__init__()
        self.params = params
        self.weights = [weights] * len(params) if isinstance(weights, float) else weights
        assert len(self.params) == len(self.weights), "Number of params and weights should match, got {} and {}".format(
            len(self.params), len(self.weights)
        )

    def __call__(self) -> torch.Tensor:
        return sum(
            weight * singular_values_entropy(param)
            for param, weight in zip(self.params, self.weights)
        )
    

def hoyer_sparsity(input: torch.Tensor) -> torch.Tensor:
    # Metric to measure sparsity of the singular values. Taken from https://arxiv.org/abs/cs/0408058
    n = input.numel()
    input = input.flatten()
    l1_norm = torch.sum(torch.abs(input))
    l2_norm = torch.norm(input, p=2)
    return math.sqrt(n) * (l1_norm / l2_norm) / math.sqrt(n - 1)

def singular_values_hoyer_sparsity(input: torch.Tensor) -> torch.Tensor:
    singular_values = torch.linalg.svdvals(input)
    return hoyer_sparsity(singular_values)

class SingularValuesHoyerSparsityRegularizer:
    def __init__(self, params: List[torch.Tensor], weights: float | List[float] = 1.0):
        super(SingularValuesHoyerSparsityRegularizer, self).__init__()
        self.params = params
        self.weights = [weights] * len(params) if isinstance(weights, float) else weights
        assert len(self.params) == len(self.weights), "Number of params and weights should match, got {} and {}".format(
            len(self.params), len(self.weights)
        )

    def __call__(self) -> torch.Tensor:
        return sum(
            weight * singular_values_hoyer_sparsity(param)
            for param, weight in zip(self.params, self.weights)
        )