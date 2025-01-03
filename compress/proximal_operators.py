from typing import List
import torch

def nuclear_norm_proximal_op(input, lr, multiplier):
    U, S, V = torch.svd(input)
    S_opt = torch.clamp(S - lr * multiplier, min=0)
    desired_param = U @ S_opt.diag() @ V.t()
    return desired_param


class NuclearNormProximalOpApplier:
    def __init__(self, params: List[torch.Tensor], lr: float, multiplier: float):
        self.params = params
        self.lr = lr
        self.multiplier = multiplier

    def __call__(self):
        for param in self.params:
            with torch.no_grad():
                param.copy_(nuclear_norm_proximal_op(param, self.lr, self.multiplier))