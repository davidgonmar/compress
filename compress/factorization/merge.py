from compress.utils import gather_submodules
from compress.factorization.low_rank_ops import LowRankLinear, LowRankConv2d
import copy
from tqdm import tqdm
import torch.nn as nn


def merge_back(model: nn.Module, inplace=True):
    if not inplace:
        model = copy.deepcopy(model)

    def _sd(mod, name):
        return isinstance(mod, LowRankLinear) or isinstance(mod, LowRankConv2d)

    modules_to_replace = gather_submodules(
        model,
        should_do=_sd,
    )
    for name, module in tqdm(modules_to_replace, desc="Replacing modules"):
        parent_module = model
        *parent_path, attr_name = name.split(".")
        for part in parent_path:
            parent_module = getattr(parent_module, part)

        setattr(
            parent_module,
            attr_name,
            (
                LowRankLinear.to_linear(module)
                if isinstance(module, LowRankLinear)
                else (
                    LowRankConv2d.to_conv2d(module)
                    if isinstance(module, LowRankConv2d)
                    else module
                )
            ),
        )

    return model
