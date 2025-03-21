from typing import Callable
from torch import nn
from compress.low_rank_ops import LowRankConv2d, LowRankLinear


def default_should_do(module: nn.Module, full_name: str):
    return (
        isinstance(module, nn.Linear)
        or isinstance(module, nn.Conv2d)
        or isinstance(module, nn.LazyLinear)
    )


def gather_submodules(model: nn.Module, should_do: Callable, prefix=""):
    mods = []
    for name, module in model.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(module, nn.Linear):
            if should_do(module, full_name):
                mods.append((full_name, module))
        elif isinstance(module, nn.Conv2d):
            if should_do(module, full_name):
                mods.append((full_name, module))
        elif isinstance(module, LowRankConv2d):
            if should_do(module, full_name):
                mods.append((full_name, module))
        elif isinstance(module, LowRankLinear):
            if should_do(module, full_name):
                mods.append((full_name, module))
        elif isinstance(module, nn.Sequential):
            for idx, sub_module in enumerate(module):
                sub_full_name = f"{full_name}.{idx}"
                if should_do(sub_module, sub_full_name):
                    mods.append((sub_full_name, sub_module))
                else:
                    mods.extend(
                        gather_submodules(sub_module, should_do, prefix=sub_full_name)
                    )
        else:
            mods.extend(gather_submodules(module, should_do, prefix=full_name))
    return mods
