import torch.nn as nn
from typing import Callable
from tqdm import tqdm
from compress.low_rank_ops import LowRankLinear, LowRankConv2d
from compress.common import gather_submodules, default_should_do


def to_low_rank(
    model: nn.Module, should_do: Callable = default_should_do, inplace=True, **kwargs
):
    modules_to_replace = gather_submodules(model, should_do=should_do, prefix="")
    if not inplace:
        model_initializer = kwargs.pop("model_initializer", None)
        assert (
            model_initializer is not None
        ), "model_initializer must be provided if inplace=False"
        model_ = model_initializer()
        model_.load_state_dict(model.state_dict())
        model = model_

    for name, module in tqdm(modules_to_replace, desc="Replacing modules"):
        parent_module = model
        *parent_path, attr_name = name.split(".")
        for part in parent_path:
            parent_module = getattr(parent_module, part)

        setattr(
            parent_module,
            attr_name,
            LowRankLinear.from_linear(module, **kwargs)
            if isinstance(module, nn.Linear) or isinstance(module, nn.LazyLinear)
            else LowRankConv2d.from_conv2d(module, **kwargs),
        )

    return model
