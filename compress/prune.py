from typing import Callable
from torch import nn
from compress.pruned_ops import PrunedLinear, PrunedConv2d
from tqdm import tqdm
from compress.pruning_strats import (
    UnstructuredGranularityLinear,
    UnstructuredGranularityConv2d,
)


def default_should_do(module: nn.Module, full_name: str):
    return (
        isinstance(module, nn.Linear)
        or isinstance(module, nn.LazyLinear)
        or isinstance(module, nn.Conv2d)
    )


def _to_low_rank_recursive(model: nn.Module, should_do: Callable, prefix=""):
    modules_to_replace = []
    for name, module in model.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(module, nn.Linear):
            if should_do(module, full_name):
                modules_to_replace.append((full_name, module))
        elif isinstance(module, nn.Conv2d):
            if should_do(module, full_name):
                modules_to_replace.append((full_name, module))
        elif isinstance(module, nn.Sequential):
            for idx, sub_module in enumerate(module):
                sub_full_name = f"{full_name}.{idx}"
                if should_do(sub_module, sub_full_name):
                    modules_to_replace.append((sub_full_name, sub_module))
                else:
                    modules_to_replace.extend(
                        _to_low_rank_recursive(
                            sub_module, should_do, prefix=sub_full_name
                        )
                    )
        else:
            modules_to_replace.extend(
                _to_low_rank_recursive(module, should_do, prefix=full_name)
            )
    return modules_to_replace


_module_to_pruned = {
    nn.Linear: PrunedLinear.from_linear,
    nn.LazyLinear: PrunedLinear.from_linear,
    nn.Conv2d: PrunedConv2d.from_conv2d,
}


class PruningPolicy:
    def __init__(self, cfg=dict()):
        self.cfg = cfg

    def get_granularity(self, module: nn.Module):
        cfg = self.cfg
        if isinstance(module, (nn.Linear, nn.LazyLinear)):
            return cfg.get(nn.Linear, UnstructuredGranularityLinear)
        if isinstance(module, nn.Conv2d):
            return cfg.get(nn.Conv2d, UnstructuredGranularityConv2d)


def to_pruned(
    model: nn.Module,
    policy: PruningPolicy,
    should_do: Callable = default_should_do,
    inplace=True,
    to_sparse_semistructured=False,
    **kwargs,
):
    modules_to_replace = _to_low_rank_recursive(model, should_do=should_do, prefix="")
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
            _module_to_pruned[type(module)](
                module, granularity_cls=policy.get_granularity(module), **kwargs
            ),
        )
        if to_sparse_semistructured:  # only linear at the moment
            try:
                if hasattr(
                    getattr(parent_module, attr_name), "to_sparse_semi_structured"
                ):
                    setattr(
                        parent_module,
                        attr_name,
                        getattr(parent_module, attr_name).to_sparse_semi_structured(),
                    )
                    print(f"Converted {attr_name} to sparse semi-structured")
            except Exception as e:
                print(
                    f"Failed to convert {attr_name} to sparse semi-structured: {e}, continuing"
                )
    return model
