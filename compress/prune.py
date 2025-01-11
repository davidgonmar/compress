from typing import Callable
from torch import nn
from compress.pruned_ops import PrunedLinear, PrunedConv2d
from tqdm import tqdm


# pruning granularity spec accepts a ndim-tensor and returns a reshaped tensor of the form (n, m), n denoting the number of groups and m denoting the number of elements in each group
class PruningGranularity:
    ndim: int

    def transform(self, tensor):
        raise NotImplementedError


class UnstructuredGranularityLinear(PruningGranularity):
    ndim = 2

    def transform(self, tensor):
        return tensor.reshape(1, -1)


class UnstructuredGranularityConv2d(PruningGranularity):
    ndim = 4

    def transform(self, tensor):
        return tensor.reshape(1, -1)


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


def to_pruned(
    model: nn.Module, should_do: Callable = default_should_do, inplace=True, **kwargs
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
            PrunedLinear.from_linear(module, **kwargs)
            if isinstance(module, (nn.Linear, nn.LazyLinear))
            else PrunedConv2d.from_conv2d(module, **kwargs),
        )

    return model
