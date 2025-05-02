from typing import Callable
from torch import nn
from compress.sparsity.pruned_ops import PrunedLinear, PrunedConv2d
from tqdm import tqdm
from compress.sparsity.pruning_strats import (
    UnstructuredGranularityLinear,
    UnstructuredGranularityConv2d,
)
import torch
from compress.sparsity.pruned_ops import _get_mask_from_already_regrouped
from compress.common import default_should_do, gather_submodules


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
        if (
            isinstance(module, (nn.Linear, nn.LazyLinear))
            or module == nn.Linear
            or module == nn.LazyLinear
        ):
            return cfg.get(nn.Linear, UnstructuredGranularityLinear)
        if isinstance(module, nn.Conv2d) or module == nn.Conv2d:
            return cfg.get(nn.Conv2d, UnstructuredGranularityConv2d)


def get_global_mask(
    model: torch.nn.Module,
    ratio_to_keep: float = 1.0,
    ratio_to_keep_in_group: float = 1.0,
    granularity_cls_map: dict = {
        nn.Linear: UnstructuredGranularityLinear,
        nn.Conv2d: UnstructuredGranularityConv2d,
    },
):
    assert (
        ratio_to_keep_in_group == 1.0
    ), "ratio_to_keep_in_group must be 1.0 for global pruning"
    regrouped_weights = []
    names = []
    regroupers = []
    for name, module in model.named_modules():
        if type(module) in granularity_cls_map:
            regrouper = granularity_cls_map[type(module)]()
            regrouped_weights.append(regrouper.transform(module.weight))
            assert regrouped_weights[-1].ndim == 2
            names.append(name)
            regroupers.append(regrouper)

    rglst = regrouped_weights
    regrouped_weights = [
        x.mean(dim=1, keepdim=True) for x in regrouped_weights
    ]  # take the mean of each group
    # concatenate all the regrouped weights
    regrouped_weights = torch.cat(regrouped_weights, dim=0)

    # get the mask
    mask = _get_mask_from_already_regrouped(
        regrouped_weights, ratio_to_keep, ratio_to_keep_in_group
    )

    # split the mask back into the original shapes
    masks = []
    idx = 0
    idxx = 0
    for (name, module), regrouped in zip(model.named_modules(), rglst):
        n_groups = regrouped.shape[0]
        masks.append(
            regroupers[idxx].untransform(
                mask[idx : idx + n_groups].broadcast_to(regrouped.shape)
            )
        )
        idx += n_groups
        idxx += 1

    return {name: mask for name, mask in zip(names, masks)}


def to_pruned(
    model: nn.Module,
    policy: PruningPolicy,
    should_do: Callable = default_should_do,
    inplace=True,
    to_sparse_semistructured=False,
    global_prune=False,
    **kwargs,
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
    if global_prune:
        global_mask = get_global_mask(
            model,
            kwargs.get("ratio_to_keep", 1.0),
            kwargs.get("ratio_to_keep_in_group", 1.0),
            {
                nn.Linear: policy.get_granularity(nn.Linear),
                nn.LazyLinear: policy.get_granularity(nn.LazyLinear),
                nn.Conv2d: policy.get_granularity(nn.Conv2d),
            },
        )
    for name, module in tqdm(modules_to_replace, desc="Replacing modules"):
        parent_module = model
        *parent_path, attr_name = name.split(".")
        for part in parent_path:
            parent_module = getattr(parent_module, part)

        if not global_prune:
            setattr(
                parent_module,
                attr_name,
                _module_to_pruned[type(module)](
                    module, granularity_cls=policy.get_granularity(module), **kwargs
                ),
            )
        elif name in global_mask:
            setattr(
                parent_module,
                attr_name,
                _module_to_pruned[type(module)](
                    module,
                    granularity_cls=policy.get_granularity(module),
                    mask=global_mask[name],
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
