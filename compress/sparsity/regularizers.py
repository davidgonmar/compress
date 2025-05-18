import torch
from compress.common.functional import (
    hoyer_sparsity,
    scad,
)
from typing import Dict, Any, Type, Callable
from compress.sparsity.groupers import (
    OutChannelGroupingGrouperConv2d,
    OutChannelGroupingGrouperLinear,
    AbstractGrouper,
)
import torch.nn as nn
from compress.sparsity.sparse_ops import (
    SparseConv2d,
    SparseLinear,
    SparseFusedConv2dBatchNorm2d,
)


_vmapped_hoyer_sparsity = torch.vmap(
    hoyer_sparsity,
    in_dims=0,  # batch over dimension 0 of the input
    out_dims=0,  # collect the scalar outputs into dim 0 of a vector
)  # (n_groups, m_elements_per_group) -> (n_groups,)

_vmapped_scad = torch.vmap(
    scad, in_dims=(0, None, None, None), out_dims=0
)  # (n_groups, m_elements_per_group) -> (n_groups,)


class L1L2IntraRatioRegularizer:
    def __call__(
        self, param: torch.Tensor, grouper: Type[OutChannelGroupingGrouperConv2d]
    ) -> torch.Tensor:
        grouped = grouper.transform(param)  # (n_groups, m_elements_per_group)
        return _vmapped_hoyer_sparsity(grouped, normalize=False)


class SCADIntraRegularizer:
    def __call__(
        self,
        param: torch.Tensor,
        grouper: Type[OutChannelGroupingGrouperConv2d],
        **kwargs,
    ) -> torch.Tensor:
        grouped = grouper.transform(param)  # (n_groups, m_elements_per_group)
        return _vmapped_scad(
            grouped, **kwargs, reduction="none"
        )  # (n_groups, m_elements_per_group)


class L1L2InterRatioRegularizer:
    def __call__(
        self, param: torch.Tensor, grouper: Type[OutChannelGroupingGrouperConv2d]
    ) -> torch.Tensor:
        grouped = grouper.transform(param)  # (n_groups, m_elements_per_group)
        l2 = (lambda x: torch.sqrt(torch.sum(x**2, dim=-1)))(grouped)
        return hoyer_sparsity(l2, False)  # (n_groups,)


def hoyer_sparsity_for_all_modules(
    param: torch.Tensor, grouper: Type[OutChannelGroupingGrouperConv2d], **kwargs
) -> torch.Tensor:
    grouped = grouper.transform(param)  # (n_groups, m_elements_per_group)
    return _vmapped_hoyer_sparsity(
        grouped, **kwargs
    )  # (n_groups, m_elements_per_group)


def get_regularizer_for_all_layers(
    model: nn.Module,
    regfn: Callable,
    conv_grouper: Type[AbstractGrouper],
    linear_grouper: Type[AbstractGrouper],
) -> Dict[str, Dict[str, Any]]:
    res = {}
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, SparseLinear)):
            grouper = linear_grouper
        elif isinstance(
            module, (nn.Conv2d, SparseConv2d, SparseFusedConv2dBatchNorm2d)
        ):
            grouper = conv_grouper
        else:
            continue

        res[name] = {
            "grouper": grouper,
            "regularizer": regfn,
            "weight": 1.0,
            "module": module,
        }
    return res


class SparsityParamRegularizer:
    def __init__(self, specs: Dict[str, Dict[str, Any]], **kwargs: Any):
        # Validate and store specs
        self.specs: Dict[str, Dict[str, Any]] = {}
        for name, cfg in specs.items():
            try:
                grouper = cfg["grouper"]
                regularizer = cfg["regularizer"]
                weight = cfg["weight"]
                module = cfg["module"]
            except KeyError as e:
                raise ValueError(f"Missing required spec key {e.args[0]} for '{name}'")
            if not isinstance(weight, (int, float)):
                raise TypeError(
                    f"Weight for '{name}' must be a float or int, got {type(weight)}"
                )
            if not isinstance(
                module,
                (
                    nn.Linear,
                    nn.Conv2d,
                    SparseLinear,
                    SparseConv2d,
                    SparseFusedConv2dBatchNorm2d,
                ),
            ):
                raise TypeError(
                    f"Module for '{name}' must be a torch.nn.Module, got {type(module)}"
                )
            self.specs[name] = {
                "grouper": grouper,
                "regularizer": regularizer,
                "weight": float(weight),
                "module": module,
            }
        self.kwargs = kwargs
        self.device = (
            next(iter(self.specs.values()))["module"].parameters().__next__().device
        )

    def loss(self) -> torch.Tensor:
        total_loss = torch.tensor(0.0, device=self.device)
        for name, spec in self.specs.items():
            grp = spec["grouper"]
            reg_fn = spec["regularizer"]
            w = spec["weight"]
            mod = spec["module"]
            param = mod.get_weight() if hasattr(mod, "get_weight") else mod.weight
            total_loss = total_loss + w * reg_fn(param, grp, **self.kwargs)
        return total_loss


class L1L2ActivationInterRegularizer:
    def __call__(
        self, activation: torch.Tensor, grouper: Type[AbstractGrouper]
    ) -> torch.Tensor:
        assert (
            grouper is OutChannelGroupingGrouperConv2d
            or grouper is OutChannelGroupingGrouperLinear
        ), "grouper must be either OutChannelGroupingGrouperConv2d or OutChannelGroupingGrouperLinear, got {}".format(
            grouper
        )
        if grouper is OutChannelGroupingGrouperConv2d:
            # For Conv2d, we need to reshape the activation to group by output channels
            activation = activation.view(
                activation.size(0), activation.size(1), -1
            )  # (B, O, H*W)
            norm = torch.norm(activation, p=2, dim=2)  # (B, O)
            batch_axes = 0
        elif grouper is OutChannelGroupingGrouperLinear:
            # act shape = (..., O)
            norm = activation
            batch_axes = activation.shape[:-1]
            assert (
                len(batch_axes) == 1
            ), "For Linear layers, the activation should have only one batch axis"

        return _vmapped_hoyer_sparsity(norm, normalize=False)  # (B, O) -> (B)


class SparsityActivationRegularizer:
    def __init__(
        self, model: torch.nn.Module, specs: Dict[str, Dict[str, Any]], **kwargs: Any
    ):
        # Validate and store specs
        self.specs: Dict[str, Dict[str, Any]] = {}
        for name, cfg in specs.items():
            try:
                grouper = cfg["grouper"]
                regularizer = cfg["regularizer"]
                weight = cfg["weight"]
                module = cfg["module"]
            except KeyError as e:
                raise ValueError(f"Missing required spec key {e.args[0]} for '{name}'")
            if not isinstance(weight, (int, float)):
                raise TypeError(
                    f"Weight for '{name}' must be a float or int, got {type(weight)}"
                )
            if not isinstance(
                module,
                (
                    nn.Linear,
                    nn.Conv2d,
                    SparseLinear,
                    SparseConv2d,
                    SparseFusedConv2dBatchNorm2d,
                ),
            ):
                raise TypeError(
                    f"Module for '{name}' must be a torch.nn.Module, got {type(module)}"
                )
            self.specs[name] = {
                "grouper": grouper,
                "regularizer": regularizer,
                "weight": float(weight),
                "module": module,
            }
        self.kwargs = kwargs
        self.device = (
            next(iter(self.specs.values()))["module"].parameters().__next__().device
        )

        # create hooks to store activations

        self.activations = {}
        self.hooks = {}
        self.model = model
        dic = dict(model.named_modules())

        for name, spec in self.specs.items():
            if name not in dic:
                raise ValueError(f"Module {name} not found in model")
            module = dic[name]
            if not hasattr(module, "register_forward_hook"):
                raise ValueError(f"Module {name} does not support forward hooks")
            hook = module.register_forward_hook(self._save_activation(name))
            self.hooks[name] = hook
        self.kwargs = kwargs

    def _save_activation(self, name):
        def hook(module, input, output):
            self.activations[name] = output

        return hook

    def loss(self) -> torch.Tensor:
        total_loss = torch.tensor(0.0, device=self.device)
        assert (
            next(iter(self.activations.values())) is not None
        ), "No activations found. Did you run the model?"

        for name, spec in self.specs.items():
            grp = spec["grouper"]
            reg_fn = spec["regularizer"]
            w = spec["weight"]
            activation = self.activations[name]
            total_loss = total_loss + w * reg_fn(activation, grp, **self.kwargs).mean()
        for name in self.activations.keys():
            self.activations[name] = None
        return total_loss
