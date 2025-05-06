import torch
from compress.common.functional import (
    hoyer_sparsity,
    squared_hoyer_sparsity,
    scad,
)
from typing import Dict, Any
from compress.sparsity.pruning_strats import (
    OutChannelGroupingGrouperConv2d,
    OutChannelGroupingGrouperLinear,
)


# METRICS APPLIED TO A SINGLE GROUP
_params_metrics = {
    "hoyer_sparsity": lambda **kwargs: (
        lambda x, **kwargs: hoyer_sparsity(x, **kwargs),
        -1.0,
    ),
    "scad": lambda **kwargs: (lambda x, **kwargs: scad(x, **kwargs), -1.0),
    "squared_hoyer_sparsity": lambda **kwargs: (
        lambda x, **kwargs: squared_hoyer_sparsity(x, **kwargs),
        -1.0,
    ),
    "noop": lambda **kwargs: (lambda x, **kwargs: torch.tensor(0.0), 1.0),
}

_vmapped_hoyer_sparsity = torch.vmap(
    hoyer_sparsity,
    in_dims=0,  # batch over dimension 0 of the input
    out_dims=0,  # collect the scalar outputs into dim 0 of a vector
)

_vmapped_scad = torch.vmap(
    scad, in_dims=(0, None, None, None), out_dims=0
)  # (n_groups, m_elements_per_group) -> (n_groups,)


class L1L2IntraRatioRegularizer:
    def __call__(self, param, grouper):
        grouped = grouper.transform(param)  # (n_groups, m_elements_per_group)
        return -_vmapped_hoyer_sparsity(grouped)


class SCADIntraRegularizer:
    def __call__(self, param, grouper):
        grouped = grouper.transform(param)  # (n_groups, m_elements_per_group)
        return _vmapped_scad(
            grouped, 0.1, 3.7, reduction="none"
        )  # (n_groups, m_elements_per_group)


class L1L2InterRatioRegularizer:
    def __call__(self, param, grouper):
        grouped = grouper.transform(param)  # (n_groups, m_elements_per_group)
        summed = torch.norm(grouped, p=2, dim=1)  # (n_groups,)
        return -hoyer_sparsity(summed, True)  # (n_groups,)


def hoyer_sparsity_for_all_modules(param, grouper, **kwargs):
    grouped = grouper.transform(param)  # (n_groups, m_elements_per_group)
    return _vmapped_hoyer_sparsity(
        grouped, **kwargs
    )  # (n_groups, m_elements_per_group)


def get_regularizer_for_all_layers(model, regfn, conv_grouper, linear_grouper):
    res = {}

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            grouper = linear_grouper
        elif isinstance(module, torch.nn.Conv2d):
            grouper = conv_grouper
        else:
            continue

        res[name] = {
            "grouper": grouper,
            "regularizer": regfn,
            "weight": 1.0,
            "parameter": module.weight,
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
                parameter = cfg["parameter"]
            except KeyError as e:
                raise ValueError(f"Missing required spec key {e.args[0]} for '{name}'")
            if not isinstance(weight, (int, float)):
                raise TypeError(
                    f"Weight for '{name}' must be a float or int, got {type(weight)}"
                )
            if not isinstance(parameter, torch.nn.Parameter):
                raise TypeError(
                    f"Parameter for '{name}' must be a torch.nn.Parameter, got {type(parameter)}"
                )
            self.specs[name] = {
                "grouper": grouper,
                "regularizer": regularizer,
                "weight": float(weight),
                "parameter": parameter,
            }
        self.kwargs = kwargs

    def loss(self) -> torch.Tensor:
        total_loss = torch.tensor(
            0.0, device=next(iter(self.specs.values()))["parameter"].device
        )
        for name, spec in self.specs.items():
            grp = spec["grouper"]
            reg_fn = spec["regularizer"]
            w = spec["weight"]
            param = spec["parameter"]
            total_loss = total_loss + w * reg_fn(param, grp, **self.kwargs).mean()
        return total_loss


class L1L2ActivationInterRegularizer:
    def __call__(self, activation: torch.Tensor, grouper):
        assert isinstance(grouper, OutChannelGroupingGrouperConv2d) or isinstance(
            grouper, OutChannelGroupingGrouperLinear
        ), "grouper must be either OutChannelGroupingGrouperConv2d or OutChannelGroupingGrouperLinear"
        if isinstance(grouper, OutChannelGroupingGrouperConv2d):
            # For Conv2d, we need to reshape the activation to group by output channels
            activation = activation.view(
                activation.size(0), activation.size(1), -1
            )  # (B, O, H*W)
            norm = torch.norm(activation, p=2, dim=2)  # (B, O)
            batch_axes = 0
        elif isinstance(grouper, OutChannelGroupingGrouperLinear):
            # act shape = (..., O)
            norm = activation
            batch_axes = activation.shape[:-1]
            assert (
                len(batch_axes) == 1
            ), "For Linear layers, the activation should have only one batch axis"

        return -_vmapped_hoyer_sparsity(norm)  # (B, O) -> (B)


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
                parameter = cfg["parameter"]
            except KeyError as e:
                raise ValueError(f"Missing required spec key {e.args[0]} for '{name}'")
            if not isinstance(weight, (int, float)):
                raise TypeError(
                    f"Weight for '{name}' must be a float or int, got {type(weight)}"
                )
            if not isinstance(parameter, torch.nn.Parameter):
                raise TypeError(
                    f"Parameter for '{name}' must be a torch.nn.Parameter, got {type(parameter)}"
                )
            self.specs[name] = {
                "grouper": grouper,
                "regularizer": regularizer,
                "weight": float(weight),
                "parameter": parameter,
            }
        self.kwargs = kwargs

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
        total_loss = torch.tensor(
            0.0, device=next(iter(self.specs.values()))["parameter"].device
        )
        for name, spec in self.specs.items():
            grp = spec["grouper"]
            reg_fn = spec["regularizer"]
            w = spec["weight"]
            activation = self.activations[name]
            total_loss = total_loss + w * reg_fn(activation, grp, **self.kwargs).mean()
        for name in self.activations.keys():
            self.activations[name] = None
        return total_loss
