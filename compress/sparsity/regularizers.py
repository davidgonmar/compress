import torch
from typing import List
from compress.utils import extract_weights
from compress.common.functional import (
    hoyer_sparsity,
    squared_hoyer_sparsity,
    scad,
)
from compress.sparsity.pruning_strats import PruningGranularity

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


def extract_weights_and_pruning_granularities(
    model, cls_list, cfg, additional_check=lambda *args: True, keywords="weight"
):
    params = extract_weights(
        model, cls_list, additional_check, keywords, ret_module=True
    )
    modules_and_names = [(name, module) for (name, module), param in params]
    granul_status = [
        (
            (module.__class__, name.split(".")[-1]),
            (module.__class__, name.split(".")[-1]) in cfg,
        )
        for name, module in modules_and_names
    ]

    if all(status for _, status in granul_status):
        print("Found pruning granularities for all modules.")
    else:
        found = [module_info for module_info, status in granul_status if status]
        not_found = [module_info for module_info, status in granul_status if not status]
        raise ValueError(
            "Cannot find pruning granularities for all modules. Found pruning granularities for: {}. Not found for: {}".format(
                found, not_found
            )
        )

    return [
        (param, cfg[(module.__class__, name.split(".")[-1])]())
        for (name, module), param in params
    ]


class SparsityRegularizer:
    def __init__(
        self,
        metric: str,
        params_and_pruning_granularities: List[tuple[torch.Tensor, PruningGranularity]],
        weights: float | List[float] = 1.0,
        mode: str = "inside_group",
        **kwargs
    ):
        self.params_and_pruning_granularities = params_and_pruning_granularities
        self.weights = (
            [weights] * len(params_and_pruning_granularities)
            if isinstance(weights, float)
            else weights
        )
        assert len(self.params_and_pruning_granularities) == len(
            self.weights
        ), "Number of params and weights should match, got {} and {}".format(
            len(self.params), len(self.weights)
        )
        self.metric = metric
        self.kwargs = kwargs

        self.fn, self.sgn = _params_metrics[metric](**kwargs)
        self.mode = mode
        # granul.transform maps the weights to a tensor of shape (n_groups, m_elements_per_group), so
        # the metric can be obtained by doing mean over dim=1.

    # tries to prune entire groups
    def call_per_group(self) -> torch.Tensor:
        return self.sgn * sum(
            weight
            * self.fn(granul.transform(param).sum(1, keepdim=False), **self.kwargs)
            for (param, granul), weight in zip(
                self.params_and_pruning_granularities, self.weights
            )
        )

    # inside each group, applies the sparsity metric locally
    def call_inside_group(self) -> torch.Tensor:
        return self.sgn * sum(
            weight
            * torch.vmap(lambda x: self.fn(x, **self.kwargs))(
                granul.transform(param)
            ).mean()
            for (param, granul), weight in zip(
                self.params_and_pruning_granularities, self.weights
            )
        )

    def __call__(self) -> torch.Tensor:
        if self.mode == "inside_group":
            return self.call_inside_group()
        elif self.mode == "per_group":
            return self.call_per_group()
