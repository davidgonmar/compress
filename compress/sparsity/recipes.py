from .policy import PruningPolicy, PolicyDict, Metric
from .groupers import (
    AbstractGrouper,
    UnstructuredGrouperConv2d,
    UnstructuredGrouperLinear,
    OutChannelGroupingGrouperConv2d,
    OutChannelGroupingGrouperLinear,
)
from compress.experiments.cifar_resnet import resnet20
import torchvision
import torch
import torch.nn as nn
import copy
import functools


def same_policy_for_all_layers(
    model: nn.Module, policy: PruningPolicy, keys: list
) -> PolicyDict:
    policies = {}
    for name, module in model.named_modules():
        if name in keys:
            assert isinstance(
                module, (nn.Conv2d, nn.Linear)
            ), f"Module {name} is not Conv2d or Linear"
            policies[name] = copy.deepcopy(policy)
    return policies


def _all_linears_and_convs(model: nn.Module) -> tuple[list, list]:
    conv_keys = []
    linear_keys = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d)):
            conv_keys.append(name)
        elif isinstance(module, (nn.Linear)):
            linear_keys.append(name)
    return conv_keys, linear_keys


def _consistent_every_layer_policy_dict(
    model: nn.Module,
    inter_metric: Metric | None,
    intra_metric: Metric | None,
    linear_grouper: AbstractGrouper,
    conv_grouper: AbstractGrouper,
) -> PolicyDict:
    policies = {}
    keys = _all_linears_and_convs(model)
    for name, module in model.named_modules():
        if name in keys:
            assert isinstance(
                module, (nn.Conv2d, nn.Linear)
            ), f"Module {name} is not Conv2d or Linear"
            if isinstance(module, nn.Conv2d):
                policies[name] = PruningPolicy(
                    grouper=conv_grouper,
                    inter_group_metric=inter_metric,
                    intra_group_metric=intra_metric,
                )
            elif isinstance(module, nn.Linear):
                policies[name] = PruningPolicy(
                    grouper=linear_grouper,
                    inter_group_metric=inter_metric,
                    intra_group_metric=intra_metric,
                )
    return policies


def _consistent_every_layer_policy_dict_cls(
    cls: type,
    inter_metric: Metric | None,
    intra_metric: Metric | None,
    linear_grouper: AbstractGrouper,
    conv_grouper: AbstractGrouper,
) -> PolicyDict:
    with torch.device("meta"):
        model = cls()
        return _consistent_every_layer_policy_dict(
            model,
            inter_metric,
            intra_metric,
            linear_grouper,
            conv_grouper,
        )


unstructured_resnet20_policy_dict = functools.partial(
    _consistent_every_layer_policy_dict_cls,
    cls=resnet20,
    linear_grouper=UnstructuredGrouperLinear(),
    conv_grouper=UnstructuredGrouperConv2d(),
)

per_output_channel_resnet20_policy_dict = functools.partial(
    _consistent_every_layer_policy_dict_cls,
    cls=resnet20,
    linear_grouper=OutChannelGroupingGrouperLinear(),
    conv_grouper=OutChannelGroupingGrouperConv2d(),
)

unstructured_resnet18_policies = functools.partial(
    _consistent_every_layer_policy_dict_cls,
    cls=torchvision.models.resnet18,
    linear_grouper=UnstructuredGrouperLinear(),
    conv_grouper=UnstructuredGrouperConv2d(),
)

per_output_channel_resnet18_policy_dict = functools.partial(
    _consistent_every_layer_policy_dict_cls,
    cls=torchvision.models.resnet18,
    linear_grouper=OutChannelGroupingGrouperLinear(),
    conv_grouper=OutChannelGroupingGrouperConv2d(),
)


def get_per_output_channel_policy_dict(model_or_name: str | nn.Module) -> PolicyDict:
    if isinstance(model_or_name, str):
        assert model_or_name in [
            "resnet18",
            "resnet20",
        ], f"Model {model_or_name} not supported"
        if model_or_name == "resnet18":
            return per_output_channel_resnet18_policy_dict()
        elif model_or_name == "resnet20":
            return per_output_channel_resnet20_policy_dict()
    else:
        assert isinstance(
            model_or_name, nn.Module
        ), f"Model {model_or_name} not supported"
        return _consistent_every_layer_policy_dict(
            model_or_name,
            inter_metric=None,
            intra_metric=None,
            linear_grouper=OutChannelGroupingGrouperLinear(),
            conv_grouper=OutChannelGroupingGrouperConv2d(),
        )
