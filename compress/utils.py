from typing import List, Callable, Dict
import torch
from torch import nn
import functools as ft


def extract_weights(
    model,
    cls_list=None,
    additional_check=lambda module: True,
    keywords={"weight"},
    ret_module=False,
):
    if isinstance(keywords, str):
        keywords = {keywords}
    weights = []
    for name, module in model.named_modules():
        if cls_list is None or isinstance(module, tuple(cls_list)):
            if not additional_check(module):
                continue
            for keyword in keywords:
                if hasattr(module, keyword):
                    if not ret_module:
                        weights.append(getattr(module, keyword))
                    else:
                        # name should include weight attr
                        _name = name + "." + keyword
                        weights.append(((_name, module), getattr(module, keyword)))
    return weights


def dims_sub(dims1: list[int], dims2: list[int]):
    # dims in 1 but not in 2
    return [dim for dim in dims1 if dim not in dims2]


def unzip(lst: List[tuple]):
    return tuple(list(map(lambda x: x[i], lst)) for i in range(len(lst[0])))


def get_all_convs_and_linears(model):
    res = []
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            res.append(name)
    return res


def default_should_do(module: nn.Module, full_name: str):
    return True


def gather_submodules(model: nn.Module, should_do: Callable) -> list:
    return [
        (name, module)
        for name, module in model.named_modules()
        if should_do(module, name)
    ]


def keys_passlist_should_do(keys):
    return ft.partial(lambda keys, module, full_name: full_name in keys, keys)


def cls_passlist_should_do(cls_list):
    return ft.partial(
        lambda cls_list, module, full_name: isinstance(module, tuple(cls_list)),
        cls_list,
    )


def combine_should_do(should_do1: Callable, should_do2: Callable) -> Callable:
    return lambda module, full_name: should_do1(module, full_name) and should_do2(
        module, full_name
    )


def replace_with_factory(
    model: nn.Module, module_dict: Dict[str, nn.Module], factory_fn: Callable
):

    for name, module in module_dict.items():
        parent_module = model
        *parent_path, attr_name = name.split(".")
        for part in parent_path:
            parent_module = getattr(parent_module, part)
        setattr(parent_module, attr_name, factory_fn(name, module))
    return model


def is_conv2d(module: nn.Module) -> bool:
    return isinstance(module, (torch.nn.Conv2d, torch.nn.LazyConv2d))


def is_linear(module: nn.Module) -> bool:
    return isinstance(module, (torch.nn.Linear, torch.nn.LazyLinear))
