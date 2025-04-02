from typing import Callable
from torch import nn
import functools as ft


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
