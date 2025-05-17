from typing import List
import torch


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
