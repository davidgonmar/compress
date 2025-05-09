import numpy as np
from functools import partial


def linear_keep_ratio(cur_iter, max_iter, target_sparsity):
    t = cur_iter / max_iter
    return 1 - (1 - target_sparsity) * t


def poly_keep_ratio(cur_iter, max_iter, target_sparsity, beta=0.5):
    t = cur_iter / max_iter
    return 1 - (1 - target_sparsity) * (t**beta)


def cos_keep_ratio(cur_iter, max_iter, target_sparsity):
    t = cur_iter / max_iter
    return target_sparsity + (1 - target_sparsity) * (1 + np.cos(np.pi * t)) / 2


def exp_keep_ratio(cur_iter, max_iter, target_sparsity, alpha=5.0):
    t = cur_iter / max_iter
    return 1 - (1 - target_sparsity) * (1 - np.exp(-alpha * t)) / (1 - np.exp(-alpha))


def get_scheduler(name: str, **kwargs):
    schedules = {
        "linear": linear_keep_ratio,
        "poly": poly_keep_ratio,
        "cos": cos_keep_ratio,
        "exp": exp_keep_ratio,
    }

    if name not in schedules:
        raise ValueError(
            f"Unknown schedule name: {name}. Available options are: {list(schedules.keys())}"
        )
    return partial(schedules[name], **kwargs)
