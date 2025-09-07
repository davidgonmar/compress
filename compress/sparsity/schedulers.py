import numpy as np
from functools import partial

# Schedulers for iterative sparsity


def linear_sparsity(cur_iter, max_iter, target_sparsity):
    t = cur_iter / max_iter
    return target_sparsity * t


def poly_sparsity(cur_iter, max_iter, target_sparsity, beta=0.5):
    t = cur_iter / max_iter
    return target_sparsity * (t**beta)


def cos_sparsity(cur_iter, max_iter, target_sparsity):
    t = cur_iter / max_iter
    return target_sparsity * (1 - np.cos(np.pi * t)) / 2


def exp_sparsity(cur_iter, max_iter, target_sparsity, alpha=5.0):
    t = cur_iter / max_iter
    return target_sparsity * (1 - np.exp(-alpha * t)) / (1 - np.exp(-alpha))


def get_scheduler(name: str, **kwargs):
    schedules = {
        "linear": linear_sparsity,
        "poly": poly_sparsity,
        "cos": cos_sparsity,
        "exp": exp_sparsity,
    }

    if name not in schedules:
        raise ValueError(
            f"Unknown schedule name: {name}. Available options are: {list(schedules.keys())}"
        )
    return partial(schedules[name], **kwargs)
