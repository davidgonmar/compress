import numpy as np
from functools import partial


def linear_keep_ratio(cur_iter: int, max_iter: int, target_density: float):
    t = cur_iter / max_iter
    return 1 - (1 - target_density) * t


def poly_keep_ratio(cur_iter: int, max_iter: int, target_density: float, beta=0.5):
    t = cur_iter / max_iter
    return 1 - (1 - target_density) * (t**beta)


def cos_keep_ratio(cur_iter: int, max_iter: int, target_density: float):
    t = cur_iter / max_iter
    return target_density + (1 - target_density) * (1 + np.cos(np.pi * t)) / 2


def exp_keep_ratio(cur_iter: int, max_iter: int, target_density: float, alpha=5.0):
    t = cur_iter / max_iter
    return 1 - (1 - target_density) * (1 - np.exp(-alpha * t)) / (1 - np.exp(-alpha))


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
