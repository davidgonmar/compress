import torch.nn as nn
from typing import Callable
from tqdm import tqdm
from compress.low_rank_ops import LowRankLinear, LowRankConv2d
from compress.common import gather_submodules, default_should_do
from compress.utils import extract_weights
import torch


def default_tensor_to_matrix_reshape(tensor: torch.Tensor) -> torch.Tensor:
    assert tensor.dim() == 2, "Expected 2D tensor, got {}".format(tensor.shape)
    return tensor


def conv2d_to_matrix_reshape(tensor: torch.Tensor) -> torch.Tensor:
    assert tensor.dim() == 4
    o, i, h, w = tensor.shape
    return tensor.permute(1, 2, 3, 0).reshape(i * h * w, o)


_module_to_reshaper = {
    (nn.Linear, "weight"): default_tensor_to_matrix_reshape,
    (nn.LazyLinear, "weight"): default_tensor_to_matrix_reshape,
    (nn.Conv2d, "weight"): conv2d_to_matrix_reshape,
}


def extract_weights_and_reshapers(
    model, cls_list, additional_check=lambda *args: True, keywords="weight"
):
    params = extract_weights(
        model, cls_list, additional_check, keywords, ret_module=True
    )
    modules_and_names = [(name, module) for (name, module), param in params]
    reshapers_status = [
        (
            (module.__class__, name.split(".")[-1]),
            (module.__class__, name.split(".")[-1]) in _module_to_reshaper,
        )
        for name, module in modules_and_names
    ]

    if all(status for _, status in reshapers_status):
        print("Found reshapers for all modules.")
    else:
        found_reshapers = [
            module_info for module_info, status in reshapers_status if status
        ]
        not_found_reshapers = [
            module_info for module_info, status in reshapers_status if not status
        ]
        raise ValueError(
            "Cannot find reshaper for all modules. Found reshapers for: {}. Not found for: {}".format(
                found_reshapers, not_found_reshapers
            )
        )

    return [
        (param, _module_to_reshaper[(module.__class__, name.split(".")[-1])])
        for (name, module), param in params
    ]


def to_low_rank(
    model: nn.Module, should_do: Callable = default_should_do, inplace=True, **kwargs
):
    modules_to_replace = gather_submodules(model, should_do=should_do, prefix="")
    if not inplace:
        model_initializer = kwargs.pop("model_initializer", None)
        assert (
            model_initializer is not None
        ), "model_initializer must be provided if inplace=False"
        model_ = model_initializer()
        model_.load_state_dict(model.state_dict())
        model = model_

    for name, module in tqdm(modules_to_replace, desc="Replacing modules"):
        parent_module = model
        *parent_path, attr_name = name.split(".")
        for part in parent_path:
            parent_module = getattr(parent_module, part)

        setattr(
            parent_module,
            attr_name,
            LowRankLinear.from_linear(module, **kwargs)
            if isinstance(module, nn.Linear) or isinstance(module, nn.LazyLinear)
            else LowRankConv2d.from_conv2d(module, **kwargs),
        )

    return model


import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from typing import Callable


def plot_singular_values(model: nn.Module, should_do: Callable = default_should_do):
    mods_and_reshapers = extract_weights_and_reshapers(
        model, cls_list=[nn.Linear, nn.Conv2d]
    )
    num_plots = len(mods_and_reshapers)

    if num_plots == 0:
        print("No applicable layers found for singular value decomposition.")
        return

    cols = min(3, num_plots)  # Max 3 columns for readability
    rows = (num_plots + cols - 1) // cols  # Compute rows needed

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.flatten() if num_plots > 1 else [axes]

    for i, (param, reshaper) in enumerate(mods_and_reshapers):
        weight = param.data
        reshaped_weight = reshaper(weight)
        U, S, V = torch.svd(reshaped_weight)

        # nornmalize singular values
        S = S / torch.norm(S, p=2)

        axes[i].plot(S.cpu().numpy())
        axes[i].set_xlabel("Singular value index")
        axes[i].set_ylabel("Singular value")
        axes[i].set_title(f"Layer {i + 1}")

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle("Singular values of the model")
    plt.tight_layout()
    plt.show()
    print("Done plotting singular values")
