import torch.nn as nn
from tqdm import tqdm
from compress.factorization.low_rank_ops import LowRankLinear, LowRankConv2d
from compress.utils import (
    gather_submodules,
    keys_passlist_should_do,
    cls_passlist_should_do,
)
from compress.utils import extract_weights
import copy
import torch
from typing import Dict
import functools
from .lp_utils import maximize_energy
from compress.utils import replace_with_factory
from compress.utils import is_conv2d, is_linear
import math
from typing import Callable


def default_tensor_to_matrix_reshape(tensor: torch.Tensor) -> torch.Tensor:
    assert tensor.dim() == 2, "Expected 2D tensor, got {}".format(tensor.shape)
    return tensor


def conv2d_to_matrix_reshape(tensor: torch.Tensor) -> torch.Tensor:
    assert tensor.dim() == 4
    o, i, h, w = tensor.shape
    return tensor.permute(1, 2, 3, 0).reshape(i * h * w, o)


_module_to_reshaper = {
    (nn.Linear): default_tensor_to_matrix_reshape,
    (nn.LazyLinear): default_tensor_to_matrix_reshape,
    (nn.Conv2d): conv2d_to_matrix_reshape,
}


def extract_weights_and_reshapers(model, cls_list, additional_check=lambda *args: True):
    params = extract_weights(model, cls_list, additional_check, ret_module=True)
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
        (param, _module_to_reshaper[(module.__class__)])
        for (name, module), param in params
    ]


def all_same_rank_ratio(
    model,
    ratio,
    should_do=cls_passlist_should_do(
        (nn.Linear, nn.Conv2d, nn.LazyLinear, nn.LazyConv2d)
    ),
):
    modules_to_replace = gather_submodules(
        model,
        should_do=should_do,
    )
    di = {}
    for name, module in modules_to_replace:
        di[name] = {
            "name": "rank_ratio_to_keep",
            "value": ratio,
        }
    return di


def all_same_svals_energy_ratio(
    model,
    energy,
    should_do=cls_passlist_should_do(
        (nn.Linear, nn.Conv2d, nn.LazyLinear, nn.LazyConv2d)
    ),
):
    modules_to_replace = gather_submodules(
        model,
        should_do=should_do,
    )
    di = {}
    for name, module in modules_to_replace:
        di[name] = {
            "name": "svals_energy_ratio_to_keep",
            "value": energy,
        }
    return di


def all_same_params_ratio(
    model,
    ratio,
    should_do=cls_passlist_should_do(
        (nn.Linear, nn.Conv2d, nn.LazyLinear, nn.LazyConv2d)
    ),
):
    modules_to_replace = gather_submodules(
        model,
        should_do=should_do,
    )
    di = {}
    for name, module in modules_to_replace:
        di[name] = {
            "name": "params_ratio_to_keep",
            "value": ratio,
        }
    return di


# ==========================================================================================================
# These functions generate a vector with the costs of keeping a certain rank R approximation for each layer.
# ==========================================================================================================
def generate_cost_flops_linear(weight_shape: tuple, out_shape: tuple) -> torch.Tensor:
    # A decomposed linear layer has shapes W_0 in [O, R] and W_1 in [R, I], input in [B, I] and output in [B, O]
    # flops(R) = min(B * R * (I + O), B * I * O)
    R = torch.arange(1, min(weight_shape[0], weight_shape[1]) + 1, 1)
    O, I = weight_shape
    B = out_shape[0]
    return B * torch.minimum(R * (I + O), torch.tensor(I * O))


def generate_cost_flops_conv2d(filter_shape: tuple, out_shape: tuple):
    # A factorized convolution has shape
    # W_0 in [R, C_in, H_k, W_k] and W_1 in [C_out, R, 1, 1]
    # flops_1(R) = B * R * H_out * W_out * C_in * H_k * W_k + B * C_out * R * H_out * W_out = B * R * H_out * W_out * (C_in * H_k * W_k + C_out)
    # flops_2(R) = B * C_out * H_out * W_out * C_in * H_k * W_k
    # flops(R) = min(flops_1(R), flops_2(R))
    R = torch.arange(
        1,
        min(filter_shape[0], filter_shape[1] * filter_shape[2] * filter_shape[3]) + 1,
        1,
    )
    C_out, C_in, H_k, W_k = filter_shape
    B, H_out, W_out = out_shape[0], out_shape[2], out_shape[3]
    return B * torch.minimum(
        R * H_out * W_out * (C_in * H_k * W_k + C_out),
        torch.tensor(C_out * H_out * W_out * H_k * W_k * C_in),
    )


def generate_cost_params_linear(weight_shape: tuple) -> torch.Tensor:
    # A decomposed linear layer has shapes W_0 in [O, R] and W_1 in [R, I]
    # params(R) = min(R * (I + O), I * O)
    r_vec = torch.arange(1, min(weight_shape[0], weight_shape[1]) + 1, 1)
    O, I = weight_shape
    return torch.minimum(
        r_vec * (I + O),
        torch.tensor(I * O),
    )


def generate_cost_params_conv2d(filter_shape: tuple) -> torch.Tensor:
    # A decomposed convolution has shapes W_0 in [R, C_in, H_k, W_k] and W_1 in [C_out, R, 1, 1]
    # params_1(R) = R * (C_in * H_k * W_k + C_out)
    # params_2(R) = C_out * C_in * H_k * W_k
    R = torch.arange(
        1,
        min(filter_shape[0], filter_shape[1] * filter_shape[2] * filter_shape[3]) + 1,
        1,
    )

    C_out, C_in, H_k, W_k = filter_shape
    return torch.minimum(
        R * (C_in * H_k * W_k + C_out),
        torch.tensor(C_out * C_in * H_k * W_k),
    )


def get_rank_to_keep_from_rank_ratio(
    X: torch.tensor, S: torch.Tensor, rank_ratio: float
):
    # truncates towards 0
    assert 0.0 <= rank_ratio <= 1.0, "rank_ratio must be in [0, 1]"
    k = math.ceil(S.shape[0] * rank_ratio)
    return max(k, 1)


def get_rank_to_keep_from_energy_ratio(
    X: torch.Tensor, S: torch.Tensor, energy_ratio: float
) -> int:
    assert 0.0 <= energy_ratio <= 1.0
    sq = S.pow(2)
    cum_energy = sq.cumsum(dim=0)
    total_energy = cum_energy[-1]
    threshold = energy_ratio * total_energy
    idx = torch.searchsorted(cum_energy, threshold)
    return idx.item() + 1


def get_rank_to_keep_from_param_number_ratio(
    X: torch.Tensor,
    S: torch.Tensor,
    param_number_ratio: float,
):
    assert X.ndim == 2, "X must be 2-dimensional"
    assert S.ndim == 1, "Singular values must be 1-dimensional"
    m, n = X.shape
    # A in R^{m x r}
    # B in R^{r x n}
    # So keeping a rank involves a total of m + n parameters
    params_per_rank_kept = torch.arange(1, S.shape[0] + 1).float() * (m + n)
    rel_params_per_rank_kept = params_per_rank_kept / params_per_rank_kept[-1]
    rank_to_keep = torch.searchsorted(
        rel_params_per_rank_kept, param_number_ratio
    )  # rank_to_keep is the number of ranks to keep
    return rank_to_keep.item() + 1


rank_to_keep_name_to_fn = {
    "rank_ratio_to_keep": get_rank_to_keep_from_rank_ratio,
    "svals_energy_ratio_to_keep": get_rank_to_keep_from_energy_ratio,
    "params_ratio_to_keep": get_rank_to_keep_from_param_number_ratio,
}


def reshape_linear(w: torch.Tensor) -> torch.Tensor:
    assert w.dim() == 2, "Weight tensor must be 2D for linear layers"
    return w.T


def reshape_conv2d(w: torch.Tensor) -> torch.Tensor:
    assert w.dim() == 4, "Weight tensor must be 4D for convolutional layers"
    C_o, C_i, H_k, W_k = w.shape
    return w.reshape(C_o, C_i * H_k * W_k).T  # reshape to [C_o, C_i * H_k * W_k]


def get_reshape(module: nn.Module) -> callable:
    """
    Returns a function to reshape the weights of the module.
    """
    if is_linear(module):
        return reshape_linear
    elif is_conv2d(module):
        return reshape_conv2d
    else:
        raise ValueError("Module should be either Linear or Conv2d")


def decompose_params(w: torch.Tensor):
    U, S, V_T = torch.linalg.svd(w, full_matrices=True)  # complete SVD
    return U, S, V_T


def crop_svd(U, S, V_T, rank):
    return U[:, :rank], S[:rank], V_T[:rank, :]


def get_factors(U, S, V_T):
    W0 = U @ torch.diag(S)
    W1 = V_T
    return W0, W1


def should_do_low_rank(W, rank):
    # it can be proved that rank is memory efficient <=> rank is compute efficient
    m, n = W.shape
    cost_base = m * n
    cost_low_rank = (m + n) * rank
    return cost_low_rank < cost_base


def factorize_linear(module, get_rank: Callable, factors=None):
    W = module.weight.T  # shape (in, out)
    if factors is None:
        U, S, V_T = decompose_params(W)
    else:
        U, S, V_T = factors
    rank = get_rank(W, U, S, V_T)
    if not should_do_low_rank(W, rank):
        return module
    U, S, V_T = crop_svd(U, S, V_T, rank)
    W0, W1 = get_factors(U, S, V_T)  # shape (in, rank), (out, rank)
    low_rank_linear = LowRankLinear(
        module.in_features,
        module.out_features,
        rank,
        bias=module.bias is not None,
    ).to(module.weight.device)
    low_rank_linear.w0.data.copy_(W0)
    low_rank_linear.w1.data.copy_(W1)
    if module.bias is not None:
        low_rank_linear.bias.data.copy_(module.bias)
    return low_rank_linear


def factorize_conv2d(module, get_rank: Callable, factors=None):
    W = module.weight
    C_o, C_i, H_k, W_k = W.shape
    reshaped = W.reshape(C_o, C_i * H_k * W_k).T
    if factors is None:
        U, S, V_T = decompose_params(reshaped)
    else:
        U, S, V_T = factors
    rank = get_rank(W, U, S, V_T)
    if not should_do_low_rank(reshaped, rank):
        return module
    U, S, V_T = crop_svd(
        U, S, V_T, rank
    )  # [C_i * H_k * W_k, rank], [rank], [rank, C_o]
    W0, W1 = get_factors(U, S, V_T)  # shape (C_i * H_k * W_k, rank), (rank, C_o)
    W1 = W1.T.reshape(C_o, rank, 1, 1)
    W0 = W0.T.reshape(rank, C_i, H_k, W_k)
    low_rank_conv2d = LowRankConv2d(
        module.in_channels,
        module.out_channels,
        (H_k, W_k),
        rank,
        stride=module.stride,
        padding=module.padding,
        dilation=module.dilation,
        groups=module.groups,
        bias=module.bias is not None,
    ).to(module.weight.device)
    low_rank_conv2d.w0.data.copy_(W0)
    low_rank_conv2d.w1.data.copy_(W1)
    if module.bias is not None:
        low_rank_conv2d.bias.data.copy_(module.bias)
    return low_rank_conv2d


def to_low_rank_manual(
    model: nn.Module,
    inplace=True,
    cfg_dict: Dict[str, Dict[str, float]] = {},
):
    """
    Converts a model to low-rank by replacing its linear and convolutional layers with low-rank approximations.

    Args:
        model (nn.Module): The model to convert.
        inplace (bool): If True, modifies the model in place. If False, returns a copy of the model with low-rank layers.
        cfg_dict (Dict[str, Dict[str, float]]): A dictionary where keys are module names and values are dictionaries with configuration parameters for low-rank approximation.
    Returns:
        nn.Module: The modified model with low-rank layers.
    """
    if not inplace:
        model = copy.deepcopy(model)

    modules_to_replace = gather_submodules(
        model,
        should_do=keys_passlist_should_do(cfg_dict.keys()),
    )

    def factory_fn(name, module):
        if is_linear(module):
            return factorize_linear(
                module,
                lambda W, U, S, V_T: rank_to_keep_name_to_fn[cfg_dict[name]["name"]](
                    W, S, cfg_dict[name]["value"]
                ),
            )
        elif is_conv2d(module):
            return factorize_conv2d(
                module,
                lambda W, U, S, V_T: rank_to_keep_name_to_fn[cfg_dict[name]["name"]](
                    W, S, cfg_dict[name]["value"]
                ),
            )
        else:
            return module

    modules_to_replace = {name: module for name, module in modules_to_replace}
    replace_with_factory(model, modules_to_replace, factory_fn)
    return model


def to_low_rank_global(
    model: nn.Module,
    metric: str,
    ratio_to_keep,
    keys,
    sample_input,
    bn_keys=None,
    inplace=True,
):
    """
    Converts a model to low-rank by replacing its linear and convolutional layers with low-rank approximations.
    Args:
        model (nn.Module): The model to convert.
        metric (str): The metric to use for low-rank approximation. Can be "rank", "flops", or "params".
        ratio_to_keep (float): The ratio of the metric to keep (e.g., 0.5 means keep 50% of the metric).
        keys (list): A list of keys to identify the modules to replace.
        sample_input (torch.Tensor): A sample input tensor to the model, used to determine output sizes of layers.
        bn_keys (list, optional): A list of tuples containing (conv_name, bn_name) pairs for batch normalization layers. Defaults to None.
        inplace (bool): If True, modifies the model in place. If False, returns a copy of the model with low-rank layers.
    Returns:
        nn.Module: The modified model with low-rank layers.
    """

    if not inplace:
        model = copy.deepcopy(model)

    modules_to_replace = gather_submodules(
        model,
        should_do=keys_passlist_should_do(keys),
    )

    rand_inp = sample_input.to(next(model.parameters()).device)

    # get output sizes of every layer
    hooks = []
    sizes = []

    def hook_fn(module, input, output):
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.LazyConv2d):
            sizes.append(output.shape)
        elif isinstance(module, nn.Linear) or isinstance(module, nn.LazyLinear):
            sizes.append(output.shape)
        else:
            raise ValueError("Module should be either Conv2d or Linear")

    for name, module in modules_to_replace:
        hook = module.register_forward_hook(hook_fn)
        hooks.append(hook)

    prev_state = model.training
    model.eval()
    with torch.no_grad():
        model(rand_inp)

    for hook in hooks:
        hook.remove()

    model.train(prev_state)

    assert len(sizes) == len(
        modules_to_replace
    ), "Sizes and modules to replace do not match"

    reshaped_params = {
        name: get_reshape(module)(module.weight.detach())
        for name, module in modules_to_replace
    }
    factors = {
        name: decompose_params(reshaped) for name, reshaped in reshaped_params.items()
    }

    cum_energies = {
        name: (cs := torch.cumsum(factor[1] ** 2, dim=0)) / cs[-1]
        for name, factor in factors.items()
    }

    cum_energies = [cum_energies[name] for name, _ in modules_to_replace]
    if bn_keys:
        bn_stats = []
        # bn_keys is a collection of (conv_name, bn_name) pairs
        bn_dict = {conv_name: bn_name for conv_name, bn_name in bn_keys}

        named_mods = model.named_modules()
        keytomods = {k: v for k, v in named_mods}

        for name, module in modules_to_replace:
            if name in bn_dict:
                bn_mod = keytomods[bn_dict[name]]
                rm = bn_mod.running_mean.detach()  # shape [C]
                rv = bn_mod.running_var.detach()  # shape [C]

                # global variance = E[X^2] - E[X]^2
                # where E[X^2] = mean(rv + rm^2), and E[X] = mean(rm)
                global_var = (rv + rm.pow(2)).mean() - rm.mean().pow(2)

                bn_stats.append(global_var)
            else:
                bn_stats.append(torch.tensor(1.0, device=module.weight.device))

        # compute importance of each layer with avg std of BN
        cum_energies = [
            energy * torch.sqrt(stat) for energy, stat in zip(cum_energies, bn_stats)
        ]

    ws = [mod.weight.detach() for _, mod in modules_to_replace]
    mods = [mod for _, mod in modules_to_replace]
    # costs
    if metric == "rank":
        costs = [torch.arange(1, len(energy) + 1, 1) for energy in cum_energies]
        costs = [torch.cumsum(cost, 0) for cost in costs]
    elif metric == "flops":
        costs = [
            (
                generate_cost_flops_linear(w.shape, out_size)
                if isinstance(mod, nn.Linear)
                else generate_cost_flops_conv2d(w.shape, out_size)
            )
            for w, out_size, mod in zip(ws, sizes, mods)
        ]
    elif metric == "params":
        costs = [
            (
                generate_cost_params_linear(w.shape)
                if isinstance(mod, nn.Linear)
                else generate_cost_params_conv2d(w.shape)
            )
            for w, out_size, mod in zip(ws, sizes, mods)
        ]
    if metric == "rank":
        n_to_keep = sum(len(energy) for energy in cum_energies) * ratio_to_keep
    elif metric == "flops":
        n_to_keep = sum(cost[-1].item() for cost in costs) * ratio_to_keep
    elif metric == "params":
        n_to_keep = sum(cost[-1].item() for cost in costs) * ratio_to_keep

    selected_indices = maximize_energy(cum_energies, costs, n_to_keep)

    selected_indices_per_module = {
        name: s for (name, _), s in zip(modules_to_replace, selected_indices)
    }

    def factory_fn(name, module):
        parent_module = model
        *parent_path, attr_name = name.split(".")
        for part in parent_path:
            parent_module = getattr(parent_module, part)

        if is_linear(module):
            return factorize_linear(
                module,
                lambda W, U, S, V_T: selected_indices_per_module[name],
                factors=factors[name],
            )
        elif is_conv2d(module):
            return factorize_conv2d(
                module,
                lambda W, U, S, V_T: selected_indices_per_module[name],
                factors=factors[name],
            )
        else:
            return module

    modules_to_replace = {name: module for name, module in modules_to_replace}
    replace_with_factory(model, modules_to_replace, factory_fn)
    return model


def obtain_whitening_matrix_cholesky(
    acts: torch.Tensor,
    module: nn.Module,
):
    """
    Computes the whitening matrix for the given activations and module.
    The whitening matrix is computed as the Cholesky decomposition of the covariance matrix of the activations.
    """
    if isinstance(module, nn.Conv2d):
        # Input should be of shape (B, Cin, H, W)
        assert acts.dim() == 4
        im2coled = nn.functional.unfold(
            acts,
            kernel_size=module.kernel_size,
            padding=module.padding,
            stride=module.stride,
        )  # shape (B, Cin * H_k * W_k, H_out * W_out)
        im2coled = im2coled.permute(0, 2, 1).reshape(
            im2coled.shape[0] * im2coled.shape[2], -1
        )
    elif isinstance(module, nn.Linear):
        # Input should be of shape (B, Cin)
        assert acts.dim() == 2
        im2coled = acts
    else:
        raise ValueError("Module should be either Conv2d or Linear")

    m = im2coled.T @ im2coled
    m = m.double()
    try:
        chol = torch.linalg.cholesky(m)
    except RuntimeError:
        eigenvalues = torch.linalg.eigvalsh(m)
        m = (-eigenvalues[0] + 1e-6) * torch.eye(m.shape[0]).to(m.device) + m
        chol = torch.linalg.cholesky(m)
    inv = torch.inverse(chol)
    return inv.float(), chol.float()


obtain_whitening_matrix = obtain_whitening_matrix_cholesky


def obtain_whitening_matrix_svd(
    acts: torch.Tensor,
    module: nn.Module,
):
    if isinstance(module, nn.Conv2d):
        assert acts.dim() == 4
        im2coled = nn.functional.unfold(
            acts,
            kernel_size=module.kernel_size,
            padding=module.padding,
            stride=module.stride,
        )
        im2coled = im2coled.permute(0, 2, 1).reshape(
            im2coled.shape[0] * im2coled.shape[2], -1
        )
    elif isinstance(module, nn.Linear):
        assert acts.dim() == 2
        im2coled = acts
    else:
        raise ValueError("Module should be either Conv2d or Linear")

    U, S, Vh = torch.linalg.svd(im2coled, full_matrices=False)
    keep = S > 1e-6
    if not torch.any(keep):
        raise RuntimeError("All singular values ≈ 0; cannot whiten.")

    S_nz = S[keep]
    V_nz = Vh[keep, :]

    return V_nz @ torch.diag(1 / S_nz), torch.diag(torch.sqrt(S_nz)) @ V_nz.T


def factorize_linear_whitened(
    module,
    get_rank: Callable,
    data_whitening_matrix,
    data_whitening_matrix_inverse,
    factors=None,
):
    W = module.weight.T
    if factors is None:
        U, S, V_T = decompose_params(data_whitening_matrix_inverse @ W)
    else:
        U, S, V_T = factors
    rank = get_rank(W, U, S, V_T)
    if not should_do_low_rank(W, rank):
        return module
    U, S, V_T = crop_svd(U, S, V_T, rank)
    W0, W1 = get_factors(U, S, V_T)  # shape (in, rank), (out, rank)
    W0 = data_whitening_matrix @ W0

    low_rank_linear = LowRankLinear(
        module.in_features,
        module.out_features,
        rank,
        bias=module.bias is not None,
    ).to(module.weight.device)
    low_rank_linear.w0.data.copy_(W0)
    low_rank_linear.w1.data.copy_(W1)
    if module.bias is not None:
        low_rank_linear.bias.data.copy_(module.bias)
    return low_rank_linear


def factorize_conv2d_whitened(
    module,
    get_rank: Callable,
    data_whitening_matrix,
    data_whitening_matrix_inverse,
    factors=None,
):
    W = module.weight
    C_o, C_i, H_k, W_k = W.shape
    reshaped = W.reshape(C_o, C_i * H_k * W_k).T
    # print(data_whitening_matrix_inverse @ data_whitening_matrix)
    if factors is None:
        U, S, V_T = decompose_params(data_whitening_matrix_inverse @ reshaped)
    else:
        U, S, V_T = factors
    rank = get_rank(W, U, S, V_T)
    if not should_do_low_rank(reshaped, rank):
        return module
    U, S, V_T = crop_svd(
        U, S, V_T, rank
    )  # [C_i * H_k * W_k, rank], [rank], [rank, C_o]
    W0, W1 = get_factors(U, S, V_T)  # [C_i * H_k * W_k, rank], [rank, C_o]
    W0 = data_whitening_matrix @ W0
    W1 = W1.T.reshape(C_o, rank, 1, 1)
    W0 = W0.T.reshape(rank, C_i, H_k, W_k)
    low_rank_conv2d = LowRankConv2d(
        module.in_channels,
        module.out_channels,
        (H_k, W_k),
        rank,
        stride=module.stride,
        padding=module.padding,
        dilation=module.dilation,
        groups=module.groups,
        bias=module.bias is not None,
    ).to(module.weight.device)
    low_rank_conv2d.w0.data.copy_(W0)
    low_rank_conv2d.w1.data.copy_(W1)
    if module.bias is not None:
        low_rank_conv2d.bias.data.copy_(module.bias)
    return low_rank_conv2d


def to_low_rank_manual_activation_aware(
    model: nn.Module, dataloader, inplace=True, cfg_dict={}
):

    if not inplace:
        model = copy.deepcopy(model)

    acts = {}
    hooks = []

    modules_to_replace = gather_submodules(
        model,
        should_do=keys_passlist_should_do(cfg_dict.keys()),
    )

    def hook_fn(module, input, output):
        input = input[0] if isinstance(input, tuple) else input
        if acts.get(module) is None:
            acts[module] = input
        else:
            acts[module] = torch.cat((acts[module], input), dim=0)

    for name, module in modules_to_replace:
        hook = module.register_forward_hook(hook_fn)
        hooks.append(hook)

    prev_state = model.training
    model.eval()
    with torch.no_grad():

        for batch in tqdm(dataloader, desc="Getting activations"):
            if isinstance(batch, dict):
                inputs = {
                    key: value.to(next(model.parameters()).device)
                    for key, value in batch.items()
                }
                model(**inputs)
            else:
                inputs, targets = batch
                inputs, targets = inputs.to(
                    next(model.parameters()).device
                ), targets.to(next(model.parameters()).device)
                model(inputs)

    for hook in hooks:
        hook.remove()

    model.train(prev_state)

    # get the cholesky decomposition of the covariance matrix of each activation im2col'ed in case of conv2d
    whit = {
        name: obtain_whitening_matrix(acts[module], module)
        for name, module in modules_to_replace
    }

    def factory_fn(name, module):
        parent_module = model
        *parent_path, attr_name = name.split(".")
        for part in parent_path:
            parent_module = getattr(parent_module, part)

        if is_linear(module):
            return factorize_linear_whitened(
                module,
                lambda W, U, S, V_T: rank_to_keep_name_to_fn[cfg_dict[name]["name"]](
                    W, S, cfg_dict[name]["value"]
                ),
                whit[name][0],
                whit[name][1],
            )
        elif is_conv2d(module):
            return factorize_conv2d_whitened(
                module,
                lambda W, U, S, V_T: rank_to_keep_name_to_fn[cfg_dict[name]["name"]](
                    W, S, cfg_dict[name]["value"]
                ),
                whit[name][0],
                whit[name][1],
            )
        else:
            return module

    modules_to_replace = {name: module for name, module in modules_to_replace}
    replace_with_factory(model, modules_to_replace, factory_fn)
    return model


# global activation aware svd
def to_low_rank_activation_aware_global(
    model: nn.Module,
    dataloader,
    keys,
    ratio_to_keep,
    bn_keys=None,
    metric="flops",
    inplace=True,
):
    if not inplace:
        model = copy.deepcopy(model)

    acts = {}
    outs = {}
    hooks = []

    modules_to_replace = gather_submodules(
        model,
        should_do=keys_passlist_should_do(keys),
    )

    def hook_fn(name, module, input, output):
        input = input[0] if isinstance(input, tuple) else input
        if acts.get(name) is None:
            acts[name] = input

        else:
            acts[name] = torch.cat((acts[name], input), dim=0)

        if outs.get(name) is None:
            outs[name] = output

        else:
            outs[name] = torch.cat((outs[name], output), dim=0)

    for name, module in modules_to_replace:
        hook = module.register_forward_hook(functools.partial(hook_fn, name))
        hooks.append(hook)

    prev_state = model.training

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Getting activations"):
            if isinstance(batch, dict):
                inputs = {
                    key: value.to(next(model.parameters()).device)
                    for key, value in batch.items()
                }
                model(**inputs)
            else:
                inputs, targets = batch
                inputs, targets = inputs.to(
                    next(model.parameters()).device
                ), targets.to(next(model.parameters()).device)
                model(inputs)

    model.train(prev_state)

    for hook in hooks:
        hook.remove()
    whit = {
        name: obtain_whitening_matrix(acts[name], module)
        for name, module in modules_to_replace
    }

    # energies
    cum_energies = []
    for name, module in modules_to_replace:
        reshaped = get_reshape(module)(module.weight.detach())
        aa = whit[name][1] @ reshaped

        # shape (Cout, Cin * H_k * W_k) @ (Cin * H_k * W_k, Cin * H_k * W_k) = (Cout, Cin * H_k * W_k) if conv
        # or (Cout, Cin) if linear
        svals = torch.linalg.svdvals(aa)
        cum_energy = torch.cumsum(svals**2, 0) / torch.sum(svals**2)
        cum_energies.append(cum_energy)

    if bn_keys:
        bn_stats = []
        # bn_keys is a collection of (conv_name, bn_name) pairs
        bn_dict = {conv_name: bn_name for conv_name, bn_name in bn_keys}

        named_mods = model.named_modules()
        keytomods = {k: v for k, v in named_mods}

        for name, module in modules_to_replace:
            if name in bn_dict:
                bn_mod = keytomods[bn_dict[name]]
                rm = bn_mod.running_mean.detach()  # shape [C]
                rv = bn_mod.running_var.detach()  # shape [C]

                # global variance = E[X^2] - E[X]^2
                # where E[X^2] = mean(rv + rm^2), and E[X] = mean(rm)
                global_var = (rv + rm.pow(2)).mean() - rm.mean().pow(2)

                bn_stats.append(global_var)
            else:
                bn_stats.append(torch.tensor(1.0, device=module.weight.device))

        # compute importance of each layer with avg std of BN
        cum_energies = [
            energy * torch.sqrt(stat) for energy, stat in zip(cum_energies, bn_stats)
        ]
    # costs
    ws = [mod.weight.detach() for _, mod in modules_to_replace]

    if metric == "rank":
        costs = [torch.arange(1, len(energy) + 1, 1) for energy in cum_energies]
        costs = [torch.cumsum(cost, 0) for cost in costs]

    elif metric == "flops":
        costs = [
            (
                generate_cost_flops_linear(w.shape, out_size)
                if len(out_size) == 2
                else generate_cost_flops_conv2d(w.shape, out_size)
            )
            for w, out_size in zip(
                ws, [outs[name].shape for name, _ in modules_to_replace]
            )
        ]

    elif metric == "params":
        costs = [
            (
                generate_cost_params_linear(w.shape)
                if len(out_size) == 2
                else generate_cost_params_conv2d(w.shape)
            )
            for w, out_size in zip(
                ws, [outs[name].shape for name, _ in modules_to_replace]
            )
        ]

    if metric == "rank":
        n_rank_to_keep = sum(len(energy) for energy in cum_energies) * ratio_to_keep

    elif metric == "flops":
        n_rank_to_keep = sum(cost[-1].item() for cost in costs) * ratio_to_keep

    elif metric == "params":
        n_rank_to_keep = sum(cost[-1].item() for cost in costs) * ratio_to_keep

    selected_indices = maximize_energy(cum_energies, costs, n_rank_to_keep)
    selected_indices_per_module = {}
    for (name, _), s in zip(modules_to_replace, selected_indices):
        selected_indices_per_module[name] = s

    def factory_fn(name, module):
        parent_module = model
        *parent_path, attr_name = name.split(".")
        for part in parent_path:
            parent_module = getattr(parent_module, part)

        if is_linear(module):
            return factorize_linear_whitened(
                module,
                lambda W, U, S, V_T: selected_indices_per_module[name],
                whit[name][0],
                whit[name][1],
            )
        elif is_conv2d(module):
            return factorize_conv2d_whitened(
                module,
                lambda W, U, S, V_T: selected_indices_per_module[name],
                whit[name][0],
                whit[name][1],
            )
        else:
            return module

    modules_to_replace = {name: module for name, module in modules_to_replace}
    replace_with_factory(model, modules_to_replace, factory_fn)
    return model
