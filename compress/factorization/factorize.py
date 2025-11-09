import torch.nn as nn
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


# ==========================================================================================================
# Some general utilities
# ==========================================================================================================


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
# If the factorization is not worth it in terms of cost, we simply not factorize it, so the cost is capped to the original cost.
# =========================================================================================================


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


# ==========================================================================================================
# These functions compute the rank to keep given a certain criterion
# ==========================================================================================================


def get_rank_to_keep_from_rank_ratio(
    X: torch.tensor, S: torch.Tensor, rank_ratio: float
):
    """
    Get the rank to keep given a rank ratio.
    Basically, it is implemented as rank = ceil(rank_ratio * max_rank)
    """
    assert 0.0 <= rank_ratio <= 1.0, "rank_ratio must be in [0, 1]"
    assert X.ndim == 2, "X must be 2-dimensional"
    max_rank = min(X.shape[0], X.shape[1])
    k = math.ceil(max_rank * rank_ratio)
    return max(k, 1)


def get_rank_to_keep_from_energy_ratio(
    X: torch.Tensor, S: torch.Tensor, energy_ratio: float
) -> int:
    """
    Get the rank to keep given an energy ratio.
    The energy is defined as the sum of squares of the singular values.
    We keep the smallest rank k such that sum_{i=1}^k S[i]^2 >= energy_ratio * sum_{i=1}^r S[i]^2
    """
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
    """
    Get the rank to keep given a parameter number ratio.
    We keep the smallest rank k such that (k * (m + n)) / (min(m, n) * max(m, n)) >= param_number_ratio
    where X is in R^{m x n}.
    Careful inspection shows that this also works for (reshaped) nn.Conv2d weights.
    """
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
    W0 = U @ torch.diag(torch.sqrt(S))
    W1 = torch.diag(torch.sqrt(S)) @ V_T
    return W0, W1


def should_do_low_rank(W, rank):
    # it can be proved that rank is memory efficient <=> rank is compute efficient
    m, n = W.shape
    cost_base = m * n
    cost_low_rank = (m + n) * rank
    return cost_low_rank < cost_base


# ==========================================================================================================
# The main functions to factorize a model using regular SVD-based factorization
# =========================================================================================================


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
    rank = get_rank(reshaped, U, S, V_T)
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
    Convert the model to a low-rank model using the provided configuration dictionary.
    A criterion dictionary should be of the form:
    {
        "layer_name": {
            "name": "rank_ratio_to_keep" | "svals_energy_ratio_to_keep" | "params_ratio_to_keep",
            "value": float in [0, 1]
        },
        ...
    }
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


def collect_cache_low_rank_auto(model, keys, dataloader):
    """
    This collects information about the model and calibration data needed by the `to_low_rank_auto` function.
    Particularly, it collects the sizes of the outputs of each layer and the variance of their activations.
    """
    modules_to_replace = gather_submodules(
        model,
        should_do=keys_passlist_should_do(keys),
    )
    device = next(model.parameters()).device
    was_training = model.training

    sizes = {}
    sums = {}
    counts = {}
    hooks = []

    def hook_fn(name):
        def fn(module, inp, output):
            if isinstance(module, (nn.Conv2d, nn.LazyConv2d, nn.Linear, nn.LazyLinear)):
                bsum = output.detach().sum(dim=0).cpu()
                if name not in sums:
                    sums[name] = torch.zeros_like(bsum)
                    counts[name] = 0
                sums[name] += bsum
                counts[name] += output.shape[0]
                if name not in sizes:
                    sizes[name] = output.shape
            else:
                raise ValueError("Module should be either Conv2d or Linear")

        return fn

    for name, module in modules_to_replace:
        hooks.append(module.register_forward_hook(hook_fn(name)))

    for inputs, _ in dataloader:
        model.eval()
        with torch.no_grad():
            _ = model(inputs.to(device))

    for h in hooks:
        h.remove()

    model.train(was_training)

    variances_per_layer = {}
    for name, module in modules_to_replace:
        mean_act = (sums[name] / counts[name]).to(device)
        var = torch.var(mean_act)
        variances_per_layer[name] = var.item()

    return {"variances": variances_per_layer, "sizes": sizes}


def to_low_rank_auto(
    model: nn.Module,
    metric: str,
    ratio_to_keep: float,
    keys,
    cache,
    inplace: bool = True,
):
    """
    Convert the model to a low-rank model using a global budget.
    The budget is defined as a ratio of the total cost (in FLOPs or parameters) of the original model.
    The layers to be factorized are defined by the keys list.
    It solves a multi-choice knapsack problem to select the rank for each layer, where the objective function can
    be deduced from the code.
    """
    if not inplace:
        model = copy.deepcopy(model)

    modules_to_replace = gather_submodules(
        model,
        should_do=keys_passlist_should_do(keys),
    )

    variances = cache["variances"]
    sizes = cache["sizes"]

    sizes, variances = (
        [sizes[name] for name, _ in modules_to_replace],
        [variances[name] for name, _ in modules_to_replace],
    )
    reshaped_params = {
        name: get_reshape(module)(module.weight.detach())
        for name, module in modules_to_replace
    }
    factors = {name: decompose_params(mat) for name, mat in reshaped_params.items()}

    cum_energies = []
    for name, _ in modules_to_replace:
        S = factors[name][1]
        energy = torch.cumsum(S**2, dim=0)
        energy = energy / energy[-1]
        cum_energies.append(energy)

    cum_energies = [
        energy * (var ** (1 / 2)) for energy, var in zip(cum_energies, variances)
    ]

    ws = [mod.weight.detach() for _, mod in modules_to_replace]
    mods = [mod for _, mod in modules_to_replace]

    if metric == "rank":
        costs = [torch.arange(1, len(e) + 1, device=e.device) for e in cum_energies]
        total_budget = sum(len(e) for e in cum_energies) * ratio_to_keep
    elif metric == "flops":
        costs = [
            (
                generate_cost_flops_linear(w.shape, out_shape)
                if isinstance(mod, nn.Linear)
                else generate_cost_flops_conv2d(w.shape, out_shape)
            )
            for w, out_shape, mod in zip(ws, sizes, mods)
        ]
        total_budget = sum(c[-1].item() for c in costs) * ratio_to_keep
    elif metric == "params":
        costs = [
            (
                generate_cost_params_linear(w.shape)
                if isinstance(mod, nn.Linear)
                else generate_cost_params_conv2d(w.shape)
            )
            for w, out_shape, mod in zip(ws, sizes, mods)
        ]
        total_budget = sum(c[-1].item() for c in costs) * ratio_to_keep
    else:
        raise ValueError(
            f"Unknown metric '{metric}'. Choose from 'rank', 'flops', or 'params'."
        )

    selected_indices = maximize_energy(cum_energies, costs, total_budget)
    selected_indices_per_module = {
        name: sel for (name, _), sel in zip(modules_to_replace, selected_indices)
    }

    def factory_fn(name, module):
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

    replace_with_factory(
        model,
        {name: module for name, module in modules_to_replace},
        factory_fn,
    )
    return model


# ==========================================================================================================
# The main functions to factorize a model using whitening + SVD-based factorization for activation-aware factorization
# The concept is heavily inspired by https://arxiv.org/abs/2403.07378 (but there are differences). We use
# eigh instead of an SVD. Algebraically, they are equivalent.
# =========================================================================================================


def obtain_whitening_matrix(
    acts: torch.Tensor,
    module: nn.Module,
):
    eigenvalues, eigenvectors = torch.linalg.eigh(acts.cuda())
    eigenvalues, eigenvectors = eigenvalues.to(acts.dtype), eigenvectors.to(acts.dtype)
    x_svals = torch.sqrt(eigenvalues)
    V = eigenvectors
    keep = x_svals > 1e-10  # of shape (G, D)
    x_svals = torch.where(keep, x_svals, torch.zeros_like(x_svals))
    x_svals_inv = torch.where(keep, 1 / x_svals, torch.zeros_like(x_svals))
    return V @ torch.diag(x_svals_inv), torch.diag(x_svals) @ V.transpose(-1, -2)


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
    if factors is None:
        U, S, V_T = decompose_params(data_whitening_matrix_inverse @ reshaped)
    else:
        U, S, V_T = factors
    rank = get_rank(reshaped, U, S, V_T)
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


def _process_act(act, mod):
    if isinstance(mod, nn.Conv2d):
        # Input should be of shape (B, Cin, H, W)
        assert act.dim() == 4
        transformed = nn.functional.unfold(
            act,
            kernel_size=mod.kernel_size,
            padding=mod.padding,
            stride=mod.stride,
        )  # shape (B, Cin * H_k * W_k, H_out * W_out)
        transformed = transformed.transpose(-1, -2).reshape(
            transformed.shape[0] * transformed.shape[2], transformed.shape[1]
        )  # shape (B * H_out * W_out, Cin * H_k * W_k)
    elif isinstance(mod, nn.Linear):
        # Input should be of shape (B, Cin)
        assert act.dim() == 2 or act.dim() == 3  # for language models, [B, L, D]
        transformed = act.reshape(-1, act.shape[-1])  # shape (N, D)
    return transformed


@torch.no_grad()
def collect_cache_activation_aware(model: nn.Module, keys, dataloader):
    assert isinstance(
        dataloader, torch.utils.data.DataLoader
    ), "dataloader should be a DataLoader, got {}".format(type(dataloader))
    length = len(dataloader.dataset)
    mods = gather_submodules(model, should_do=keys_passlist_should_do(keys))
    device = next(model.parameters()).device
    acts, outsizes, hooks, inner_dim_count = {}, {}, [], {}

    def fn(n, m, inp, out):
        x = inp[0] if isinstance(inp, tuple) else inp
        a = _process_act(x.detach(), m)
        if acts.get(n) is None:
            acts[n] = torch.zeros(a.shape[1], a.shape[1], device=device, dtype=a.dtype)
        acts[n] = acts[n].to(device, non_blocking=True)
        acts[n] += (a.transpose(-1, -2) @ a) / length
        # we divide by length to avoid accumulation of very big numbers
        # but we actually need to divide by the total number of elements reduced in the inner dimension
        # this will be done later
        acts[n] = acts[n].to("cpu").detach()
        outsizes.setdefault(n, out.shape)
        inner_dim_count[n] = inner_dim_count.get(n, 0) + a.shape[0]

    for n, m in mods:
        hooks.append(m.register_forward_hook(functools.partial(fn, n)))
    state = model.training
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                model(batch[0].to(device))
            else:
                raise ValueError(
                    "Data should be a tensor or a tuple/list (as in an ImageFolder dataset)"
                )
    model.train(state)
    for h in hooks:
        h.remove()

    # now correct the acts by dividing by the total number of elements reduced in the inner dimension
    for n in acts.keys():
        acts[n] = acts[n] * (length / inner_dim_count[n])
    return {"acts": acts, "outsizes": outsizes}


def to_low_rank_manual_activation_aware(
    model: nn.Module,
    cache,
    inplace=True,
    cfg_dict={},
):
    """
    Convert the model to a low-rank model using the provided configuration dictionary.
    A criterion dictionary should be of the form:
    {
        "layer_name": {
            "name": "rank_ratio_to_keep" | "svals_energy_ratio_to_keep" | "params_ratio_to_keep",
            "value": float in [0, 1]
        },
        ...
    }

    Implements activation-aware factorization via whitening + SVD (inspired by https://arxiv.org/abs/2403.07378).
    """
    if not inplace:
        model = copy.deepcopy(model)

    acts = cache["acts"]
    modules_to_replace = gather_submodules(
        model,
        should_do=keys_passlist_should_do(cfg_dict.keys()),
    )

    whit = {
        name: obtain_whitening_matrix(acts[name], module)
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


def to_low_rank_activation_aware_auto(
    model: nn.Module,
    cache,
    keys,
    ratio_to_keep,
    metric: str = "flops",
    inplace: bool = True,
):
    """
    Similar to the `to_low_rank_auto` function, but implements activation-aware factorization.
    Note that here, we do not weight importances by feature variances, for two reasons:
    1. It works empirically.
    2. It makes sense, as the activation-aware factorization takes into account output distortion.
    """

    if not inplace:
        model = copy.deepcopy(model)

    modules_to_replace = gather_submodules(
        model,
        should_do=keys_passlist_should_do(keys),
    )

    acts = cache["acts"]
    outsizes = cache["outsizes"]

    whit = {
        name: obtain_whitening_matrix(acts[name], module)
        for name, module in modules_to_replace
    }

    outsizes = [outsizes[name] for name, _ in modules_to_replace]

    cum_energies = []
    for name, module in modules_to_replace:
        reshaped = get_reshape(module)(module.weight.detach())
        aa = whit[name][1] @ reshaped
        svals = torch.linalg.svdvals(aa)
        cum_energy = torch.cumsum(svals**2, 0)
        cum_energy = cum_energy / cum_energy[-1]
        cum_energies.append(cum_energy)

    ws = [mod.weight.detach() for _, mod in modules_to_replace]

    if metric == "rank":
        costs = [torch.arange(1, len(e) + 1, device=e.device) for e in cum_energies]
        total_budget = sum(len(e) for e in cum_energies) * ratio_to_keep
    elif metric == "flops":
        costs = [
            (
                generate_cost_flops_linear(w.shape, oshape)
                if len(oshape) == 2
                else generate_cost_flops_conv2d(w.shape, oshape)
            )
            for w, oshape in zip(ws, outsizes)
        ]
        total_budget = sum(c[-1].item() for c in costs) * ratio_to_keep
    elif metric == "params":
        costs = [
            (
                generate_cost_params_linear(w.shape)
                if len(oshape) == 2
                else generate_cost_params_conv2d(w.shape)
            )
            for w, oshape in zip(ws, outsizes)
        ]
        total_budget = sum(c[-1].item() for c in costs) * ratio_to_keep
    else:
        raise ValueError(
            f"Unknown metric '{metric}'. Choose from 'flops', 'params', 'rank'."
        )

    selected_indices = maximize_energy(cum_energies, costs, total_budget)

    selected_indices_per_module = {
        name: sel for (name, _), sel in zip(modules_to_replace, selected_indices)
    }

    def factory_fn(name, module):
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
            raise ValueError("Module should be either Linear or Conv2d")

    replace_with_factory(
        model,
        {name: module for name, module in modules_to_replace},
        factory_fn,
    )
    return model
