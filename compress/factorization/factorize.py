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
    W0 = U @ torch.diag(torch.sqrt(S))
    W1 = torch.diag(torch.sqrt(S)) @ V_T
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
    ratio_to_keep: float,
    keys,
    sample_input: torch.Tensor,
    inplace: bool = True,
):
    if not inplace:
        model = copy.deepcopy(model)

    modules_to_replace = gather_submodules(
        model,
        should_do=keys_passlist_should_do(keys),
    )

    device = next(model.parameters()).device
    rand_inp = sample_input.to(device)

    hooks, sizes = [], []

    def hook_fn(module, _, output):
        if isinstance(module, (nn.Conv2d, nn.LazyConv2d, nn.Linear, nn.LazyLinear)):
            sizes.append(output.shape)
        else:
            raise ValueError("Module should be either Conv2d or Linear")

    for _, module in modules_to_replace:
        hooks.append(module.register_forward_hook(hook_fn))

    was_training = model.training
    model.eval()
    with torch.no_grad():
        _ = model(rand_inp)
    for h in hooks:
        h.remove()
    model.train(was_training)

    reshaped_params = {
        name: get_reshape(module)(module.weight.detach())
        for name, module in modules_to_replace
    }
    factors = {name: decompose_params(mat) for name, mat in reshaped_params.items()}

    cum_energies = []
    entropies = []
    for name, _ in modules_to_replace:
        S = factors[name][1]
        energy = torch.cumsum(S**2, dim=0)
        energy /= energy[-1]
        cum_energies.append(energy)

        p = (S**2) / torch.sum(S**2)
        H = -(p * torch.log(p + 1e-12)).sum()
        entropies.append(H)

    cum_energies = [
        energy * torch.sqrt(w) for energy, w in zip(cum_energies, entropies)
    ]

    ws = [mod.weight.detach() for _, mod in modules_to_replace]
    mods = [mod for _, mod in modules_to_replace]

    if metric == "rank":
        costs = [
            torch.cumsum(torch.arange(1, len(e) + 1, device=e.device), 0)
            for e in cum_energies
        ]
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
    V_nz = Vh[keep, :].T

    return V_nz @ torch.diag(1 / S_nz), torch.diag(S_nz) @ V_nz.T


def obtain_whitening_matrix_eigh(
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

    m = im2coled.T @ im2coled
    eigenvalues, eigenvectors = torch.linalg.eigh(m)
    x_svals = torch.sqrt(eigenvalues)
    V = eigenvectors
    keep = x_svals > 1e-6
    x_svals = x_svals[keep]
    V = V[:, keep]

    return V @ torch.diag(1 / x_svals), torch.diag(x_svals) @ V.T


def obtain_whitening_matrix(
    acts: torch.Tensor,
    module: nn.Module,
    method: str = "eigh",
):
    """
    Computes the whitening matrix for the given activations and module.
    The method can be "cholesky", "svd", or "eigh". "cholesky" will return non-precise results if the covariance matrix is not positive definite.
    """
    if method == "cholesky":
        return obtain_whitening_matrix_cholesky(acts, module)
    elif method == "svd":
        return obtain_whitening_matrix_svd(acts, module)
    elif method == "eigh":
        return obtain_whitening_matrix_eigh(acts, module)
    else:
        raise ValueError("Method must be one of 'cholesky', 'svd', or 'eigh'.")


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
    model: nn.Module,
    dataloader,
    inplace=True,
    cfg_dict={},
    data_whitening_impl="eigh",
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
        name: obtain_whitening_matrix(acts[module], module, method=data_whitening_impl)
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


def to_low_rank_activation_aware_global(
    model: nn.Module,
    dataloader,
    keys,
    ratio_to_keep,
    metric: str = "flops",
    inplace: bool = True,
    data_whitening_impl: str = "eigh",
):
    if not inplace:
        model = copy.deepcopy(model)

    device = next(model.parameters()).device

    acts: Dict[str, torch.Tensor] = {}
    outs: Dict[str, torch.Tensor] = {}
    hooks = []

    modules_to_replace = gather_submodules(
        model,
        should_do=keys_passlist_should_do(keys),
    )

    def hook_fn(name, module, input, output):
        x = input[0] if isinstance(input, tuple) else input

        if name not in acts:
            acts[name] = x.detach()
        else:
            acts[name] = torch.cat((acts[name], x.detach()), dim=0)

        if name not in outs:
            outs[name] = output.detach()
        else:
            outs[name] = torch.cat((outs[name], output.detach()), dim=0)

    for name, module in modules_to_replace:
        hooks.append(module.register_forward_hook(functools.partial(hook_fn, name)))

    prev_state = model.training
    model.eval()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Getting activations"):
            if isinstance(batch, dict):
                inputs = {k: v.to(device) for k, v in batch.items()}
                _ = model(**inputs)
            else:
                inputs, _ = batch if len(batch) == 2 else (batch, None)
                inputs = inputs.to(device)
                _ = model(inputs)

    model.train(prev_state)

    for h in hooks:
        h.remove()

    whit = {
        name: obtain_whitening_matrix(acts[name], module, method=data_whitening_impl)
        for name, module in modules_to_replace
    }

    cum_energies = []
    for name, module in modules_to_replace:
        reshaped = get_reshape(module)(module.weight.detach())
        aa = whit[name][1] @ reshaped
        svals = torch.linalg.svdvals(aa)
        cum_energy = torch.cumsum(svals**2, 0) / torch.sum(svals**2)
        cum_energies.append(cum_energy)

    act_vars = []
    for name, module in modules_to_replace:
        x = acts[name].float()
        dims = tuple(range(x.ndim))
        ex = x.mean(dims)
        ex2 = (x * x).mean(dims)
        global_var = ex2 - ex.pow(2)
        act_vars.append(global_var.mean())

    cum_energies = [
        energy * torch.sqrt(var.to(energy.device))
        for energy, var in zip(cum_energies, act_vars)
    ]

    ws = [mod.weight.detach() for _, mod in modules_to_replace]
    out_shapes = [outs[name].shape for name, _ in modules_to_replace]

    if metric == "rank":
        costs = [
            torch.cumsum(torch.arange(1, len(e) + 1, device=e.device), 0)
            for e in cum_energies
        ]
        total_budget = sum(len(e) for e in cum_energies) * ratio_to_keep
    elif metric == "flops":
        costs = [
            (
                generate_cost_flops_linear(w.shape, oshape)
                if len(oshape) == 2
                else generate_cost_flops_conv2d(w.shape, oshape)
            )
            for w, oshape in zip(ws, out_shapes)
        ]
        total_budget = sum(c[-1].item() for c in costs) * ratio_to_keep
    elif metric == "params":
        costs = [
            (
                generate_cost_params_linear(w.shape)
                if len(oshape) == 2
                else generate_cost_params_conv2d(w.shape)
            )
            for w, oshape in zip(ws, out_shapes)
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
            return module

    replace_with_factory(
        model,
        {name: module for name, module in modules_to_replace},
        factory_fn,
    )
    return model
