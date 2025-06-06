import torch.nn as nn
from typing import Callable
from tqdm import tqdm
from compress.factorization.low_rank_ops import LowRankLinear, LowRankConv2d
from compress.common import (
    gather_submodules,
    default_should_do,
    keys_passlist_should_do,
    cls_passlist_should_do,
)
from compress.utils import extract_weights
import copy
import torch
from typing import Dict
import functools
from dataclasses import dataclass


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


def to_low_rank_manual(
    model: nn.Module, inplace=True, cfg_dict: Dict[str, Dict[str, float]] = None
):
    if not inplace:
        model = copy.deepcopy(model)

    modules_to_replace = gather_submodules(
        model,
        should_do=keys_passlist_should_do(cfg_dict.keys()),
    )

    for name, module in tqdm(modules_to_replace, desc="Replacing modules"):
        parent_module = model
        *parent_path, attr_name = name.split(".")
        for part in parent_path:
            parent_module = getattr(parent_module, part)

        setattr(
            parent_module,
            attr_name,
            (
                LowRankLinear.from_linear(
                    module,
                    cfg_dict[name],
                )
                if isinstance(module, nn.Linear) or isinstance(module, nn.LazyLinear)
                else (
                    LowRankConv2d.from_conv2d(
                        module,
                        cfg_dict[name],
                    )
                    if isinstance(module, nn.Conv2d)
                    or isinstance(module, nn.LazyConv2d)
                    else module
                )
            ),
        )
    return model


def plot_singular_values(model: nn.Module, should_do: Callable = default_should_do):
    import matplotlib.pyplot as plt

    mods_and_reshapers = extract_weights_and_reshapers(
        model, cls_list=[nn.Linear, nn.Conv2d]
    )[:2]
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
        S = S / S[0]

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


def maximize_energy_pulp(
    cum_energy_vectors, cumulative_cost_vectors, total_cost, minimize=False
):
    import pulp

    # We are given N vectors of cumulative energies and of cumulative vectors. We want to, by selecting a (cumulative)
    # subset of indices from each vector (cumulative in the sense that if j is chosen, all(j' < j) are also chosen), maximize
    # the sum of energies at the selected indices such that the sum of the cumulative costs at the selected indices is less than or equal to the total cost.

    # Let x_{i, j} be a binary variable indicating whether the j-th index in the i-th vector is selected.
    # Then, we want to maximize sum_{i, j} x_{i, j} * cum_energy_vectors[i][j] subject to the constraints:
    # 1. sum_{j} x_{i, j} = 1 for all i
    # 2. sum_{i, j} j * x_{i, j} * cost_vectors[i][j] <= total_cost

    prob = (
        pulp.LpProblem("MaximizeEnergy", pulp.LpMaximize)
        if not minimize
        else pulp.LpProblem("MinimizeEnergy", pulp.LpMinimize)
    )

    selection_vars = {}
    for vec_idx, vec in enumerate(cum_energy_vectors):
        for idx in range(len(vec)):
            selection_vars[(vec_idx, idx)] = pulp.LpVariable(
                f"x_{vec_idx}_{idx}", cat="Binary"
            )
    prob += pulp.lpSum(
        selection_vars[(vec_idx, idx)] * cum_energy_vectors[vec_idx][idx].item()
        for vec_idx, vec in enumerate(cum_energy_vectors)
        for idx in range(len(vec))
    )

    prob += (
        pulp.lpSum(
            selection_vars[(vec_idx, idx)]
            * cumulative_cost_vectors[vec_idx][idx].item()
            for vec_idx, vec in enumerate(cum_energy_vectors)
            for idx in range(len(vec))
        )
        <= total_cost
    )
    for vec_idx, vec in enumerate(cum_energy_vectors):
        prob += (
            pulp.lpSum(selection_vars[(vec_idx, idx)] for idx in range(len(vec))) == 1
        )

    prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=60))

    selected_indices = {}
    for vec_idx, vec in enumerate(cum_energy_vectors):
        sel = [pulp.value(selection_vars[(vec_idx, idx)]) for idx in range(len(vec))]
        selected_indices[vec_idx] = torch.argmax(torch.tensor(sel)).item()

    print("Time to solve:", prob.solutionTime)
    print("Status:", pulp.LpStatus[prob.status])
    print("Objective value:", pulp.value(prob.objective))
    print("Exhausted metric:")
    # prints the total used cost
    print(
        sum(
            cumulative_cost_vectors[i][selected_indices[i]].item()
            for i in range(len(cum_energy_vectors))
        )
    )
    return selected_indices


def generate_cost_flops_linear(mat, size):
    # generates a vector of length s.shape[0]
    # where each element is the cost of selecting the corresponding singular value
    # in the context of a linear layer

    # a decomposed linear layer has shapes W_o in [O, R] and W_i in [R, I]
    # flops(R) = B * I * R + O * R

    r_vec = torch.arange(1, min(mat.shape[0], mat.shape[1]) + 1, 1)
    i, o = mat.shape
    return r_vec * (i + o)


def generate_cost_flops_conv2d(mat, out_size):
    # a separated convolution requires
    # flops(R) = R * C_in * H_k * W_k * H_out * W_out + C_out * R * H_out * W_out = R * H_out * W_out * (C_in * H_k * W_k + C_out)

    R = torch.arange(
        1, min(mat.shape[0], mat.shape[1] * mat.shape[2] * mat.shape[3]) + 1, 1
    )

    C_out, C_in, H_k, W_k = mat.shape
    H_out, W_out = out_size[2], out_size[3]

    return R * H_out * W_out * (C_in * H_k * W_k + C_out)


def generate_cost_params_linear(mat, size):
    # instead of flops, takes into account the number of parameters
    # a decomposed linear layer has shapes W_o in [O, R] and W_i in [R, I]
    # params(R) = R * (I + O)
    r_vec = torch.arange(1, min(mat.shape[0], mat.shape[1]) + 1, 1)
    i, o = mat.shape
    return r_vec * (i + o)


def generate_cost_params_conv2d(mat, out_size):
    # instead of flops, takes into account the number of parameters
    # a separated convolution requires
    # params(R) = R * C_in * H_k * W_k + C_out * R
    R = torch.arange(
        1, min(mat.shape[0], mat.shape[1] * mat.shape[2] * mat.shape[3]) + 1, 1
    )

    C_out, C_in, H_k, W_k = mat.shape
    return R * (C_in * H_k * W_k + C_out)


# Given a total rank ratio, estimates the rank ratio for each layer
def to_low_rank_global(
    model: nn.Module,
    ratio_to_keep,
    keys,
    sample_input,
    bn_keys=None,
    metric="flops",
    inplace=True,
    **kwargs,
):
    if not inplace:
        model = copy.deepcopy(model)

    modules_to_replace = gather_submodules(
        model,
        should_do=keys_passlist_should_do(keys),
    )
    reshapeds = []

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

    with torch.no_grad():
        model(rand_inp)

    for hook in hooks:
        hook.remove()

    assert len(sizes) == len(
        modules_to_replace
    ), "Sizes and modules to replace do not match"

    def _get_cumulative_energies(module):
        if isinstance(module, nn.Linear) or isinstance(module, nn.LazyLinear):
            weight = module.weight.detach()
            reshapeds.append(weight)
        elif isinstance(module, nn.Conv2d) or isinstance(module, nn.LazyConv2d):
            reshaped = _module_to_reshaper[(module.__class__, "weight")](module.weight)
            weight = reshaped.detach()
            reshapeds.append(reshaped)
        else:
            return None

        vals = torch.linalg.svdvals(weight)
        return torch.cumsum(vals**2, 0) / torch.sum(vals**2)  # range [0, 1]

    cum_energies = [
        _get_cumulative_energies(module) for _, module in modules_to_replace
    ]

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
        costs = [torch.arange(0, len(energy), 1) for energy in cum_energies]
        costs = [torch.cumsum(cost, 0) for cost in costs]
    elif metric == "flops":
        costs = [
            (
                generate_cost_flops_linear(w, out_size)
                if isinstance(mod, nn.Linear)
                else generate_cost_flops_conv2d(w, out_size)
            )
            / 1000000
            for w, out_size, mod in zip(ws, sizes, mods)
        ]
    elif metric == "params":
        costs = [
            (
                generate_cost_params_linear(w, out_size)
                if isinstance(mod, nn.Linear)
                else generate_cost_params_conv2d(w, out_size)
            )
            for w, out_size, mod in zip(ws, sizes, mods)
        ]

    # print("lengths of costs")
    # print([len(cost) for cost in cum_costs])
    # print("lengths of energies")
    # print([len(energy) for energy in cum_energies])
    if metric == "rank":
        n_rank_to_keep = sum(len(energy) for energy in cum_energies) * ratio_to_keep
    elif metric == "flops":
        n_rank_to_keep = sum(cost[-1].item() for cost in costs) * ratio_to_keep
    elif metric == "params":
        n_rank_to_keep = sum(cost[-1].item() for cost in costs) * ratio_to_keep

    selected_indices = maximize_energy_pulp(cum_energies, costs, n_rank_to_keep)

    selected_indices_per_module = {}
    for i, (name, module) in enumerate(modules_to_replace):
        selected_idx = selected_indices[i]
        if selected_idx == -1:
            selected_indices_per_module[name] = 1.0
        else:
            selected_indices_per_module[name] = selected_idx / len(cum_energies[i])

    for name, module in tqdm(modules_to_replace, desc="Replacing modules"):
        parent_module = model
        *parent_path, attr_name = name.split(".")
        for part in parent_path:
            parent_module = getattr(parent_module, part)
        # print("Replacing", name, "ratio=", selected_indices_per_module[name])
        setattr(
            parent_module,
            attr_name,
            (
                LowRankLinear.from_linear(
                    module,
                    keep_metric={
                        "name": "rank_ratio_to_keep",
                        "value": selected_indices_per_module[name],
                    },
                )
                if isinstance(module, nn.Linear) or isinstance(module, nn.LazyLinear)
                else LowRankConv2d.from_conv2d(
                    module,
                    keep_metric={
                        "name": "rank_ratio_to_keep",
                        "value": selected_indices_per_module[name],
                    },
                )
            ),
        )

    return model


@dataclass
class TaylorEstimationInfo:
    mean_grads: Dict[str, torch.Tensor]
    mean_grads_squareds: Dict[str, torch.Tensor]


def get_taylor_estimation_info(
    model: nn.Module, dataloader: torch.utils.data.DataLoader
) -> TaylorEstimationInfo:
    device = next(model.parameters()).device
    criterion = nn.CrossEntropyLoss(reduction="sum")

    sum_grads = {
        n: torch.zeros_like(p, device=device) for n, p in model.named_parameters()
    }
    sum_grads_sq = {
        n: torch.zeros_like(p, device=device) for n, p in model.named_parameters()
    }

    model.eval()
    for batch in tqdm(dataloader, desc="Accumulating grads"):
        model.zero_grad()
        if isinstance(batch, dict):
            batch = {k: v.to(device) for k, v in batch.items()}
            assert (
                batch["input_ids"].shape[0] == 1
            ), "Batch size should be 1 for Fisher estimation"
            outputs = model(batch["input_ids"], attention_mask=batch["attention_mask"])
            loss = criterion(outputs.logits, batch["label"])
        else:
            inputs, targets = batch
            assert inputs.shape[0] == 1, "Batch size should be 1 for Fisher estimation"
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        loss.backward()
        with torch.no_grad():
            for name, param in model.named_parameters():
                g = param.grad.detach()
                sum_grads[name] += g
                sum_grads_sq[name] += g.pow(2)

    N = len(dataloader)
    mean_grads = {n: g / N for n, g in sum_grads.items()}
    mean_grads_squareds = {n: g2 / N for n, g2 in sum_grads_sq.items()}

    model.zero_grad()
    model.eval()

    return TaylorEstimationInfo(
        mean_grads=mean_grads, mean_grads_squareds=mean_grads_squareds
    )


def to_low_rank_global2(
    model: nn.Module,
    dataloader,
    taylor_estimation_info: TaylorEstimationInfo,
    keys,
    sample_input,
    ratio_to_keep,
    metric: str = "flops",
    inplace: bool = True,
    **kwargs,
):
    """
    Taylor-based global low-rank compression (version 2) with the same cost logic
    used in `to_low_rank_global`.  *Energy logic is untouched.*
    """
    import copy
    import torch
    from tqdm import tqdm

    if not inplace:
        model = copy.deepcopy(model)

    # --- gather candidate modules -------------------------------------------------
    modules_to_replace = gather_submodules(
        model,
        should_do=keys_passlist_should_do(keys),
    )

    # --- record output shapes needed to estimate FLOPs/params ---------------------
    sizes, hooks = [], []

    def _hook_fn(module, _inp, output):
        if isinstance(module, (nn.Conv2d, nn.LazyConv2d, nn.Linear, nn.LazyLinear)):
            sizes.append(output.shape)
        else:  # should never happen
            raise ValueError("Unsupported module type inside _hook_fn")

    for _name, module in modules_to_replace:
        hooks.append(module.register_forward_hook(_hook_fn))

    rand_inp = sample_input.to(next(model.parameters()).device)
    with torch.no_grad():
        model(rand_inp)

    for h in hooks:
        h.remove()

    assert len(sizes) == len(
        modules_to_replace
    ), "`sizes` and `modules_to_replace` length mismatch"

    cum_energies = []
    ws, mods = [], []

    for name, module in modules_to_replace:
        # store raw weights for cost estimation later
        ws.append(module.weight.detach())
        mods.append(module)

        if isinstance(module, (nn.Linear, nn.LazyLinear)):
            weight = module.weight.detach()
        else:  # Conv2d / LazyConv2d
            weight = _module_to_reshaper[(module.__class__, "weight")](module.weight)

        U, S, V = torch.linalg.svd(weight)
        k = min(U.shape[1], V.shape[0])
        U, V, S = U[:, :k], V[:k, :], S[:k]

        gradsq = taylor_estimation_info.mean_grads_squareds[name + ".weight"]
        if isinstance(module, (nn.Conv2d, nn.LazyConv2d)):
            gradsq = _module_to_reshaper[(module.__class__, "weight")](gradsq)

        # = sigma_i u_i v_i^t
        # todo -- efficiently compute this
        sum_sigma_u_v = torch.zeros((k, *gradsq.shape), device=gradsq.device)
        for i in range(k):
            sum_sigma_u_v[i] = S[i] * torch.outer(U[:, i], V[i, :])

        cum_sum_sigma_u_v = torch.cumsum(sum_sigma_u_v, dim=0)
        cum_sum_sigma_u_v_inv = cum_sum_sigma_u_v[-1] - cum_sum_sigma_u_v

        uvgrads = torch.einsum("kmn,mn -> k", cum_sum_sigma_u_v_inv**2, gradsq)

        cum_energies.append(uvgrads)

    if metric == "rank":
        costs = [torch.cumsum(torch.arange(len(e)), 0) for e in cum_energies]
    elif metric == "flops":
        costs = [
            (
                generate_cost_flops_linear(w, out_size)
                if isinstance(m, (nn.Linear, nn.LazyLinear))
                else generate_cost_flops_conv2d(w, out_size)
            )
            / 1_000_000
            for w, out_size, m in zip(ws, sizes, mods)
        ]
    elif metric == "params":
        costs = [
            (
                generate_cost_params_linear(w, out_size)
                if isinstance(m, (nn.Linear, nn.LazyLinear))
                else generate_cost_params_conv2d(w, out_size)
            )
            for w, out_size, m in zip(ws, sizes, mods)
        ]
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    if metric == "rank":
        total_budget = sum(len(e) for e in cum_energies) * ratio_to_keep
    else:
        total_budget = sum(c[-1].item() for c in costs) * ratio_to_keep

    selected_indices = maximize_energy_pulp(
        cum_energies, costs, total_budget, minimize=True
    )

    selected_indices_per_module = {
        name: (
            1.0
            if selected_indices[i] == -1
            else selected_indices[i] / len(cum_energies[i])
        )
        for i, (name, _module) in enumerate(modules_to_replace)
    }

    for name, module in tqdm(modules_to_replace, desc="Replacing modules"):
        parent_module = model
        *parent_path, attr = name.split(".")
        for part in parent_path:
            parent_module = getattr(parent_module, part)

        keep_ratio = selected_indices_per_module[name]

        replacement = (
            LowRankLinear.from_linear(
                module, keep_metric={"name": "rank_ratio_to_keep", "value": keep_ratio}
            )
            if isinstance(module, (nn.Linear, nn.LazyLinear))
            else LowRankConv2d.from_conv2d(
                module, keep_metric={"name": "rank_ratio_to_keep", "value": keep_ratio}
            )
        )
        setattr(parent_module, attr, replacement)

    return model


def merge_back(model: nn.Module, inplace=True):
    if not inplace:
        model = copy.deepcopy(model)

    def _sd(mod, name):
        return isinstance(mod, LowRankLinear) or isinstance(mod, LowRankConv2d)

    modules_to_replace = gather_submodules(
        model,
        should_do=_sd,
    )
    # print("mods to replace", list(modules_to_replace))
    for name, module in tqdm(modules_to_replace, desc="Replacing modules"):
        parent_module = model
        *parent_path, attr_name = name.split(".")
        for part in parent_path:
            parent_module = getattr(parent_module, part)

        setattr(
            parent_module,
            attr_name,
            (
                LowRankLinear.to_linear(module)
                if isinstance(module, LowRankLinear)
                else (
                    LowRankConv2d.to_conv2d(module)
                    if isinstance(module, LowRankConv2d)
                    else module
                )
            ),
        )

    return model


def to_low_rank_manual_activation_aware(
    model: nn.Module,
    dataloader,
    inplace=True,
    cfg_dict=None,
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

    # get the cholesky decomposition of the covariance matrix of each activation im2col'ed in case of conv2d
    chols = {}
    for module, act in acts.items():
        if isinstance(module, nn.Conv2d):
            # Input should be of shape (B, Cin, H, W)

            assert act.dim() == 4
            im2coled = nn.functional.unfold(
                act,
                kernel_size=module.kernel_size,
                padding=module.padding,
                stride=module.stride,
            )  # shape (B, Cin * H_k * W_k, H_out * W_out)
            # shape (B * H_out * W_out, Cin * H_k * W_k)
            im2coled = im2coled.permute(0, 2, 1).reshape(
                im2coled.shape[0] * im2coled.shape[2], -1
            )
        elif isinstance(module, nn.Linear):
            # Input should be of shape (B, Cin)
            assert act.dim() == 2
            im2coled = act
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

        chols[module] = chol.float()

    for name, module in tqdm(modules_to_replace, desc="Replacing modules"):
        parent_module = model
        *parent_path, attr_name = name.split(".")
        for part in parent_path:
            parent_module = getattr(parent_module, part)

        setattr(
            parent_module,
            attr_name,
            (
                LowRankLinear.from_linear_activation(
                    module, chols[module], cfg_dict[name]
                )
                if isinstance(module, nn.Linear) or isinstance(module, nn.LazyLinear)
                else LowRankConv2d.from_conv2d_activation(
                    module, chols[module], cfg_dict[name]
                )
            ),
        )

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

    # get the cholesky decomposition of the covariance matrix of each activation im2col'ed in case of conv2d
    chols = {}
    for name, module in modules_to_replace:
        act = acts[name]
        if isinstance(module, nn.Conv2d):
            # Input should be of shape (B, Cin, H, W)

            assert act.dim() == 4
            im2coled = nn.functional.unfold(
                act,
                kernel_size=module.kernel_size,
                padding=module.padding,
                stride=module.stride,
            )  # shape (B, Cin * H_k * W_k, H_out * W_out)
            # shape (B * H_out * W_out, Cin * H_k * W_k)
            im2coled = im2coled.permute(0, 2, 1).reshape(
                im2coled.shape[0] * im2coled.shape[2], -1
            )
        elif isinstance(module, nn.Linear):
            # Input should be of shape (B, Cin)
            assert act.dim() == 2

            im2coled = act
        else:
            raise ValueError("Module should be either Conv2d or Linear")

        m = im2coled.T @ im2coled
        m = m.double()
        try:
            chol = torch.linalg.cholesky(m)

        except RuntimeError:
            print("Cholesky failed, using eigvalsh")
            eigenvalues = torch.linalg.eigvalsh(m)
            m = (-eigenvalues[0] + 1e-6) * torch.eye(m.shape[0]).to(m.device) + m
            chol = torch.linalg.cholesky(m)

        chols[module] = chol.float()
        # if conv, chols is of shape [Cin * H_k * W_k, Cin * H_k * W_k]
        # if linear, chols is of shape [Cin, Cin]

    # energies
    cum_energies = []
    for name, module in modules_to_replace:
        if isinstance(module, nn.Linear):
            weight = module.weight.detach()  # shape (Cout, Cin)
        elif isinstance(module, nn.Conv2d):
            reshaped = _module_to_reshaper[(module.__class__, "weight")](module.weight)
            weight = reshaped.detach().T  # shape (Cout, Cin * H_k * W_k)

        aa = weight @ chols[module]

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
        costs = [torch.arange(0, len(energy), 1) for energy in cum_energies]
        costs = [torch.cumsum(cost, 0) for cost in costs]

    elif metric == "flops":
        costs = [
            (
                generate_cost_flops_linear(w, out_size)
                if len(out_size) == 2
                else generate_cost_flops_conv2d(w, out_size)
            )
            / 1000000
            for w, out_size in zip(
                ws, [outs[name].shape for name, _ in modules_to_replace]
            )
        ]

    elif metric == "params":
        costs = [
            (
                generate_cost_params_linear(w, out_size)
                if len(out_size) == 2
                else generate_cost_params_conv2d(w, out_size)
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

    selected_indices = maximize_energy_pulp(cum_energies, costs, n_rank_to_keep)
    selected_indices_per_module = {}
    for i, (name, module) in enumerate(modules_to_replace):
        selected_idx = selected_indices[i]
        if selected_idx == -1:
            selected_indices_per_module[name] = 1.0
        else:
            selected_indices_per_module[name] = selected_idx / len(cum_energies[i])

    for name, module in tqdm(modules_to_replace, desc="Replacing modules"):
        parent_module = model
        *parent_path, attr_name = name.split(".")
        for part in parent_path:
            parent_module = getattr(parent_module, part)

        metric = {
            "name": "rank_ratio_to_keep",
            "value": selected_indices_per_module[name],
        }
        # print("Replacing", name, "ratio=", selected_indices_per_module[name])
        setattr(
            parent_module,
            attr_name,
            (
                LowRankLinear.from_linear_activation(
                    module,
                    chols[module],
                    metric,
                )
                if isinstance(module, nn.Linear) or isinstance(module, nn.LazyLinear)
                else LowRankConv2d.from_conv2d_activation(
                    module,
                    chols[module],
                    metric,
                )
            ),
        )

    return model
