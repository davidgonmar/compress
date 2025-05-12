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


def maximize_energy_pulp(cum_energy_vectors, j):
    import pulp

    # We are given N vectors of cumulative energies. We want to, by selecting a total of j indices for all vectors (consecutive in each vector),
    # maximize the sum of energies at the selected indices. This can be formulated as a binary linear program:
    # Let x_{i, j} be a binary variable indicating whether the j-th index in the i-th vector is selected.
    # Then, we want to maximize sum_{i, j} x_{i, j} * cum_energy_vectors[i][j] subject to the constraints:
    # 1. sum_{j} x_{i, j} = 1 for all i
    # 2. sum_{i, j} j * x_{i, j} = j
    prob = pulp.LpProblem("MaximizeEnergy", pulp.LpMaximize)

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
            idx * selection_vars[(vec_idx, idx)]
            for vec_idx, vec in enumerate(cum_energy_vectors)
            for idx in range(len(vec))
        )
        == j
    )
    for vec_idx, vec in enumerate(cum_energy_vectors):
        prob += (
            pulp.lpSum(selection_vars[(vec_idx, idx)] for idx in range(len(vec))) == 1
        )

    prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=10))

    selected_indices = {}
    for vec_idx, vec in enumerate(cum_energy_vectors):
        sel = [pulp.value(selection_vars[(vec_idx, idx)]) for idx in range(len(vec))]
        selected_indices[vec_idx] = torch.argmax(torch.tensor(sel)).item()

    print("Time to solve:", prob.solutionTime)
    print("Status:", pulp.LpStatus[prob.status])
    print("Objective value:", pulp.value(prob.objective))
    return selected_indices


# Given a total rank ratio, estimates the rank ratio for each layer
def to_low_rank_global(
    model: nn.Module, should_do: Callable = default_should_do, inplace=True, **kwargs
):
    modules_to_replace = gather_submodules(
        model,
        should_do=should_do,
    )
    if not inplace:
        model = copy.deepcopy(model)

    def _get_cumulative_energies(module):
        if isinstance(module, nn.Linear) or isinstance(module, nn.LazyLinear):
            weight = module.weight.detach()
        elif isinstance(module, nn.Conv2d) or isinstance(module, nn.LazyConv2d):
            reshaped = _module_to_reshaper[(module.__class__, "weight")](module.weight)
            weight = reshaped.detach()
        else:
            return None

        vals = torch.linalg.svdvals(weight)
        return torch.cumsum(vals**2, 0) / torch.sum(vals**2)  # range [0, 1]

    cum_energies = [
        _get_cumulative_energies(module) for _, module in modules_to_replace
    ]

    n_rank_to_keep = (
        sum(len(energy) for energy in cum_energies) * kwargs["ratio_to_keep"]
    )

    selected_indices = maximize_energy_pulp(cum_energies, n_rank_to_keep)

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

        setattr(
            parent_module,
            attr_name,
            (
                LowRankLinear.from_linear(
                    module, ratio_to_keep=selected_indices_per_module[name]
                )
                if isinstance(module, nn.Linear) or isinstance(module, nn.LazyLinear)
                else LowRankConv2d.from_conv2d(
                    module, ratio_to_keep=selected_indices_per_module[name]
                )
            ),
        )

    return model


def get_grads(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader):
    oneitem = next(iter(dataloader))
    if isinstance(oneitem, dict):
        # huggingface dataset
        model.train()
        model.zero_grad()
        device = next(model.parameters()).device
        crit = torch.nn.CrossEntropyLoss()
        for batch in tqdm(dataloader, desc="Getting grads"):
            inputs = {key: value.to(device) for key, value in batch.items()}
            outputs = model(
                inputs["input_ids"], attention_mask=inputs["attention_mask"]
            )
            loss = crit(outputs.logits, inputs["label"])
            loss.backward()

        with torch.no_grad():
            mean_grads = {
                name: param.grad.div_(len(dataloader))
                for name, param in model.named_parameters()
            }

        model.zero_grad()
        model.eval()
        return mean_grads
    else:  # torchvision cifar10
        model.train()
        model.zero_grad()
        device = next(model.parameters()).device
        crit = torch.nn.CrossEntropyLoss()
        for batch in tqdm(dataloader, desc="Getting grads"):
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = crit(outputs, targets)
            loss.backward()

        with torch.no_grad():
            mean_grads = {
                name: param.grad.div_(len(dataloader))
                for name, param in model.named_parameters()
            }

        model.zero_grad()
        model.eval()
        return mean_grads


# Given a total rank ratio, estimates the rank ratio for each layer
def to_low_rank_global2(
    model: nn.Module,
    dataloader,
    should_do: Callable = default_should_do,
    inplace=True,
    **kwargs,
):

    if not inplace:
        model = copy.deepcopy(model)
    modules_to_replace = gather_submodules(
        model,
        should_do=should_do,
    )
    grads = get_grads(model, dataloader)

    cum_energies = []
    for name, module in modules_to_replace:
        if isinstance(module, nn.Linear) or isinstance(module, nn.LazyLinear):
            weight = module.weight.detach()
        elif isinstance(module, nn.Conv2d) or isinstance(module, nn.LazyConv2d):
            reshaped = _module_to_reshaper[(module.__class__, "weight")](module.weight)
            weight = reshaped.detach()
        else:
            continue

        U, S, V = torch.linalg.svd(weight)
        k = min(U.shape[1], V.shape[0])
        # get k vectors from u and v
        U = U[:, :k]  # shape (m, k)
        V = V[:k, :]  # shape (k, n)
        # get k singular values
        S = S[:k]  # shape (k,)

        grad = grads[name + ".weight"]
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.LazyConv2d):
            grad = _module_to_reshaper[(module.__class__, "weight")](grad)

        uvgrads = torch.einsum("k, mk, kn, mn -> k", S, U, V, grad)  # shape (k,)
        total = torch.sum(uvgrads)
        cum_energy = total - torch.cumsum(uvgrads, 0)  # shape (k,)
        cum_energies.append(cum_energy)

    n_rank_to_keep = (
        sum(len(energy) for energy in cum_energies) * kwargs["ratio_to_keep"]
    )

    selected_indices = maximize_energy_pulp(cum_energies, n_rank_to_keep)

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

        setattr(
            parent_module,
            attr_name,
            (
                LowRankLinear.from_linear(
                    module, ratio_to_keep=selected_indices_per_module[name]
                )
                if isinstance(module, nn.Linear) or isinstance(module, nn.LazyLinear)
                else LowRankConv2d.from_conv2d(
                    module, ratio_to_keep=selected_indices_per_module[name]
                )
            ),
        )

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


def factorize_with_activation_aware_svd(
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


def hoyer_svd_sparsity_grad_adder(params: torch.Tensor, weight):
    for param in params:
        if param.dim() not in [2, 4]:
            continue
        if param.dim() == 4:
            # reshape to (O, I * H * W) from (O, I, H, W)
            param_rs = param.reshape(param.shape[0], -1)
        else:
            param_rs = param
        U, S, Vt = torch.linalg.svd(param_rs)
        fro = S.pow(2).sum().sqrt()
        nuc = S.sum()
        # crop either U or Vt
        if U.shape[0] > Vt.shape[1]:
            U = U[:, : Vt.shape[1]]
        else:
            Vt = Vt[: U.shape[0], :]
        update = (1 / fro) * ((U @ Vt) - (nuc / (fro**2)) * param_rs)
        if len(param.shape) == 4:
            # reshape back to (O, I, H, W)
            update = update.reshape(
                param.shape[0], param.shape[1], param.shape[2], param.shape[3]
            )
        param.grad = (param.grad if param.grad is not None else 0) + (weight * update)


def hoyer_svd_sparsity_grad_adder_given_svds(params, svds, weight):
    for name, param in params:
        if param.dim() not in [2, 4]:
            continue
        if param.dim() == 4:
            # reshape to (O, I * H * W) from (O, I, H, W)
            param_rs = param.reshape(param.shape[0], -1)
        else:
            param_rs = param
        U, S, Vt = svds[name]
        fro = S.pow(2).sum().sqrt()
        nuc = S.sum()
        # crop either U or Vt
        if U.shape[0] > Vt.shape[1]:
            U = U[:, : Vt.shape[1]]
        else:
            Vt = Vt[: U.shape[0], :]
        update = (1 / fro) * ((U @ Vt) - (nuc / (fro**2)) * param_rs)
        if len(param.shape) == 4:
            # reshape back to (O, I, H, W)
            update = update.reshape(
                param.shape[0], param.shape[1], param.shape[2], param.shape[3]
            )
        param.grad = (param.grad if param.grad is not None else 0) + (weight * update)
