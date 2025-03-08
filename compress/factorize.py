import torch.nn as nn
from typing import Callable
from tqdm import tqdm
from compress.low_rank_ops import LowRankLinear, LowRankConv2d
from compress.common import gather_submodules, default_should_do
from compress.utils import extract_weights
import copy
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
            (
                LowRankLinear.from_linear(module, **kwargs)
                if isinstance(module, nn.Linear) or isinstance(module, nn.LazyLinear)
                else LowRankConv2d.from_conv2d(module, **kwargs)
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

    return selected_indices


# Given a total rank ratio, estimates the rank ratio for each layer
def to_low_rank_global(
    model: nn.Module, should_do: Callable = default_should_do, inplace=True, **kwargs
):
    if not inplace:
        model = copy.deepcopy(model)
    modules_to_replace = gather_submodules(model, should_do=should_do, prefix="")

    def _get_cumulative_energies(module):
        if isinstance(module, nn.Linear) or isinstance(module, nn.LazyLinear):
            weight = module.weight.detach()
        elif isinstance(module, nn.Conv2d) or isinstance(module, nn.LazyConv2d):
            reshaped = _module_to_reshaper[(module.__class__, "weight")](module.weight)
            weight = reshaped.detach()
        else:
            return None

        vals = torch.linalg.svdvals(weight)
        return torch.cumsum(vals**2, 0)

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


def get_activation_norms(model, data_loader):
    if isinstance(data_loader, torch.Tensor):
        data_loader = [data_loader]
    activation_norms = {}
    hooks = []
    model.eval()
    for name, module in gather_submodules(
        model, should_do=default_should_do, prefix=""
    ):
        activation_norms[module] = 0

        def hook_fn(activations):
            def _hook_fn(module, input, output):
                # first flatten the activations
                if isinstance(module, nn.Linear) or isinstance(module, nn.LazyLinear):
                    act = output.view(output.size(0), -1)  # (batch_size, hidden_size)

                    def _norm(x):
                        # consider all dims batched except the last one
                        norm2 = torch.sum(x**2, dim=-1)
                        return torch.sqrt(norm2).mean()

                    activation_norms[module] = (
                        activation_norms[module] + _norm(act).detach()
                    )
                elif isinstance(module, nn.Conv2d) or isinstance(module, nn.LazyConv2d):
                    raise NotImplementedError("Conv2d activations not supported yet")

            return _hook_fn

        hooks.append(module.register_forward_hook(hook_fn(activation_norms[module])))

    for element in tqdm(data_loader, desc="Getting activation norms"):
        model(element["input_ids"].cuda(), element["attention_mask"].cuda())

    for hook in hooks:
        hook.remove()

    return {module: norm.item() for module, norm in activation_norms.items()}


def get_layer_order(model, data_loader):
    if isinstance(data_loader, torch.Tensor):
        data_loader = [data_loader]
    hooks = []
    model.eval()
    layers_called = list()
    for name, module in gather_submodules(
        model, should_do=default_should_do, prefix=""
    ):

        def hook_fn(activations):
            def _hook_fn(module, input, output):
                layers_called.append({"module": module})

            return _hook_fn

        hooks.append(module.register_forward_hook(hook_fn(None)))
    element = next(iter(data_loader))
    model(element["input_ids"].cuda(), element["attention_mask"].cuda())

    for hook in hooks:
        hook.remove()

    return layers_called


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

    modules_to_replace = gather_submodules(model, should_do=should_do, prefix="")
    acts = get_activation_norms(model, dataloader)
    layers_called = get_layer_order(model, dataloader)

    # filter out layers called that are not in modules_to_replace
    layers_called = [
        layer
        for layer in layers_called
        if any(layer["module"] == module for _, module in modules_to_replace)
    ]
    all_singular_vals = {}
    for name, module in modules_to_replace:
        if isinstance(module, nn.Linear) or isinstance(module, nn.LazyLinear):
            weight = module.weight.detach()
        elif isinstance(module, nn.Conv2d) or isinstance(module, nn.LazyConv2d):
            reshaped = _module_to_reshaper[(module.__class__, "weight")](module.weight)
            weight = reshaped.detach()
        else:
            continue

        vals = torch.linalg.svdvals(weight)
        all_singular_vals[module] = vals

    first_singular_vals = {
        module: torch.sqrt(torch.mean(vals**2))
        for module, vals in all_singular_vals.items()
    }

    last_layer_idx = len(layers_called) - 1

    def mul(seq):
        res = 1
        for i in seq:
            res *= i
        return res

    def _get_cumulative_energies(module, name):
        # first term is mult(layer_idx, last_layer_idx, val=singular_val_1[layer_idx])
        # layers from layer_idx to last_layer_idx
        layers_from_to = []
        firstidx = layers_called.index({"module": module})
        for i in range(firstidx, last_layer_idx + 1):
            layers_from_to.append({"module": layers_called[i]["module"]})

        # get the singular values for the layers
        singvals = [first_singular_vals[layer["module"]] for layer in layers_from_to]

        first_term = mul(singvals)

        # second term is ||f(x, W1:Wthislayer)||_2
        second_term = acts[module]

        # now, the energy obtained from choosing a rank on this layer is first_term * second_term * sing_val[rank_chosen]

        return -all_singular_vals[module] * second_term * first_term

    cum_energies = [
        _get_cumulative_energies(module, name) for name, module in modules_to_replace
    ]

    # assert none is nan/inf
    for energy in cum_energies:
        assert (
            not torch.isnan(energy).any() and not torch.isinf(energy).any()
        ), "Energy is nan or inf: {}".format(energy)
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
