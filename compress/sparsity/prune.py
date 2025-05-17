from torch import nn
from compress.sparsity.sparse_ops import PrunedLinear, PrunedConv2d
from compress.sparsity.groupers import (
    OutChannelGroupingGrouperConv2d,
    OutChannelGroupingGrouperLinear,
)
import torch
from typing import Dict
from collections import defaultdict
from compress.sparsity.policy import PolicyDict
from compress.sparsity.runner import Runner


def apply_masks(
    model: nn.Module,
    masks: Dict[str, torch.Tensor],
    allow_reprune: bool = True,
) -> nn.Module:
    def _recursive_apply(parent: nn.Module, prefix: str = ""):
        for child_name, child_mod in parent.named_children():
            # build the “full” name of this submodule
            full_name = prefix + child_name if not prefix else prefix + "." + child_name
            # if there’s a mask for this module, replace it
            if full_name in masks:
                mask = masks[full_name]
                if isinstance(child_mod, nn.Conv2d):
                    pruned = PrunedConv2d.from_conv2d(child_mod, mask=mask)
                    setattr(parent, child_name, pruned)
                    child_mod = pruned
                elif isinstance(child_mod, nn.Linear):
                    pruned = PrunedLinear.from_linear(child_mod, mask=mask)
                    setattr(parent, child_name, pruned)
                    child_mod = pruned
                elif isinstance(child_mod, (PrunedConv2d, PrunedLinear)):
                    if not allow_reprune:
                        raise ValueError(
                            f"Module {full_name} is already pruned, and allow_reprune is False"
                        )
                    # if the module is already pruned, just update the mask
                    assert child_mod.mask.shape == mask.shape, (
                        child_mod.mask.shape,
                        mask.shape,
                        child_mod.mask.device,
                        mask.device,
                    )
                    child_mod.mask = mask
            # recurse into (possibly replaced) child
            _recursive_apply(child_mod, full_name)

    _recursive_apply(model)
    return model


class AbstractPruner:
    def __init__(self):
        pass

    def prune(self):
        raise NotImplementedError


class MagnitudeIntraGroupPruner(AbstractPruner):
    def __init__(self, model: nn.Module, policies: PolicyDict):
        self.model = model
        self.policies = policies

    def prune(self):
        # iterate over policies and apply pruning
        mods = dict(self.model.named_modules())
        masks = dict()
        for name, policy in self.policies.items():
            module = mods[name]
            assert isinstance(
                module, (nn.Conv2d, nn.Linear)
            ), f"Module {name} is not nn.Conv2d, nn.Linear or PrunedConv2d, PrunedLinear"
            reshaped = policy.grouper.transform(
                module.weight
            )  # shape [n_groups, m_elements_per_group]
            # prune the elements in each group, so we end up with n_groups
            assert policy.intra_group_metric.name in ["sparsity_ratio", "threshold"]

            if policy.intra_group_metric.name == "sparsity_ratio":
                sparsity_ratio = policy.intra_group_metric.value
                ranked_elements = torch.topk(
                    reshaped.abs(),
                    k=int(sparsity_ratio * reshaped.shape[1]),
                    dim=1,
                ).indices  # n_groups, int(sparsity_ratio * m_elements_per_group)
                mask = torch.zeros_like(reshaped, dtype=torch.bool)
                mask.scatter_(1, ranked_elements, 1)
                # now we need to untransform the mask
                mask = policy.grouper.untransform(mask, module.weight)
                masks[name] = mask
            elif policy.intra_group_metric.name == "threshold":
                threshold = policy.intra_group_metric.value
                keep = reshaped.abs() > threshold
                mask = keep.to(reshaped.device)
                # now we need to untransform the mask
                mask = policy.grouper.untransform(mask, module.weight)
                masks[name] = mask
        return apply_masks(self.model, masks)


class NormInterGroupPruner(AbstractPruner):
    def __init__(self, model: nn.Module, policies: PolicyDict):
        self.model = model
        self.policies = policies

    def prune(self):
        # iterate over policies and apply pruning
        mods = dict(self.model.named_modules())
        masks = dict()
        for name, policy in self.policies.items():
            module = mods[name]
            assert isinstance(
                module, (nn.Conv2d, nn.Linear)
            ), f"Module {name} is not a Linear or Conv2d"
            reshaped = policy.grouper.transform(
                module.weight
            )  # n_groups, m_elements_per_group
            # prune the elements in each group, so we end up with n_groups
            metric = policy.inter_group_metric
            assert metric.name in ["sparsity_ratio", "threshold"]
            if metric.name == "sparsity_ratio":
                sparsity_ratio = metric.value
                ranked_elements = torch.topk(
                    reshaped.norm(dim=1),  # shape [n_groups]
                    k=int(sparsity_ratio * reshaped.shape[0]),
                    dim=0,
                ).indices
                mask = torch.zeros(
                    (reshaped.shape[0],), dtype=torch.bool, device=reshaped.device
                )
                mask.scatter_(0, ranked_elements, 1)
                # now we need to untransform the mask and expand it to the original shape
                mask = mask.unsqueeze(1).expand(-1, reshaped.shape[1])
                mask = policy.grouper.untransform(mask, module.weight)
                masks[name] = mask
            elif metric.name == "threshold":
                threshold = metric.value
                keep = reshaped.norm(dim=1) > threshold
                mask = keep.to(reshaped.device)
                # now we need to untransform the mask
                # expand the mask
                mask = mask.unsqueeze(1).expand(-1, reshaped.shape[1])
                mask = policy.grouper.untransform(mask, module.weight)
                masks[name] = mask
        return apply_masks(self.model, masks)


class ActivationInterGroupPruner:
    def __init__(self, model: nn.Module, policies: PolicyDict, runner: Runner):
        self.model = model
        self.policies = policies
        self.runner = runner

    def prune(self):
        activations = {}

        def get_hook(name):
            def hook(module, input, output):
                if name not in self.activations:
                    activations[name] = output.detach().mean(dim=0).unsqueeze(0)
                else:
                    activations[name] += output.detach().mean(dim=0).unsqueeze(0)

            return hook

        # hook for policied modules
        for name, module in self.model.named_modules():
            if name in self.policies.keys():
                assert isinstance(
                    module, (nn.Conv2d, nn.Linear)
                ), f"Module {name} is not a Linear or Conv2d"
                module.register_forward_hook(get_hook(name))

        while True:
            try:
                loss = self.runner.iteration()
                self.model.zero_grad()
                loss.backward()
            except StopIteration:
                break

        masks = dict()

        for name, module in self.model.named_modules():
            if name in self.policies.keys():
                assert isinstance(
                    module, (nn.Conv2d, nn.Linear)
                ), f"Module {name} is not a Linear or Conv2d"
                # if conv -> weight shape [O, I, H_k, W_k]
                # if linear -> weight shape [O, I]
                acts = self.activations[name].mean(0)
                policy = self.policies[name]
                if isinstance(module, nn.Conv2d):
                    assert isinstance(
                        policy.grouper, OutChannelGroupingGrouperConv2d
                    ), "Policy grouper is not a OutChannelGroupingGrouperConv2d"
                    reshaped_acts = acts.reshape(
                        acts.shape[0], -1
                    )  # shape [O, H_out * W_out]
                    out_score = torch.norm(reshaped_acts, dim=1)  # shape [O]
                    # now we have the saliencies for this layer
                    saliencies = out_score
                    if policy.inter_group_metric["name"] == "sparsity_ratio":
                        sparsity_ratio = policy.inter_group_metric["value"]
                        # shape [O]
                        selector = torch.topk(
                            saliencies,
                            k=int(sparsity_ratio * saliencies.shape[0]),
                            dim=0,
                        ).indices
                        mask = torch.zeros_like(saliencies, dtype=torch.bool)
                        mask.scatter_(0, selector, 1)
                        # broadcast mask back to original shape
                        o, i, hk, wk = module.weight.shape
                        mask = mask.reshape(o, 1, 1, 1).expand(-1, i, hk, wk)
                    elif policy.inter_group_metric["name"] == "threshold":
                        threshold = policy.inter_group_metric["value"]
                        keep = saliencies > threshold
                        mask = keep.to(saliencies.device)
                        # broadcast mask back to original shape
                        o, i, hk, wk = module.weight.shape
                        mask = mask.reshape(o, 1, 1, 1).expand(-1, i, hk, wk)
                else:  # if isinstance(module, nn.Linear):
                    assert isinstance(
                        policy.grouper, OutChannelGroupingGrouperLinear
                    ), "Policy grouper is not a OutChannelGroupingGrouperLinear"

                    # linear
                    # acts of shape [..., I]
                    # weight of shape [O, I]
                    # reshape acts to [combine(...), O]
                    acts = acts.reshape(-1, acts.shape[-1])
                    norm = torch.norm(acts, dim=0, keepdim=False)  # shape [0]
                    saliencies = norm
                    if policy.inter_group_metric["name"] == "sparsity_ratio":
                        sparsity_ratio = policy.inter_group_metric["value"]
                        # shape [O]
                        selector = torch.topk(
                            saliencies,
                            k=int(sparsity_ratio * saliencies.shape[0]),
                            dim=0,
                        ).indices
                        mask = torch.zeros_like(saliencies, dtype=torch.bool)
                        mask.scatter_(0, selector, 1)
                        # broadcast mask back to original shape
                        o = module.weight.shape[0]
                        mask = mask.reshape(o, 1).expand(module.weight.shape)
                    elif policy.inter_group_metric["name"] == "threshold":
                        threshold = policy.inter_group_metric["value"]
                        keep = saliencies > threshold
                        mask = keep.to(saliencies.device)
                        # broadcast mask back to original shape
                        o = module.weight.shape[0]
                        mask = mask.reshape(o, 1).expand(module.weight.shape)
                mask = policy.grouper.untransform(mask, module.weight)
                masks[name] = mask
        return apply_masks(self.model, masks)


class TaylorIntraExpansionPruner:
    def __init__(
        self, model, policies, runner, n_iters, approx="fisher", *args, **kwargs
    ):
        # runner runs one batch of data through the model

        self.model = model
        self.policies = policies
        self.runner = runner
        self.args = args
        self.kwargs = kwargs
        self.approx = approx
        self.n_iters = n_iters

    def dispose(self):
        self.model = None

    def prune(self):
        grads = defaultdict(
            lambda: torch.tensor(0).to(next(self.model.parameters()).device)
        )
        fisher_hessian = defaultdict(
            lambda: torch.tensor(0).to(next(self.model.parameters()).device)
        )
        for i in range(self.n_iters):
            loss = self.runner()
            self.model.zero_grad()
            loss.backward()
            for name, mod in self.model.named_modules():
                if name in self.policies.keys():
                    assert isinstance(
                        mod, (nn.Conv2d, nn.Linear)
                    ), f"Module {name} is not a Linear or Conv2d"
                    grads[name] = (
                        grads[name] + (mod.weight.grad.data / self.n_iters).detach()
                    )
                    fisher_hessian[name] = (
                        fisher_hessian[name]
                        + (mod.weight.grad.data**2 / self.n_iters).detach()
                    )

        with torch.no_grad():
            # now we have the gradients for each layer
            assert self.approx == "fisher"

            # now we have the fisher information for each layer

            # perturbation = grad + 1/2 * fisher * (param ** 2)
            saliencies = {}

            for name, mod in self.model.named_modules():
                if name in self.policies.keys():
                    saliencies[name] = (
                        +1 / 2 * fisher_hessian[name] * (mod.weight.data**2)
                    )

            # now we have the saliencies for each layer
            # we can use the saliencies to prune the model
            masks = dict()
            for name, policy in self.policies.items():
                module = dict(self.model.named_modules())[name]
                assert isinstance(
                    module, (nn.Conv2d, nn.Linear)
                ), f"Module {name} is not a Linear or Conv2d"
                reshaped = policy.grouper.transform(module.weight)
                saliency_reshaped = policy.grouper.transform(saliencies[name])
                # first prune the elements in groups, so we end up with n_groups, m_elements_per_group where the last dim has ordered by magnitude idxs
                intra = policy.intra_group_metric["value"]
                assert policy.intra_group_metric["name"] == "sparsity_ratio"
                ranked_elements = torch.topk(
                    saliency_reshaped,
                    k=int(intra * reshaped.shape[1]),
                    dim=1,
                ).indices
                mask = torch.zeros_like(reshaped, dtype=torch.bool)
                mask.scatter_(1, ranked_elements, 1)
                # now we need to untransform the mask
                mask = policy.grouper.untransform(mask, module.weight)
                masks[name] = mask

        return apply_masks(self.model, masks)


class TaylorExpansionInterPruner:
    def __init__(
        self, model, policies, runner, n_iters, approx="fisher_diag", *args, **kwargs
    ):
        # runner runs one batch of data through the model

        self.model = model
        self.policies = policies
        self.runner = runner
        self.args = args
        self.kwargs = kwargs
        self.approx = approx
        self.n_iters = n_iters

    def prune(self):

        with torch.no_grad():

            assert self.approx == "fisher_sum_of_individual_contribs"
            if self.approx == "fisher_sum_of_individual_contribs":
                grads = defaultdict(
                    lambda: torch.tensor(0).to(next(self.model.parameters()).device)
                )
                fisher_hessian = defaultdict(
                    lambda: torch.tensor(0).to(next(self.model.parameters()).device)
                )
                for i in range(self.n_iters):
                    loss = self.runner()
                    self.model.zero_grad()
                    loss.backward()
                    for name, mod in self.model.named_modules():
                        if name in self.policies.keys():
                            assert isinstance(
                                mod, (nn.Conv2d, nn.Linear)
                            ), f"Module {name} is not a Linear or Conv2d"
                            grads[name] = (
                                grads[name]
                                + (mod.weight.grad.data / self.n_iters).detach()
                            )
                            fisher_hessian[name] = (
                                fisher_hessian[name]
                                + (mod.weight.grad.data**2 / self.n_iters).detach()
                            )
                        # now we have the gradients for each layer

                # now we have the fisher information for each layer

                # perturbation = grad + 1/2 * fisher * (param ** 2)
                saliencies = {}

                for name, mod in self.model.named_modules():
                    if name in self.policies.keys():
                        saliencies[name] = (
                            +1 / 2 * fisher_hessian[name] * (mod.weight.data**2)
                        )

                # now we have the saliencies for each layer
                # we can use the saliencies to prune the model
                masks = dict()
                for name, policy in self.policies.items():
                    module = dict(self.model.named_modules())[name]
                    assert isinstance(
                        module, (nn.Conv2d, nn.Linear)
                    ), f"Module {name} is not a Linear or Conv2d"
                    reshaped = policy.grouper.transform(module.weight)
                    saliency_reshaped = policy.grouper.transform(
                        saliencies[name]
                    )  # shape [n_groups, m_elements_per_group]
                    # first prune the elements in groups, so we end up with n_groups, m_elements_per_group where the last dim has ordered by magnitude idxs
                    inter = policy.inter_group_metric["value"]

                    assert policy.inter_group_metric["name"] == "sparsity_ratio"
                    ranked_elements = torch.topk(
                        saliency_reshaped,
                        k=int(inter * reshaped.shape[0]),
                        dim=0,
                    ).indices
                    mask = torch.zeros_like(reshaped, dtype=torch.bool)
                    mask.scatter_(0, ranked_elements, 1)
                    # now we need to untransform the mask
                    mask = policy.grouper.untransform(mask, module.weight)
                    masks[name] = mask
            elif self.approx == "fisher_whole":
                # now we have the gradients for each layer
                # now we have the fisher information for each layer

                # perturbation = grad + 1/2 * fisher * (param ** 2)
                saliencies = {}

                for name, mod in self.model.named_modules():
                    if name in self.policies.keys():
                        saliencies[name] = (
                            +1 / 2 * fisher_hessian[name] * (mod.weight.data**2)
                        )

        return apply_masks(self.model, masks)


def make_vision_runner(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: str,
    backwards=False,
    verbose=False,
):

    iter_ = iter(dataloader)

    def runner():
        nonlocal iter_
        try:
            batch = next(iter_)
        except StopIteration:
            iter_ = iter(dataloader)
            batch = next(iter_)
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        model(x)
        loss = criterion(model(x), y)
        if backwards:
            model.zero_grad()
            loss.backward()
        return loss

    return runner


class ActivationMagnitudeIntraSparsityPruner:
    # https://arxiv.org/abs/2306.11695
    def __init__(self, model: nn.Module, policies: PolicyDict, runner, n_iters):
        self.model = model
        self.policies = policies
        self.runner = runner
        self.n_iters = n_iters
        self.activations = {}

        # hooks
        def get_hook(name):
            def hook(module, input, output):
                if name not in self.activations:
                    self.activations[name] = input[0].detach().mean(dim=0).unsqueeze(0)
                else:
                    self.activations[name] += input[0].detach().mean(dim=0).unsqueeze(0)

            return hook

        # hook for policied modules
        for name, module in self.model.named_modules():
            if name in self.policies.keys():
                assert isinstance(
                    module,
                    (
                        nn.Conv2d,
                        nn.Linear,
                        nn.LazyConv2d,
                        nn.LazyLinear,
                        PrunedConv2d,
                        PrunedLinear,
                    ),
                ), f"Module {name} is not a Linear or Conv2d"
                module.register_forward_hook(get_hook(name))

    def prune(self):
        for i in range(self.n_iters):
            loss = self.runner()
            # self.model.zero_grad()
            # loss.backward()
            masks = dict()
        for name, module in self.model.named_modules():
            if name in self.policies.keys():
                assert isinstance(
                    module, (nn.Conv2d, nn.Linear, PrunedConv2d, PrunedLinear)
                ), f"Module {name} is not a Linear or Conv2d, got {type(module)}"
                weight = module.weight.data
                # if conv -> shape [O, I, H_k, W_k]
                # if linear -> shape [O, I]
                acts = self.activations[name]
                # stack
                acts = acts.mean(dim=0)
                # assert acts.ndim == 3, f"Activations for {name} are not 4D, got {acts.ndim} for layer {name}"
                if isinstance(module, (nn.Conv2d, PrunedConv2d)):
                    # IM2COL
                    acts = torch.nn.functional.unfold(
                        acts,
                        module.kernel_size,
                        stride=module.stride,
                        padding=module.padding,
                    )  # shape [I * H_k * W_k, H_out * W_out]
                    w = weight  # shape [O, I, H_k, W_k]
                    if isinstance(module, PrunedConv2d):
                        assert w.shape == module.weight.shape
                        w = w * module.mask

                    # print(acts.shape, w.shape)
                    o, i, hk, wk = w.shape
                    assert acts.shape[0] == i * hk * wk
                    # print(acts.shape, w.shape, i, hk, wk, o)
                    acts = acts.reshape(
                        i, hk, wk, -1
                    )  # shape [I, H_k, W_k, H_out * W_out]
                    vnorm = lambda x: torch.sqrt(
                        torch.sum(x**2, dim=(-1), keepdim=False)
                    )
                    sal = vnorm(acts) * w  # shape [O, I, H_k, W_k]
                    assert sal.shape == module.weight.shape
                else:
                    # linear
                    # acts of shape [..., I]
                    # weight of shape [O, I]
                    w = weight.T  # shape [I, O]
                    if isinstance(module, PrunedLinear):
                        assert w.shape == module.weight.shape
                        w = w * module.mask.T
                    acts = acts.reshape(*acts.shape, 1)  # shape [..., I, 1]
                    extra_dims = acts.shape[:-2]
                    vnorm = lambda x: torch.sqrt(
                        torch.sum(x**2, dim=(extra_dims), keepdim=False)
                    )
                    sal = vnorm(acts)  # shape [I, 1]
                    sal = sal * w  # shape [I, O]
                    assert sal.T.shape == module.weight.shape
                    sal = sal.T  # shape [O, I]

                # now we have the saliencies for this layer
                # we can use the saliencies to prune the model
                policy = self.policies[name]
                reshaped = policy.grouper.transform(module.weight)
                # first prune the elements in groups, so we end up with n_groups, m_elements_per_group where the last dim has ordered by magnitude idxs
                assert policy.intra_group_metric["name"] == "sparsity_ratio"
                sparsity_ratio = policy.intra_group_metric["value"]
                ranked_elements = torch.topk(
                    reshaped.abs(),
                    k=int(sparsity_ratio * reshaped.shape[1]),
                    dim=1,
                ).indices
                mask = torch.zeros_like(reshaped, dtype=torch.bool)
                mask.scatter_(1, ranked_elements, 1)
                # now we need to untransform the mask
                mask = policy.grouper.untransform(mask, module.weight)
                masks[name] = mask
                # detach the activations
                self.activations[name] = []

        return apply_masks(self.model, masks)

    def dispose(self):
        for name, module in self.model.named_modules():
            if name in self.policies.keys():
                assert isinstance(
                    module,
                    (
                        nn.Conv2d,
                        nn.Linear,
                        nn.LazyConv2d,
                        nn.LazyLinear,
                        PrunedConv2d,
                        PrunedLinear,
                    ),
                ), f"Module {name} is not a Linear or Conv2d"
                module._forward_hooks.clear()
                module._backward_hooks.clear()
        self.activations = {}
        self.model = None


def get_sparsity_information(model: nn.Module) -> int:
    nonzero_params = 0
    total_prunable_params = 0
    total_parameters = 0

    for module in model.modules():
        if isinstance(module, (PrunedLinear, PrunedConv2d)):
            nonzero_params += module.nonzero_params()
            total_prunable_params += module.weight.numel()
            total_parameters += module.weight.numel()
            if module.bias is not None:
                total_parameters += module.bias.numel()
        else:
            for param in module.parameters(recurse=False):
                nonzero_params += param.count_nonzero()
                total_parameters += param.numel()

    pruned_parameters = total_parameters - nonzero_params
    non_pruned_parameters = total_parameters - pruned_parameters
    return {
        "nonzero_params": nonzero_params,
        "total_prunable_params": total_prunable_params,
        "sparsity_ratio_wrt_prunable": non_pruned_parameters / total_prunable_params,
        "sparsity_ratio_wrt_total": non_pruned_parameters / total_parameters,
    }


def get_sparsity_information_str(model: nn.Module) -> str:
    """
    Get the sparsity information of a model.
    """
    info = get_sparsity_information(model)
    return (
        f"Nonzero params: {info['nonzero_params']}, "
        f"Total prunable params: {info['total_prunable_params']}, "
        f"Sparsity ratio (wrt prunable): {info['sparsity_ratio_wrt_prunable']:.2%}, "
        f"Sparsity ratio (wrt total): {info['sparsity_ratio_wrt_total']:.2%}"
    )


def merge_pruned_modules(model: nn.Module) -> nn.Module:
    """
    Merge pruned modules into their parent module, replacing the pruned
    module with a normal one.
    """
    for name, module in list(model.named_modules()):
        if isinstance(module, (PrunedLinear, PrunedConv2d)):
            parts = name.split(".")
            parent = model
            for p in parts[:-1]:
                parent = getattr(parent, p)
            child_name = parts[-1]
            if isinstance(module, PrunedLinear):
                new_mod = module.to_linear()
            else:
                new_mod = module.to_conv2d()
            setattr(parent, child_name, new_mod)
    return model
