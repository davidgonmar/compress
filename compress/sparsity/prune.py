from torch import nn
from compress.sparsity.sparse_ops import (
    SparseLinear,
    SparseConv2d,
    SparseFusedConv2dBatchNorm2d,
)
from compress.sparsity.groupers import (
    OutChannelGroupingGrouperConv2d,
    OutChannelGroupingGrouperLinear,
)
import torch
from typing import Dict
from collections import defaultdict
from compress.sparsity.policy import PolicyDict
from compress.sparsity.runner import Runner


_sparse_layers = (SparseLinear, SparseConv2d, SparseFusedConv2dBatchNorm2d)


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
                masks_ = masks[full_name]
                weight_mask = masks_.get("weight", None)
                bias_mask = masks_.get("bias", None)
                if isinstance(child_mod, nn.Conv2d):
                    pruned = SparseConv2d.from_conv2d(
                        child_mod, weight_mask=weight_mask, bias_mask=bias_mask
                    )
                    setattr(parent, child_name, pruned)
                    child_mod = pruned
                elif isinstance(child_mod, nn.Linear):
                    pruned = SparseLinear.from_linear(
                        child_mod, weight_mask=weight_mask, bias_mask=bias_mask
                    )
                    setattr(parent, child_name, pruned)
                    child_mod = pruned
                elif isinstance(child_mod, _sparse_layers):
                    if not allow_reprune:
                        raise ValueError(
                            f"Module {full_name} is already pruned, and allow_reprune is False"
                        )
                    # if the module is already pruned, just update the mask
                    if weight_mask is not None:
                        child_mod.weight_mask = weight_mask * child_mod.weight_mask
                    if bias_mask is not None:
                        child_mod.bias_mask = bias_mask * child_mod.bias_mask
                else:
                    raise ValueError(
                        f"Module {full_name} is not a Linear or Conv2d, got {type(child_mod)}"
                    )
            # recurse into (possibly replaced) child
            _recursive_apply(child_mod, full_name)

    _recursive_apply(model)
    return model


class AbstractPruner:
    def __init__(self):
        pass

    def prune(self):
        raise NotImplementedError


class WeightMagnitudeIntraGroupPruner(AbstractPruner):
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
                module, (nn.Conv2d, nn.Linear, *_sparse_layers)
            ), f"Module {name} is not nn.Conv2d, nn.Linear or SparseConv2d, SparseLinear"
            reshaped = policy.grouper.transform(
                module.weight
            )  # shape [n_groups, m_elements_per_group]
            # prune the elements in each group, so we end up with n_groups
            assert policy.intra_group_metric.name in ["sparsity_ratio", "threshold"]

            if policy.intra_group_metric.name == "sparsity_ratio":
                sparsity_ratio = policy.intra_group_metric.value
                if isinstance(
                    module, (SparseConv2d, SparseLinear, SparseFusedConv2dBatchNorm2d)
                ):
                    # if the module is pruned, we need to use the mask
                    reshaped = reshaped * policy.grouper.transform(module.weight_mask)
                ranked_elements = torch.topk(
                    reshaped.abs(),
                    k=int((1 - sparsity_ratio) * reshaped.shape[1]),
                    dim=1,
                ).indices  # n_groups, int(sparsity_ratio * m_elements_per_group)
                mask = torch.zeros_like(reshaped, dtype=torch.bool)
                mask.scatter_(1, ranked_elements, 1)
                # now we need to untransform the mask
                mask = policy.grouper.untransform(mask, module.weight)
                masks[name] = {"weight": mask}
            elif policy.intra_group_metric.name == "threshold":
                threshold = policy.intra_group_metric.value
                keep = reshaped.abs() > threshold
                mask = keep.to(reshaped.device)
                # now we need to untransform the mask
                mask = policy.grouper.untransform(mask, module.weight)
                masks[name] = mask
                # if has existing mask, we need to combine the masks
                if isinstance(
                    module, (SparseConv2d, SparseLinear, SparseFusedConv2dBatchNorm2d)
                ):
                    # if the module is pruned, we need to use the mask
                    assert mask.shape == module.weight_mask.shape
                    masks[name] = mask * module.weight_mask

                masks[name] = {"weight": masks[name]}
        return apply_masks(self.model, masks)


class WeightNormInterGroupPruner(AbstractPruner):
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
                module, (nn.Conv2d, nn.Linear, *_sparse_layers)
            ), f"Module {name} is not a Linear or Conv2d"
            reshaped = policy.grouper.transform(
                module.weight
            )  # n_groups, m_elements_per_group
            # prune the elements in each group, so we end up with n_groups
            metric = policy.inter_group_metric
            assert metric.name in ["sparsity_ratio", "threshold"]
            if metric.name == "sparsity_ratio":
                density = 1 - metric.value
                if isinstance(module, _sparse_layers):
                    reshaped = reshaped * policy.grouper.transform(
                        module.weight_mask
                    )  # if mask is 0, the norm is 0 so it will not be selected
                ranked_elements = torch.topk(
                    reshaped.norm(dim=1),  # shape [n_groups]
                    k=int(density * reshaped.shape[0]),
                    dim=0,
                ).indices
                mask = torch.zeros(
                    (reshaped.shape[0],), dtype=torch.bool, device=reshaped.device
                )
                mask.scatter_(0, ranked_elements, 1)
                # now we need to untransform the mask and expand it to the original shape
                weight_mask = mask.unsqueeze(1).expand(-1, reshaped.shape[1])
                weight_mask = policy.grouper.untransform(weight_mask, module.weight)
                masks[name] = {"weight": weight_mask}
                # if we have output channels pruning, we can also prune the bias
                if (
                    policy.grouper is OutChannelGroupingGrouperConv2d
                    or policy.grouper is OutChannelGroupingGrouperLinear
                ):
                    # if the module is pruned, we need to use the mask
                    assert mask.shape == module.bias_mask.shape
                    masks[name] = {**masks[name], "bias": mask}

            elif metric.name == "threshold":
                threshold = metric.value
                keep = reshaped.norm(dim=1) > threshold
                mask = keep.to(reshaped.device)
                # now we need to untransform the mask
                # expand the mask
                weight_mask = mask.unsqueeze(1).expand(-1, reshaped.shape[1])
                weight_mask = policy.grouper.untransform(weight_mask, module.weight)
                masks[name] = weight_mask
                if isinstance(
                    module, (SparseConv2d, SparseLinear, SparseFusedConv2dBatchNorm2d)
                ):
                    # if the module is pruned, we need to use the previous mask also
                    assert weight_mask.shape == module.weight_mask.shape
                    masks[name] = weight_mask * module.weight_mask
                masks[name] = {"weight": masks[name]}
                if (
                    policy.grouper is OutChannelGroupingGrouperConv2d
                    or policy.grouper is OutChannelGroupingGrouperLinear
                ):
                    # if the module is pruned, we need to use the mask
                    assert mask.shape == module.bias_mask.shape
                    masks[name] = {**masks[name], "bias": mask}
        return apply_masks(self.model, masks)


class ActivationNormInterGroupPruner:
    def __init__(self, model: nn.Module, policies: PolicyDict, runner: Runner):
        self.model = model
        self.policies = policies
        self.runner = runner

    def prune(self):
        activations = {}
        hooks = {}

        def get_hook(name):
            nonlocal activations

            def hook(module, input, output):
                if name not in activations:
                    activations[name] = output.detach().mean(dim=0).unsqueeze(0)
                else:
                    activations[name] += output.detach().mean(dim=0).unsqueeze(0)

            return hook

        # hook for policied modules
        for name, module in self.model.named_modules():
            if name in self.policies.keys():
                assert isinstance(
                    module, (nn.Conv2d, nn.Linear, *_sparse_layers)
                ), f"Module {name} is not a Linear or Conv2d"
                handle = module.register_forward_hook(get_hook(name))
                hooks[name] = handle

        while True:
            try:
                loss = self.runner.iteration()
                self.model.zero_grad()
                loss.backward()
            except StopIteration:
                break

        # remove hooks
        for name, module in self.model.named_modules():
            if name in hooks.keys():
                hook = hooks[name]
                hook.remove()

        masks = dict()

        for name, module in self.model.named_modules():
            if name in self.policies.keys():
                assert isinstance(
                    module, (nn.Conv2d, nn.Linear)
                ), f"Module {name} is not a Linear or Conv2d"
                # if conv -> weight shape [O, I, H_k, W_k]
                # if linear -> weight shape [O, I]
                acts = activations[name].mean(0)
                policy = self.policies[name]
                if isinstance(
                    module, (nn.Conv2d, SparseFusedConv2dBatchNorm2d, SparseConv2d)
                ):
                    assert (
                        policy.grouper is OutChannelGroupingGrouperConv2d
                    ), "Policy grouper is not a OutChannelGroupingGrouperConv2d"
                    reshaped_acts = acts.reshape(
                        acts.shape[0], -1
                    )  # shape [O, H_out * W_out]
                    out_score = torch.norm(reshaped_acts, dim=1)  # shape [O]
                    # now we have the saliencies for this layer
                    saliencies = out_score
                    if policy.inter_group_metric.name == "sparsity_ratio":
                        density_ratio = 1 - policy.inter_group_metric.value
                        # we do not need to multiply by existing masks, as activations will be 0 if the corresponding mask is 0
                        # shape [O]
                        selector = torch.topk(
                            saliencies,
                            k=int(density_ratio * saliencies.shape[0]),
                            dim=0,
                        ).indices
                        mask = torch.zeros_like(saliencies, dtype=torch.bool)
                        mask.scatter_(0, selector, 1)
                        # broadcast mask back to original shape
                        o, i, hk, wk = module.weight.shape
                        weight_mask = mask.reshape(o, 1, 1, 1).expand(-1, i, hk, wk)

                        # bias mask is simply the same
                        bias_mask = mask.reshape(o)

                        mask = {
                            "weight": weight_mask,
                            "bias": bias_mask,
                        }

                    elif policy.inter_group_metric.name == "threshold":
                        threshold = policy.inter_group_metric.value
                        keep = saliencies > threshold
                        mask = keep.to(saliencies.device)
                        # broadcast mask back to original shape
                        o, i, hk, wk = module.weight.shape
                        weight_mask = mask.reshape(o, 1, 1, 1).expand(-1, i, hk, wk)
                        # bias mask is simply the same
                        bias_mask = mask.reshape(o)
                        mask = {
                            "weight": weight_mask,
                            "bias": bias_mask,
                        }

                else:  # if isinstance(module, nn.Linear):
                    assert (
                        policy.grouper is OutChannelGroupingGrouperLinear
                    ), "Policy grouper is not a OutChannelGroupingGrouperLinear"
                    # linear
                    # acts of shape [..., I]
                    # weight of shape [O, I]
                    # reshape acts to [combine(...), O]
                    acts = acts.reshape(-1, acts.shape[-1])
                    norm = torch.norm(acts, dim=0, keepdim=False)  # shape [0]
                    saliencies = norm
                    if policy.inter_group_metric.name == "sparsity_ratio":
                        density = 1 - policy.inter_group_metric.value
                        # shape [O]
                        selector = torch.topk(
                            saliencies,
                            k=int(density * saliencies.shape[0]),
                            dim=0,
                        ).indices
                        mask = torch.zeros_like(saliencies, dtype=torch.bool)
                        mask.scatter_(0, selector, 1)
                        # broadcast mask back to original shape
                        o = module.weight.shape[0]
                        weight_mask = mask.reshape(o, 1).expand(module.weight.shape)
                        # bias mask is simply the same
                        bias_mask = mask.reshape(o)
                        mask = {
                            "weight": weight_mask,
                            "bias": bias_mask,
                        }
                    elif policy.inter_group_metric.name == "threshold":
                        threshold = policy.inter_group_metric.value
                        keep = saliencies > threshold
                        mask = keep.to(saliencies.device)
                        # broadcast mask back to original shape
                        o = module.weight.shape[0]
                        weight_mask = mask.reshape(o, 1).expand(module.weight.shape)
                        # bias mask is simply the same
                        bias_mask = mask.reshape(o)
                        mask = {
                            "weight": weight_mask,
                            "bias": bias_mask,
                        }
                masks[name] = mask

        return apply_masks(self.model, masks)


class TaylorExpansionIntraGroupPruner:
    def __init__(
        self,
        model: nn.Module,
        policies: PolicyDict,
        runner: Runner,
        approx="fisher_diag",
    ):
        # runner runs one batch of data through the model
        self.model = model
        self.policies = policies
        self.runner = runner
        self.approx = approx
        assert self.approx == "fisher_diag", "Only fisher approximation is supported"

    def prune(self):

        fisher_hessian_diag = defaultdict(
            lambda: torch.tensor(0).to(next(self.model.parameters()).device)
        )
        total_examples = 0
        while True:
            try:
                res = self.runner.iteration()
                self.model.zero_grad()
                res["loss"].backward()
                n_examples = res["data"][0].shape[0]
                total_examples += n_examples
                for name, mod in self.model.named_modules():
                    if name in self.policies.keys():
                        assert isinstance(
                            mod, (nn.Conv2d, nn.Linear, *_sparse_layers)
                        ), f"Module {name} is not a Linear or Conv2d"
                        if hasattr(mod, "_cached_weight"):
                            grad = mod._cached_weight.grad.data
                        else:
                            grad = mod.weight.grad.data
                        fisher_hessian_diag[name] = (
                            fisher_hessian_diag[name]
                            + (grad**2).detach()
                        )

            except StopIteration:
                break

        with torch.no_grad():
            # now we have the fisher information for each layer
            # perturbation = 1/2 * fisher * (param ** 2) (we ignore gradients)
            saliencies = {}

            for name, mod in self.model.named_modules():
                if name in self.policies.keys():
                    saliencies[name] = (
                        1
                        / 2
                        * fisher_hessian_diag[name]
                        / total_examples
                        * (mod.weight.data**2)
                    )

            # now we have the saliencies for each layer
            # we can use the saliencies to prune the model
            masks = dict()
            for name, policy in self.policies.items():
                module = dict(self.model.named_modules())[name]
                assert isinstance(
                    module, (nn.Conv2d, nn.Linear, *_sparse_layers)
                ), f"Module {name} is not a Linear or Conv2d"
                reshaped = policy.grouper.transform(module.weight)
                saliency_reshaped = policy.grouper.transform(saliencies[name])
                if isinstance(
                    module, (SparseConv2d, SparseLinear, SparseFusedConv2dBatchNorm2d)
                ):
                    reshaped = reshaped * policy.grouper.transform(module.weight_mask)
                    saliency_reshaped = saliency_reshaped * policy.grouper.transform(
                        module.weight_mask
                    )
                # first prune the elements in groups, so we end up with n_groups, m_elements_per_group where the last dim has ordered by magnitude idxs
                density = 1 - policy.intra_group_metric.value
                assert policy.intra_group_metric.name == "sparsity_ratio"
                ranked_elements = torch.topk(
                    saliency_reshaped,
                    k=int(density * reshaped.shape[1]),
                    dim=1,
                ).indices
                mask = torch.zeros_like(reshaped, dtype=torch.bool)
                mask.scatter_(1, ranked_elements, 1)
                # now we need to untransform the mask
                mask = policy.grouper.untransform(mask, module.weight)
                masks[name] = mask

        return apply_masks(self.model, masks)


class TaylorExpansionInterGroupPruner:
    def __init__(
        self,
        model: nn.Module,
        policies: PolicyDict,
        runner: Runner,
        approx="fisher_diag",
    ):
        # runner runs one batch of data through the model

        self.model = model
        self.policies = policies
        self.runner = runner
        self.approx = approx

    def prune(self):
        fisher_hessian = defaultdict(
            lambda: torch.tensor(0).to(next(self.model.parameters()).device)
        )
        total_examples = 0
        while True:
            try:
                loss = self.runner.iteration()
                self.model.zero_grad()
                loss["loss"].backward()
                n_examples = loss["data"][0].shape[0]
                total_examples += n_examples
                for name, mod in self.model.named_modules():
                    if name in self.policies.keys():
                        assert isinstance(
                            mod, (nn.Conv2d, nn.Linear, *_sparse_layers)
                        ), f"Module {name} is not a Linear or Conv2d"
                        if hasattr(mod, "_cached_weight"):
                            grad = mod._cached_weight.grad.data
                        else:
                            grad = mod.weight.grad.data
                        fisher_hessian[name] = (
                            fisher_hessian[name] + (grad**2).detach()
                        )

            except StopIteration:
                break
        if self.approx == "fisher_diag":
            # now we have the gradients for each layer
            # now we have the fisher information for each layer
            # perturbation = grad + 1/2 * fisher * (param ** 2)
            saliencies = {}

            for name, mod in self.model.named_modules():
                if name in self.policies.keys():
                    saliencies[name] = (
                        +1
                        / 2
                        * fisher_hessian[name]
                        * (mod.weight.data**2)
                        / total_examples
                    )
            masks = dict()
            # now we have the saliencies for each layer
            for name, mod in self.model.named_modules():
                policy = self.policies[name]
                assert policy.inter_group_metric.name == "sparsity_ratio"
                assert isinstance(
                    mod, (nn.Conv2d, nn.Linear, *_sparse_layers)
                ), f"Module {name} is not a Linear or Conv2d"
                saliencies_reshaped = policy.grouper.transform(saliencies[name])
                if isinstance(mod, (SparseConv2d, SparseLinear)):
                    saliencies_reshaped = saliencies_reshaped * mod.weight_mask

                # saliencies_reshaped have shape [n_groups, m_elements_per_group]
                # we need to prune individual groups (inter-group sparsity)
                per_group_saliencies = saliencies_reshaped.sum(
                    dim=1
                )  # shape [n_groups]
                # now we have the saliencies for this layer
                # we can use the saliencies to prune the model

                topk = int(
                    (1 - policy.inter_group_metric.value)
                    * per_group_saliencies.shape[0]
                )
                ranked_elements = torch.topk(
                    per_group_saliencies,
                    k=topk,
                    dim=0,
                ).indices  # shape [topk]
                mask = torch.zeros_like(per_group_saliencies, dtype=torch.bool)
                mask.scatter_(0, ranked_elements, 1)

                weight_mask = mask.unsqueeze(1).expand(-1, saliencies_reshaped.shape[1])

                # now we need to untransform the mask
                weight_mask = policy.grouper.untransform(weight_mask, mod.weight)

                # if we have output channels pruning, we can also prune the bias

                if (
                    policy.grouper is OutChannelGroupingGrouperConv2d
                    or policy.grouper is OutChannelGroupingGrouperLinear
                ):
                    # if the module is pruned, we need to use the mask
                    assert mask.shape == mod.bias_mask.shape
                    bias_mask = mask.reshape(mod.bias_mask.shape)
                    masks[name] = {"weight": weight_mask, "bias": bias_mask}

                else:
                    masks[name] = {"weight": weight_mask}
        return apply_masks(self.model, masks)


class ActivationMagnitudeIntraGroupPruner:
    # adapted from https://arxiv.org/abs/2306.11695
    def __init__(self, model: nn.Module, policies: PolicyDict, runner: Runner):
        self.model = model
        self.policies = policies
        self.runner = runner

    def prune(self):
        activations = {}
        total_acts = 0
        hooks = {}

        # hooks
        def get_hook(name):
            nonlocal total_acts, activations

            def hook(module, input, output):
                nonlocal total_acts, activations

                if name not in activations:
                    activations[name] = input[0].detach().sum(dim=0)
                else:
                    activations[name] += input[0].detach().sum(dim=0)
                total_acts += input[0].shape[0]

            return hook

        # hook for policied modules
        for name, module in self.model.named_modules():
            if name in self.policies.keys():
                assert isinstance(
                    module,
                    (
                        nn.Conv2d,
                        nn.Linear,
                        *_sparse_layers,
                    ),
                ), f"Module {name} is not a Linear or Conv2d"
                hook = module.register_forward_hook(get_hook(name))
                hooks[name] = hook

        while True:
            try:
                self.runner.iteration()
            except StopIteration:
                break

        # remove hooks
        for name, module in self.model.named_modules():
            if name in hooks.keys():
                handle = hooks[name]
                handle.remove()

        masks = dict()
        for name, module in self.model.named_modules():
            if name in self.policies.keys():
                assert isinstance(
                    module,
                    (
                        nn.Conv2d,
                        nn.Linear,
                        SparseConv2d,
                        SparseLinear,
                        SparseFusedConv2dBatchNorm2d,
                    ),
                ), f"Module {name} is not a Linear or Conv2d, got {type(module)}"
                weight = module.weight.data
                # if conv -> shape [O, I, H_k, W_k]
                # if linear -> shape [O, I]
                acts = activations[name]
                acts = acts / total_acts
                # assert acts.ndim == 3, f"Activations for {name} are not 4D, got {acts.ndim} for layer {name}"
                if isinstance(
                    module, (nn.Conv2d, SparseConv2d, SparseFusedConv2dBatchNorm2d)
                ):
                    # IM2COL
                    acts = torch.nn.functional.unfold(
                        acts,
                        module.kernel_size,
                        stride=module.stride,
                        padding=module.padding,
                    )  # shape [I * H_k * W_k, H_out * W_out]
                    w = weight  # shape [O, I, H_k, W_k]
                    if isinstance(module, (SparseConv2d, SparseFusedConv2dBatchNorm2d)):
                        assert w.shape == module.weight.shape
                        w = w * module.weight_mask

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
                    if isinstance(module, SparseLinear):
                        assert w.shape == module.weight.T.shape
                        w = w * module.weight_mask.T
                    acts = acts.reshape(*acts.shape, 1)  # shape [..., I, 1]
                    extra_dims = acts.shape[:-2]
                    vnorm = lambda x: torch.sqrt(
                        torch.sum(x**2, dim=(extra_dims), keepdim=False)
                    )
                    sal = vnorm(acts)  # shape [I, 1]
                    sal = sal * w  # shape [I, O]
                    sal = sal.T  # shape [O, I]
                    assert sal.shape == module.weight.shape

                # now we have the saliencies for this layer
                # we can use the saliencies to prune the model
                policy = self.policies[name]
                sal = policy.grouper.transform(sal)
                # if existing mask, we need to use it
                if isinstance(
                    module, (SparseConv2d, SparseLinear, SparseFusedConv2dBatchNorm2d)
                ):
                    sal = sal * policy.grouper.transform(module.weight_mask)

                # first prune the elements in groups, so we end up with n_groups, m_elements_per_group where the last dim has ordered by magnitude idxs
                assert policy.intra_group_metric.name == "sparsity_ratio"
                density = 1 - policy.intra_group_metric.value
                ranked_elements = torch.topk(
                    sal.abs(),
                    k=int(density * sal.shape[1]),
                    dim=1,
                ).indices
                mask = torch.zeros_like(sal, dtype=torch.bool)
                mask.scatter_(1, ranked_elements, 1)
                # now we need to untransform the mask
                mask = policy.grouper.untransform(mask, module.weight)
                masks[name] = {"weight": mask}
        return apply_masks(self.model, masks)


def get_sparsity_information(model: nn.Module) -> dict:
    """
    Recursively walk the module tree, but **stop** descending once a module
    belongs to `_sparse_layers`.  Everything else is explored depth-first.

    Returns a dict with sparsity information for the whole model and for the
    subset of weights inside `_sparse_layers`.
    """
    # running totals
    nz_total = total = 0
    nz_prunable = total_prunable = 0

    stack = [model]
    while stack:
        module = stack.pop()

        if isinstance(module, _sparse_layers):
            # treat this block as atomic
            nz = module.nonzero_params()
            tot = module.total_params()

            nz_prunable += nz
            total_prunable += tot
            nz_total += nz
            total += tot
            continue

        # count parameters that belong *directly* to this module
        for p in module.parameters(recurse=False):
            numel = p.numel()
            total += numel
            nz_total += torch.count_nonzero(p).item()

        # explore children
        stack.extend(module.children())

    sparsity_total = 1.0 - nz_total / total if total else 0.0
    sparsity_prunable = 1.0 - nz_prunable / total_prunable if total_prunable else 0.0

    return {
        "nonzero_params": nz_total,
        "total_params": total,
        "total_prunable_params": total_prunable,
        "sparsity_ratio_wrt_prunable": sparsity_prunable,
        "sparsity_ratio_wrt_total": sparsity_total,
    }


def get_sparsity_information_str(dic: dict) -> str:
    """
    Returns a string with the sparsity information.
    """
    return (
        f"Nonzero params: {dic['nonzero_params']}, "
        f"Total prunable params: {dic['total_prunable_params']}, "
        f"Sparsity ratio w.r.t. prunable params: {dic['sparsity_ratio_wrt_prunable']:.2%}, "
        f"Sparsity ratio w.r.t. total params: {dic['sparsity_ratio_wrt_total']:.2%}"
    )


def merge_pruned_modules(model: nn.Module) -> nn.Module:
    """
    Merge pruned modules into their parent module, replacing the pruned
    module with a normal one.
    """
    for name, module in list(model.named_modules()):
        if isinstance(
            module, (SparseLinear, SparseConv2d, SparseFusedConv2dBatchNorm2d)
        ):
            parts = name.split(".")
            parent = model
            for p in parts[:-1]:
                parent = getattr(parent, p)
            child_name = parts[-1]
            if isinstance(module, SparseLinear):
                new_mod = module.to_linear()
            else:  # SparseFusedConv2dBatchNorm2d gets converted to Conv2d
                new_mod = module.to_conv2d()
            setattr(parent, child_name, new_mod)
    return model


def fuse_bn_conv_sparse_train(conv, bn, conv_name, bn_name):
    return SparseFusedConv2dBatchNorm2d.from_conv_bn(
        conv,
        bn,
    )
