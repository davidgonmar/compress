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
import math


_sparse_layers = (SparseLinear, SparseConv2d, SparseFusedConv2dBatchNorm2d)


def apply_masks(
    model: nn.Module,
    masks: Dict[str, Dict[str, torch.Tensor]],
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
                        child_mod.weight_mask.data = weight_mask * child_mod.weight_mask
                    if bias_mask is not None:
                        child_mod.bias_mask.data = bias_mask * child_mod.bias_mask
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
            ), f"Module {name} is not nn.Conv2d, nn.Linear or sparse layer"
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
                    # if the module is pruned, this will make it so that already pruned
                    # elements have importance of 0
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
                keep = reshaped.abs() >= threshold
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
    def __init__(
        self, model: nn.Module, policies: PolicyDict, metric="l2", metric_invariant=True
    ):
        self.model = model
        self.policies = policies
        self.metric_name, self.metric_invariant = metric, metric_invariant

        # the x passed to the metric is of shape [n_groups, m_elements_per_group]
        # metrics are computed over each group and generally normalized
        if metric == "l2":
            if not metric_invariant:
                self.metric = lambda x: torch.norm(x, dim=1)
            else:
                self.metric = lambda x: torch.norm(x, dim=1) / math.sqrt(x.shape[1])
        elif metric == "l1":
            if not metric_invariant:
                self.metric = lambda x: torch.sum(x.abs(), dim=1)
            else:
                self.metric = lambda x: torch.sum(x.abs(), dim=1) / x.shape[1]
        else:
            raise ValueError(f"Unknown metric {metric}")

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
                    self.metric(reshaped),  # shape [n_groups]
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
                    masks[name] = {**masks[name], "bias": mask}

            elif metric.name == "threshold":
                threshold = metric.value
                keep = self.metric(reshaped) >= threshold
                if isinstance(
                    module, (SparseConv2d, SparseLinear, SparseFusedConv2dBatchNorm2d)
                ):
                    # if the module is pruned, we need to use the previous mask also
                    old_mask = policy.grouper.transform(module.weight_mask)[
                        :, 0
                    ]  # assume it is broadcasted
                    keep = keep * old_mask
                mask = keep.to(reshaped.device)
                # now we need to untransform the mask
                # expand the mask
                weight_mask = mask.unsqueeze(1).expand(-1, reshaped.shape[1])
                weight_mask = policy.grouper.untransform(weight_mask, module.weight)
                masks[name] = {"weight": weight_mask}
                if (
                    policy.grouper is OutChannelGroupingGrouperConv2d
                    or policy.grouper is OutChannelGroupingGrouperLinear
                ) and isinstance(module, _sparse_layers):
                    # if the module is pruned, we need to use the mask
                    assert mask.shape == module.bias_mask.shape
                    masks[name] = {**masks[name], "bias": mask}
        return apply_masks(self.model, masks)


class ActivationNormInterGroupPruner:
    def __init__(
        self,
        model: nn.Module,
        policies: PolicyDict,
        runner: Runner,
        metric="l2",
        metric_invariant=True,
    ):
        self.model = model
        self.policies = policies
        self.runner = runner

        self.metric_name, self.metric_invariant = metric, metric_invariant
        # the x passed to the metric is of shape [n_groups, m_elements_per_group]
        # metrics are computed over each group and generally normalized
        if metric == "l2":
            if not metric_invariant:
                self.metric = lambda x: torch.norm(x, dim=1)
            else:
                self.metric = lambda x: torch.norm(x, dim=1) / math.sqrt(x.shape[1])
        elif metric == "l1":
            if not metric_invariant:
                self.metric = lambda x: torch.sum(x.abs(), dim=1)
            else:
                self.metric = lambda x: torch.sum(x.abs(), dim=1) / x.shape[1]
        else:
            raise ValueError(f"Unknown metric {metric}")
        
    def prune(self):
        activations = {}
        hooks = {}
        number_passes = 0
        training_state = self.model.training
        self.model.eval()

        def get_hook(name):
            nonlocal activations

            def hook(module, input, output):
                if name not in activations:
                    activations[name] = output.detach().mean(dim=0)
                else:
                    activations[name] += output.detach().mean(dim=0)
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
                loss["loss"].backward()
                number_passes += 1
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
                    module, (nn.Conv2d, nn.Linear, *_sparse_layers)
                ), f"Module {name} is not a Linear or Conv2d"
                # if conv -> weight shape [O, I, H_k, W_k]
                # if linear -> weight shape [O, I]
                acts = (
                    activations[name] / number_passes
                )  # normalize over number of batches
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
                    out_score = self.metric(reshaped_acts)  # shape [O]
                    # now we have the saliencies for this layer
                    saliencies = out_score  # shape [O]
                    if isinstance(module, (SparseConv2d, SparseFusedConv2dBatchNorm2d)):
                        saliencies = (
                            saliencies
                            * policy.grouper.transform(module.weight_mask)[:, 0]
                        )  # assumes existing mask is broadcasted from [O, 1, 1, 1] (i.e. it is an inter-channel mask)
                    if policy.inter_group_metric.name == "sparsity_ratio":
                        density_ratio = 1 - policy.inter_group_metric.value
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
                        keep = saliencies >= threshold
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
                    # acts of shape [O]
                    # weight of shape [O, I]
                    # reshape acts to [O]
                    acts = acts.reshape(acts.shape[-1], 1)
                    saliencies = self.metric(acts)  # shape [O]
                    if isinstance(module, SparseLinear):
                        saliencies = (
                            saliencies
                            * policy.grouper.transform(module.weight_mask)[:, 0]
                        )
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
                        keep = saliencies >= threshold
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
        self.model.train(training_state)
        return apply_masks(self.model, masks)


class TaylorExpansionIntraGroupPruner:
    def __init__(
        self,
        model: nn.Module,
        policies: PolicyDict,
        runner: Runner,
        approx="fisher_diag",
    ):
        self.model = model
        self.policies = policies
        self.runner = (
            runner  # assumes runner has batch size 1 for correct fisher estimation!!!
        )
        self.approx = approx
        assert self.approx == "fisher_diag", "Only fisher approximation is supported at the moment"

    def prune(self):
        training_state = self.model.training
        self.model.eval()
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
                        # For fused conv2d + bn layers, we want to use the importance not of the conv2d weights,
                        # but of the fused weights, so as to really gauge the importance of the parameters that will
                        # be used for inference. _cached_weight and _cached_bias are stored so that one can access their gradients
                        # (see ./sparse_ops.py)
                        if hasattr(mod, "_cached_weight"):
                            grad = mod._cached_weight.grad.data
                        else:
                            grad = mod.weight.grad.data
                        fisher_hessian_diag[name] = (
                            fisher_hessian_diag[name] + (grad**2).detach()
                        )
            except StopIteration:
                break # runner stopped

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
                saliency_reshaped = policy.grouper.transform(saliencies[name])
                if isinstance(
                    module, (SparseConv2d, SparseLinear, SparseFusedConv2dBatchNorm2d)
                ):
                    saliency_reshaped = saliency_reshaped * policy.grouper.transform(
                        module.weight_mask
                    ) # if weight is already pruned, we manually set its saliency to 0
                
                # first prune the elements in groups, so we end up with n_groups, m_elements_per_group where the last dim has ordered by magnitude idxs
                density = 1 - policy.intra_group_metric.value
                assert policy.intra_group_metric.name == "sparsity_ratio"
                ranked_elements = torch.topk(
                    saliency_reshaped,
                    k=int(density * saliency_reshaped.shape[1]),
                    dim=1,
                ).indices
                mask = torch.zeros_like(saliency_reshaped, dtype=torch.bool)
                mask.scatter_(1, ranked_elements, 1)
                # now we need to untransform the mask
                mask = policy.grouper.untransform(mask, module.weight)
                masks[name] = {"weight": mask}

        self.model.train(training_state)
        return apply_masks(self.model, masks)


class TaylorExpansionInterGroupPruner:
    def __init__(
        self,
        model: nn.Module,
        policies: PolicyDict,
        runner: Runner,
        approx="fisher_diag",
        use_bias=True,
    ):
        self.model = model
        self.policies = policies
        self.runner = (
            runner  # assumes runner has batch size 1 for correct fisher estimation!!!
        )
        self.approx = approx
        self.use_bias = use_bias
        assert self.approx == "fisher_diag", "Only fisher approximation is supported at the moment"

    def prune(self):
        training_state = self.model.training
        # as we are in the case of inter-group pruning, we need to estimate the fisher information for the weights and biases
        fisher_hessian = defaultdict(
            lambda: torch.tensor(0).to(next(self.model.parameters()).device)
        )
        fisher_hessian_biases = defaultdict(
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
                        # For fused conv2d + bn layers, we want to use the importance not of the conv2d weights,
                        # but of the fused weights, so as to really gauge the importance of the parameters that will
                        # be used for inference. _cached_weight and _cached_bias are stored so that one can access their gradients
                        # (see ./sparse_ops.py)
                        if hasattr(mod, "_cached_weight"):
                            grad = mod._cached_weight.grad.data
                        else:
                            grad = mod.weight.grad.data
                        fisher_hessian[name] = fisher_hessian[name] + (grad**2).detach()
                        if self.use_bias:
                            if hasattr(mod, "_cached_bias"):
                                grad = mod._cached_bias.grad.data
                            else:
                                grad = mod.bias.grad.data
                            fisher_hessian_biases[name] = (
                                fisher_hessian_biases[name] + (grad**2).detach()
                            )
            except StopIteration:
                break # runner stopped
        if self.approx == "fisher_diag":
            # now we have the fisher information for each layer
            # perturbation = 1/2 * fisher * (param ** 2) (we ignore gradients)
            saliencies = {}
            saliencies_biases = {}
            for name, mod in self.model.named_modules():
                if name in self.policies.keys():
                    saliencies[name] = self.policies[name].grouper.transform(
                        1
                        / 2
                        * fisher_hessian[name]
                        * (mod.weight.data**2)
                        / total_examples
                    )
                    if self.use_bias:
                        assert (
                            self.policies[name].grouper
                            is OutChannelGroupingGrouperConv2d
                            or self.policies[name].grouper
                            is OutChannelGroupingGrouperLinear
                        )
                        sal = (
                            1
                            / 2
                            * fisher_hessian_biases[name]
                            * (mod.bias.data**2)
                            / total_examples
                        )  # shape [O]
                        saliencies_biases[name] = sal
            masks = dict()
            # now we have the saliencies for each layer
            for name, mod in self.model.named_modules():
                if name not in self.policies.keys():
                    continue
                policy = self.policies[name]
                assert isinstance(
                    mod, (nn.Conv2d, nn.Linear, *_sparse_layers)
                ), f"Module {name} is not a Linear or Conv2d"
                saliencies_reshaped = saliencies[name]
                if isinstance(
                    mod, (SparseConv2d, SparseLinear, SparseFusedConv2dBatchNorm2d)
                ):
                    saliencies_reshaped = (
                        saliencies_reshaped * policy.grouper.transform(mod.weight_mask)
                    ) # if weight is already pruned, we manually set its saliency to 0
                    if self.use_bias:
                        saliencies_biases[name] = saliencies_biases[
                            name
                        ] * mod.bias_mask.reshape(-1) # if bias is already pruned, we manually set its saliency to 0

                # saliencies_reshaped have shape [n_groups, m_elements_per_group]
                # we need to prune individual groups (inter-group sparsity)
                per_group_saliencies = saliencies_reshaped.sum(
                    dim=1
                )  # shape [n_groups]

                # As these are hessian reductions, the natural way to estimate total importance is to sum them up.
                # Alternatively, we could use e.g. the gradient w.r.t the outputs instead of the parameters
                if self.use_bias:
                    per_group_saliencies = (
                        per_group_saliencies + saliencies_biases[name]
                    )
                # now we have the saliencies for this layer
                # we can use the saliencies to prune the model
                if policy.inter_group_metric.name == "sparsity_ratio":
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

                    weight_mask = mask.unsqueeze(1).expand(
                        -1, saliencies_reshaped.shape[1]
                    )

                    # now we need to untransform the mask
                    weight_mask = policy.grouper.untransform(weight_mask, mod.weight)

                    # if we have output channels pruning, we can also prune the bias
                    if (
                        policy.grouper is OutChannelGroupingGrouperConv2d
                        or policy.grouper is OutChannelGroupingGrouperLinear
                    ):
                        # if the module is pruned, we need to use the mask
                        assert mask.shape == mod.bias.shape
                        bias_mask = mask
                        masks[name] = {"weight": weight_mask, "bias": bias_mask}
                    else:
                        masks[name] = {"weight": weight_mask}
                elif policy.inter_group_metric.name == "threshold":
                    threshold = policy.inter_group_metric.value
                    keep = per_group_saliencies >= threshold
                    mask = keep.to(per_group_saliencies.device)
                    weight_mask = mask.unsqueeze(1).expand(
                        -1, saliencies_reshaped.shape[1]
                    )
                    weight_mask = policy.grouper.untransform(weight_mask, mod.weight)
                    masks[name] = {"weight": weight_mask}
                    if (
                        policy.grouper is OutChannelGroupingGrouperConv2d
                        or policy.grouper is OutChannelGroupingGrouperLinear
                    ):
                        assert mask.shape == mod.bias.shape
                        bias_mask = mask
                        masks[name] = {"weight": weight_mask, "bias": bias_mask}
                    else:
                        masks[name] = {"weight": weight_mask}

        self.model.train(training_state)
        return apply_masks(self.model, masks)


class ActivationMagnitudeIntraGroupPruner:
    # Inspired by Wanda (https://arxiv.org/abs/2306.11695)
    # The logic is that each element of a parameter is weighted by the magnitude of the activations it "influences"
    def __init__(self, model: nn.Module, policies: PolicyDict, runner: Runner):
        self.model = model
        self.policies = policies
        self.runner = runner

    def prune(self):
        training_state = self.model.training
        activations = {}
        total_acts = {}
        hooks = {}
        # hooks
        def get_hook(name):
            nonlocal activations, total_acts

            def hook(module, input, output):
                nonlocal activations, total_acts
                if name not in activations:
                    activations[name] = input[0].detach().sum(dim=0)
                else:
                    activations[name] += input[0].detach().sum(dim=0)
                total_acts[name] = (total_acts.get(name, 0) + input[0].shape[0])
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

                # if conv -> shape [O, I, H_k, W_k]
                # if linear -> shape [O, I]
                weight = module.weight.data

                acts = activations[name]
                acts = acts / total_acts[name]
                if isinstance(
                    module, (nn.Conv2d, SparseConv2d, SparseFusedConv2dBatchNorm2d)
                ):
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

                    o, i, hk, wk = w.shape
                    assert acts.shape[0] == i * hk * wk
                    acts = acts.reshape(
                        i, hk, wk, -1
                    )  # shape [I, H_k, W_k, H_out * W_out]
                    # Given a kernel element K[o, i, h, w], it interacts with the row of size H_out * W_out of acts
                    # Unlike Wanda, we normalize
                    vnorm = (lambda x: torch.sqrt(
                        torch.sum(x**2, dim=(-1), keepdim=False)
                    ) / math.sqrt(x.shape[-1]))(acts) # shape = [I, H_k, W_k]
                    sal = vnorm * w.abs()  # shape [O, I, H_k, W_k] (gets braodcasted in the output channel)
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
                    vnorm = (lambda x: torch.sqrt(
                        torch.sum(x**2, dim=(extra_dims), keepdim=False)
                    ) / math.sqrt(math.prod(extra_dims)))(acts)
                    sal = vnorm  # shape [I, 1]
                    sal = sal * w.abs()  # shape [I, O]
                    sal = sal.T  # shape [O, I]
                    assert sal.shape == module.weight.shape

                # Now use saliencies to prune the layer
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
                    sal,
                    k=int(density * sal.shape[1]),
                    dim=1,
                ).indices
                mask = torch.zeros_like(sal, dtype=torch.bool)
                mask.scatter_(1, ranked_elements, 1)
                # now we need to untransform the mask
                mask = policy.grouper.untransform(mask, module.weight)
                masks[name] = {"weight": mask}

        self.model.train(training_state)
        return apply_masks(self.model, masks)


def get_sparsity_information(model: nn.Module) -> dict:
    nz_total = 0
    total = 0

    stack = [model]
    while stack:
        module = stack.pop()

        if isinstance(module, _sparse_layers):
            nz_total += module.nonzero_params()
            total += module.total_params()
            continue

        for p in module.parameters(recurse=False):
            total += p.numel()

        stack.extend(module.children())
    sparsity_ratio = 1.0 - nz_total / total if total else 0.0
    return {
        "nonzero_params": nz_total,
        "total_params": total,
        "sparsity_ratio": sparsity_ratio,
    }


def get_sparsity_information_str(dic: dict) -> str:

    return (
        f"Nonzero params: {dic['nonzero_params']}, "
        f"Total params: {dic['total_params']}, "
        f"Sparsity ratio: {dic['sparsity_ratio']:.2%}"
    )


def fuse_bn_conv_sparse_train(conv, bn, conv_name, bn_name):
    return SparseFusedConv2dBatchNorm2d.from_conv_bn(
        conv,
        bn,
    )
