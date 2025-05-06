from torch import nn
from compress.sparsity.pruned_ops import PrunedLinear, PrunedConv2d
from tqdm import tqdm
from compress.sparsity.pruning_strats import (
    UnstructuredGrouperLinear,
    UnstructuredGrouperConv2d,
    PruningGrouper,
)
import torch
from typing import Dict
import torchvision
from collections import defaultdict

_module_to_pruned = {
    nn.Linear: PrunedLinear.from_linear,
    nn.LazyLinear: PrunedLinear.from_linear,
    nn.Conv2d: PrunedConv2d.from_conv2d,
}


def vision_model_runner(dataloader: torch.utils.data.DataLoader, model: nn.Module):
    for batch in tqdm(dataloader, desc="Running model"):
        x, y = batch
        x = x.to(model.device)
        y = y.to(model.device)
        model(x)
    return model


class PruningPolicy:
    def __init__(
        self,
        grouper: PruningGrouper,
        inter_group_sparsity: float,
        intra_group_sparsity: float,
    ):
        self.grouper = grouper
        self.inter_group_sparsity = inter_group_sparsity
        self.intra_group_sparsity = intra_group_sparsity


PolicyDict = Dict[str, PruningPolicy]  # per layer


def unstructured_resnet18_policies(sparsity: float = 0.5) -> PolicyDict:
    # example policy for resnet18
    conv_keys = {}
    linear_keys = {}
    with torch.device("meta"):
        model = torchvision.models.resnet18(weights=None)
        keys = model.named_modules()
        for name, module in keys:
            if isinstance(module, nn.Conv2d):
                conv_keys[name] = PruningPolicy(
                    grouper=UnstructuredGrouperConv2d(),
                    inter_group_sparsity=1.0,
                    intra_group_sparsity=sparsity,
                )
            elif isinstance(module, nn.Linear):
                linear_keys[name] = module
                linear_keys[name] = PruningPolicy(
                    grouper=UnstructuredGrouperLinear(),
                    inter_group_sparsity=1.0,
                    intra_group_sparsity=sparsity,
                )

    return {
        **conv_keys,
        **linear_keys,
    }


def apply_masks(
    model: nn.Module,
    masks: Dict[str, torch.Tensor],
) -> nn.Module:
    """
    Walks the model hierarchy and replaces any nn.Conv2d or nn.Linear
    whose full module name is in masks with a PrunedConv2d or PrunedLinear.
    """

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

            # recurse into (possibly replaced) child
            _recursive_apply(child_mod, full_name)

    _recursive_apply(model)
    return model


class MagnitudePruner:
    # this does not need to track activations
    def __init__(
        self, model: nn.Module, policies: PolicyDict, global_prune: bool = False
    ):
        self.model = model
        self.policies = policies
        assert global_prune is False, "Global pruning not supported yet"

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
            # first prune the elements in groups, so we end up with n_groups, m_elements_per_group where the last dim has ordered by magnitude idxs
            ranked_elements = torch.topk(
                reshaped.abs(),
                k=int(policy.intra_group_sparsity * reshaped.shape[1]),
                dim=1,
            ).indices  # n_groups, m_elements_per_group
            mask = torch.zeros_like(reshaped, dtype=torch.bool)
            mask.scatter_(1, ranked_elements, 1)
            # now we need to untransform the mask
            mask = policy.grouper.untransform(mask, module.weight)
            masks[name] = mask

        return apply_masks(self.model, masks)


class TaylorExpansionPruner:
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

    def prune(self):
        grads = defaultdict(
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
                    grads[name] = grads[name] + mod.weight.grad.data / self.n_iters

        with torch.no_grad():
            # now we have the gradients for each layer
            assert self.approx == "fisher"
            hess = {}
            for name, mod in self.model.named_modules():
                if name in self.policies.keys():
                    hess[name] = -mod.weight.grad.data**2

            # now we have the fisher information for each layer

            # perturbation = grad + 1/2 * fisher * (param ** 2)
            saliencies = {}

            for name, mod in self.model.named_modules():
                if name in self.policies.keys():
                    saliencies[name] = (
                        -mod.weight.grad.data * mod.weight.data
                        + 1 / 2 * hess[name] * (mod.weight.data**2)
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
                ranked_elements = torch.topk(
                    saliency_reshaped.abs(),
                    k=int(policy.intra_group_sparsity * reshaped.shape[1]),
                    dim=1,
                ).indices
                mask = torch.zeros_like(reshaped, dtype=torch.bool)
                mask.scatter_(1, ranked_elements, 1)
                # now we need to untransform the mask
                mask = policy.grouper.untransform(mask, module.weight)
                masks[name] = mask

        return apply_masks(self.model, masks)


class WandaPruner:
    # this one needs to store activations
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
                    module, (nn.Conv2d, nn.Linear)
                ), f"Module {name} is not a Linear or Conv2d"
                module.register_forward_hook(get_hook(name))

    def prune(self):
        for i in range(self.n_iters):
            loss = self.runner()
            self.model.zero_grad()
            loss.backward()
            masks = dict()
        for name, module in self.model.named_modules():
            if name in self.policies.keys():
                assert isinstance(
                    module, (nn.Conv2d, nn.Linear)
                ), f"Module {name} is not a Linear or Conv2d"
                weight = module.weight.data
                # if conv -> shape [O, I, H_k, W_k]
                # if linear -> shape [O, I]
                acts = self.activations[name]
                # stack
                acts = acts.mean(dim=0)
                # assert acts.ndim == 3, f"Activations for {name} are not 4D, got {acts.ndim} for layer {name}"
                if isinstance(module, nn.Conv2d):
                    # IM2COL
                    acts = torch.nn.functional.unfold(
                        acts,
                        module.kernel_size,
                        stride=module.stride,
                        padding=module.padding,
                    )  # shape [I * H_k * W_k, H_out * W_out]
                    w = weight  # shape [O, I, H_k, W_k]
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
                ranked_elements = torch.topk(
                    reshaped.abs(),
                    k=int(policy.intra_group_sparsity * reshaped.shape[1]),
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


def measure_nonzero_params(model: nn.Module) -> int:
    """
    Count the number of non-zero (i.e. *un*-pruned) parameters in the model,
    delegating to `module.nonzero_params()` for any pruned modules.
    """
    nonzero_params = 0

    for module in model.modules():
        if isinstance(module, (PrunedLinear, PrunedConv2d)):
            nonzero_params += module.nonzero_params()
        else:
            for param in module.parameters(recurse=False):
                nonzero_params += param.numel()

    return nonzero_params
