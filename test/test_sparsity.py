import torch
import torch.nn as nn
from compress.sparsity.pruned_ops import PrunedLinear, PrunedConv2d
from compress.sparsity.pruning_strats import (
    UnstructuredGrouperLinear,
    UnstructuredGrouperConv2d,
)

from compress.sparsity.prune import (
    apply_masks,
    MagnitudePruner,
    unstructured_resnet18_policies,
    ActivationPruner,
    PruningPolicy,
)


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 2, 1)
        self.linear = nn.Linear(4, 2)

    def forward(self, x):
        return self.linear(x)


def test_apply_masks_conv():
    model = DummyModel()
    mask = torch.zeros_like(model.conv.weight, dtype=torch.bool)
    flat = mask.view(-1)
    flat[: mask.numel() // 2] = True
    mask = flat.view_as(mask)

    apply_masks(model, {"conv": mask})
    assert isinstance(model.conv, PrunedConv2d)
    assert torch.equal(model.conv.mask, mask)


def test_apply_masks_linear():
    model = DummyModel()
    # All-True mask
    mask = torch.ones_like(model.linear.weight, dtype=torch.bool)

    apply_masks(model, {"linear": mask})
    assert isinstance(model.linear, PrunedLinear)
    assert torch.equal(model.linear.mask, mask)


def test_magnitude_pruner_reduces_weights():
    model = DummyModel()
    orig_count = model.linear.weight.numel()

    policy = PruningPolicy(
        grouper=UnstructuredGrouperLinear(),
        inter_group_sparsity=1.0,
        intra_group_sparsity=0.5,
    )
    pruner = MagnitudePruner(model, {"linear": policy})
    pruner.prune()

    # After pruning, module should be PrunedLinear
    assert isinstance(model.linear, PrunedLinear)
    kept = model.linear.mask.sum().item()
    assert kept <= orig_count * 0.5 + 1


def test_unstructured_resnet18_policies():
    sparsity = 0.3
    policies = unstructured_resnet18_policies(sparsity)
    # Must include Conv2d and Linear policies
    assert any(
        isinstance(p.grouper, UnstructuredGrouperConv2d) for p in policies.values()
    )
    assert any(
        isinstance(p.grouper, UnstructuredGrouperLinear) for p in policies.values()
    )
    for p in policies.values():
        assert p.intra_group_sparsity == sparsity
        assert p.inter_group_sparsity == 1.0


def test_activation_pruner_accumulates_and_detaches():
    model = DummyModel()
    runner = lambda m: m(torch.randn(1, 4))
    pruner = ActivationPruner(model, ["linear"], runner, pruning_ratio=0.5)
    pruner.run()

    assert "linear" in pruner.accumulated_state
    activations = pruner.accumulated_state["linear"]
    assert isinstance(activations, list)
    assert activations[0].shape == (1, 2)
    assert pruner.has_run is True
    pruner.detach()
    assert pruner.hooks == []
    assert pruner.accumulated_state == {}
    assert pruner.model is None
    assert pruner.runner is None
