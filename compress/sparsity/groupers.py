"""
Pruning groupers for structured and unstructured pruning.

This submodule defines a set of "grouper" classes, each responsible for
transforming tensors into a 2D grouping of the form `(n_groups, group_size)`
to support pruning algorithms. A grouper also provides an `untransform`
method to restore the original tensor shape after grouping.

These are used for intra- and inter-group pruning methods.

"""

import torch


class AbstractGrouper:
    """
    Abstract base class for all groupers.
    """

    @staticmethod
    def transform(tensor: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def untransform(tensor: torch.Tensor, orig_tensor: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class UnstructuredGrouperLinear(AbstractGrouper):
    """
    Grouper that produces a single group with all weights in it. Intended for nn.Linear layers.
    """

    @staticmethod
    def transform(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.reshape(1, -1)

    @staticmethod
    def untransform(tensor: torch.Tensor, orig_tensor: torch.Tensor) -> torch.Tensor:
        return tensor.reshape(orig_tensor.shape)


class UnstructuredGrouperConv2d(AbstractGrouper):
    """
    Grouper that produces a single group with all weights in it. Intended for nn.Conv2d layers.
    """

    @staticmethod
    def transform(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.reshape(1, -1)

    @staticmethod
    def untransform(tensor: torch.Tensor, orig_tensor: torch.Tensor) -> torch.Tensor:
        return tensor.reshape(orig_tensor.shape)


class OutChannelGroupingGrouperLinear(AbstractGrouper):
    """
    Each group is composed of the weights associated with an output channel. Intended for nn.Linear layers.
    For example, for a weight tensor of shape (out_channels, in_channels), we have out_channels groups of size in_channels.
    """

    @staticmethod
    def transform(tensor: torch.Tensor) -> torch.Tensor:
        # shape is (out_channels, in_channels), so we want each out channel to be a group
        return tensor

    @staticmethod
    def untransform(tensor: torch.Tensor, orig_tensor: torch.Tensor) -> torch.Tensor:
        return tensor


class OutChannelGroupingGrouperConv2d(AbstractGrouper):
    """
    Each group is composed of the weights associated with an output channel. Intended for nn.Conv2d layers.
    """

    @staticmethod
    def transform(tensor: torch.Tensor) -> torch.Tensor:
        # shape is (out_channels, in_channels, kernel_size, kernel_size), so we want each out channel to be a group
        return tensor.reshape(tensor.shape[0], -1)

    @staticmethod
    def untransform(tensor: torch.Tensor, orig_tensor: torch.Tensor) -> torch.Tensor:
        return tensor.reshape(orig_tensor.shape)


class GroupsOf4(AbstractGrouper):
    """
    Each group is composed of 4 contiguous weights (contiguous means contiguous after flattening).
    """

    @staticmethod
    def transform(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.reshape(-1, 4)

    @staticmethod
    def untransform(tensor: torch.Tensor, orig_tensor: torch.Tensor) -> torch.Tensor:
        return tensor.reshape(orig_tensor.shape)
