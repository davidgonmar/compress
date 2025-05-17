import torch


# PruningGrouper is a base class for all pruning groupers.
# A PronningGrouper is responsible for transforming a tensor into a 2D tensor of the form (n_groups, group_size)
class PruningGrouper:
    @staticmethod
    def transform(tensor: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @staticmethod
    def untransform(tensor: torch.Tensor, orig_tensor: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class UnstructuredGrouperLinear(PruningGrouper):
    @staticmethod
    def transform(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.reshape(1, -1)

    @staticmethod
    def untransform(tensor: torch.Tensor, orig_tensor: torch.Tensor) -> torch.Tensor:
        return tensor.reshape(orig_tensor.shape)


class UnstructuredGrouperConv2d(PruningGrouper):

    @staticmethod
    def transform(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.reshape(1, -1)

    @staticmethod
    def untransform(tensor: torch.Tensor, orig_tensor: torch.Tensor) -> torch.Tensor:
        return tensor.reshape(orig_tensor.shape)


class OutChannelGroupingGrouperLinear(PruningGrouper):

    @staticmethod
    def transform(tensor: torch.Tensor) -> torch.Tensor:
        # shape is (out_channels, in_channels), so we want each out channel to be a group
        return tensor

    @staticmethod
    def untransform(tensor: torch.Tensor, orig_tensor: torch.Tensor) -> torch.Tensor:
        return tensor


class OutChannelGroupingGrouperConv2d(PruningGrouper):
    @staticmethod
    def transform(tensor: torch.Tensor) -> torch.Tensor:
        # shape is (out_channels, in_channels, kernel_size, kernel_size), so we want each out channel to be a group
        return tensor.reshape(tensor.shape[0], -1)

    @staticmethod
    def untransform(tensor: torch.Tensor, orig_tensor: torch.Tensor) -> torch.Tensor:
        return tensor.reshape(orig_tensor.shape)


class GroupsOf4(PruningGrouper):
    @staticmethod
    def transform(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.reshape(-1, 4)

    @staticmethod
    def untransform(tensor: torch.Tensor, orig_tensor: torch.Tensor) -> torch.Tensor:
        return tensor.reshape(orig_tensor.shape)
