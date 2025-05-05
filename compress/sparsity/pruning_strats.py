from functools import partial


# pruning grouper spec accepts a ndim-tensor and returns a reshaped tensor of the form (n, m), n denoting the number of groups and m denoting the number of elements in each group
class PruningGrouper:
    ndim: int

    def transform(self, tensor):
        raise NotImplementedError

    def untransform(self, tensor, orig_tensor):
        raise NotImplementedError


class UnstructuredGrouperLinear(PruningGrouper):
    ndim = 2

    def transform(self, tensor):
        return tensor.reshape(1, -1)

    def untransform(self, tensor, orig_tensor):
        return tensor.reshape(orig_tensor.shape)


class UnstructuredGrouperConv2d(PruningGrouper):
    ndim = 4

    def transform(self, tensor):
        return tensor.reshape(1, -1)

    def untransform(self, tensor, orig_tensor):
        return tensor.reshape(orig_tensor.shape)


class OutChannelGroupingGrouperLinear(PruningGrouper):
    ndim = 2

    def transform(self, tensor):
        # shape is (out_channels, in_channels), so we want each out channel to be a group
        return tensor.permute(1, 0)

    def untransform(self, tensor, orig_tensor):
        return tensor.permute(1, 0)


class OutChannelGroupingGrouperConv2d(PruningGrouper):
    ndim = 4

    def transform(self, tensor):
        # shape is (out_channels, in_channels, kernel_size, kernel_size), so we want each out channel to be a group
        return tensor.reshape(tensor.shape[0], -1).permute(
            1, 0
        )  # shape (in_channels * kernel_size * kernel_size, out_channels)

    def untransform(self, tensor, orig_tensor):
        return tensor.permute(1, 0).reshape(orig_tensor.shape)


class GroupsOfN(PruningGrouper):
    ndim = None

    def __init__(self, n):
        self.n = n

    def transform(self, tensor):
        return tensor.reshape(-1, self.n)

    def untransform(self, tensor, orig_tensor):
        return tensor.reshape(orig_tensor.shape)


def linear_grouper_from_str(st: str) -> PruningGrouper:
    if st == "unstructured":
        return UnstructuredGrouperLinear
    elif st == "out_channel_grouping":
        return OutChannelGroupingGrouperLinear
    elif st == "2:4":
        return partial(GroupsOfN, 4)
    else:
        raise ValueError(f"Grouper {st} not recognized")


def conv2d_grouper_from_str(st: str):
    if st == "unstructured":
        return UnstructuredGrouperConv2d
    elif st == "out_channel_grouping":
        return OutChannelGroupingGrouperConv2d
    elif st == "2:4":
        return partial(GroupsOfN, 4)
    else:
        raise ValueError(f"Grouper {st} not recognized")
