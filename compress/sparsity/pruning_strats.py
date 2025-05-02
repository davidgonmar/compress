from functools import partial


# pruning granularity spec accepts a ndim-tensor and returns a reshaped tensor of the form (n, m), n denoting the number of groups and m denoting the number of elements in each group
class PruningGranularity:
    ndim: int

    def transform(self, tensor):
        raise NotImplementedError

    def untransform(self, tensor):
        raise NotImplementedError


class UnstructuredGranularityLinear(PruningGranularity):
    ndim = 2

    def transform(self, tensor):
        self.orig_shape = tensor.shape
        return tensor.reshape(-1, 1)

    def untransform(self, tensor):
        return tensor.reshape(self.orig_shape)


class UnstructuredGranularityConv2d(PruningGranularity):
    ndim = 4

    def transform(self, tensor):
        self.orig_shape = tensor.shape
        return tensor.reshape(-1, 1)

    def untransform(self, tensor):
        return tensor.reshape(self.orig_shape)


class OutChannelGroupingGranularityLinear(PruningGranularity):
    ndim = 2

    def transform(self, tensor):
        # shape is (out_channels, in_channels), so we want each out channel to be a group
        return tensor.permute(1, 0)

    def untransform(self, tensor):
        return tensor.permute(1, 0)


class OutChannelGroupingGranularityConv2d(PruningGranularity):
    ndim = 4

    def transform(self, tensor):
        self.orig_shape = tensor.shape
        # shape is (out_channels, in_channels, kernel_size, kernel_size), so we want each out channel to be a group
        return tensor.reshape(tensor.shape[0], -1).permute(
            1, 0
        )  # shape (in_channels * kernel_size * kernel_size, out_channels)

    def untransform(self, tensor):
        return tensor.permute(1, 0).reshape(self.orig_shape)


class GroupsOfN(PruningGranularity):
    ndim = None

    def __init__(self, n):
        self.n = n

    def transform(self, tensor):
        self.orig_shape = tensor.shape
        return tensor.reshape(-1, self.n)

    def untransform(self, tensor):
        return tensor.reshape(self.orig_shape)


def linear_granularity_from_str(st: str) -> PruningGranularity:
    if st == "unstructured":
        return UnstructuredGranularityLinear
    elif st == "out_channel_grouping":
        return OutChannelGroupingGranularityLinear
    elif st == "2:4":
        return partial(GroupsOfN, 4)
    else:
        raise ValueError(f"Granularity {st} not recognized")


def conv2d_granularity_from_str(st: str):
    if st == "unstructured":
        return UnstructuredGranularityConv2d
    elif st == "out_channel_grouping":
        return OutChannelGroupingGranularityConv2d
    elif st == "2:4":
        return partial(GroupsOfN, 4)
    else:
        raise ValueError(f"Granularity {st} not recognized")
