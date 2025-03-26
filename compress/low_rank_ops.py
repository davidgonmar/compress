import torch
import torch.nn as nn


def _get_rank_ratio_to_keep(S: torch.Tensor, ratio_to_keep: float):
    return max(int(S.shape[0] * ratio_to_keep), 1)


def _get_rank_energy_to_keep(S: torch.Tensor, energy_to_keep: float):
    # chooses rank such that
    # sum(S[:rank] ** 2) <= energy_to_keep * sum(S ** 2)
    assert 0.0 <= energy_to_keep <= 1.0, "energy_to_keep must be in [0, 1]"
    total_energy = torch.sum(S**2)
    energy = 0.0
    rank = 0
    for s in S:
        energy += s**2
        rank += 1
        if energy / total_energy >= energy_to_keep:
            break
    return rank


def _get_rank(
    S: torch.Tensor,
    ratio_to_keep: float | None = None,
    energy_to_keep: float | None = None,
):
    assert S.ndim == 1, "Singular values must be 1-dimensional"
    if ratio_to_keep is not None:
        return _get_rank_ratio_to_keep(S, ratio_to_keep)
    elif energy_to_keep is not None:
        return _get_rank_energy_to_keep(S, energy_to_keep)
    else:
        raise ValueError("Either ratio_to_keep or energy_to_keep must be provided")


class LowRankLinear(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, rank: int, bias: bool = True
    ):
        super(LowRankLinear, self).__init__()
        self.w0 = nn.Parameter(torch.randn(in_features, rank))
        self.w1 = nn.Parameter(torch.randn(rank, out_features))
        self.bias = nn.Parameter(torch.randn(out_features)) if bias else None

    @staticmethod
    def from_linear(
        linear: nn.Linear,
        ratio_to_keep: float | None = None,
        energy_to_keep: float | None = None,
        keep_singular_values_separated: bool = False,
    ):
        # Original linear -> O = X @ W.T + b
        # Low rank linear -> W = U @ S @ V_T -> O = X @ (U @ S @ V_T).T + b = (X @ W1.T) @ W0.T + b
        W, b = linear.weight, linear.bias
        U, S, V_T = torch.linalg.svd(W, full_matrices=True)  # complete SVD
        rank = _get_rank(S, ratio_to_keep=ratio_to_keep, energy_to_keep=energy_to_keep)
        S = torch.diag(S[:rank])  # in R^{MIN(IN, OUT) x MIN(IN, OUT)}
        out_f, in_f = W.shape
        assert S.shape == (rank, rank)
        assert U.shape == (out_f, out_f)
        assert V_T.shape == (in_f, in_f)
        W0 = (
            U[:, :rank] @ S if not keep_singular_values_separated else U[:, :rank]
        )  # in R^{OUT x RANK}
        W1 = V_T[:rank, :]  # in R^{RANK x IN}
        low_rank_linear = LowRankLinear(
            linear.weight.shape[1],
            linear.weight.shape[0],
            rank,
            bias=linear.bias is not None,
        )
        low_rank_linear.w0.data = W0
        low_rank_linear.w1.data = W1
        if b is not None and linear.bias is not None:
            low_rank_linear.bias.data = b
        else:
            low_rank_linear.bias = None
        if keep_singular_values_separated:
            low_rank_linear.S = nn.Parameter(S)
        low_rank_linear.keep_singular_values_separated = keep_singular_values_separated
        return low_rank_linear

    def from_linear_activation(
        linear: nn.Linear,
        act_cov_mat_chol: torch.Tensor,  # shape (in_features, in_features). Result of cholesky factorization of the input activations covariance matrix (X @ X^T) = L @ L^T
        ratio_to_keep: float | None = None,
        energy_to_keep: float | None = None,
    ):
        # adapted from https://arxiv.org/abs/2403.07378
        W, b = linear.weight, linear.bias
        U, S, V_T = torch.linalg.svd(W @ act_cov_mat_chol, full_matrices=True)
        rank = _get_rank(S, ratio_to_keep=ratio_to_keep, energy_to_keep=energy_to_keep)
        S = torch.diag(S[:rank])
        out_f, in_f = W.shape
        assert S.shape == (rank, rank)
        assert U.shape == (out_f, out_f)
        assert V_T.shape == (in_f, in_f)
        act_cov_mat_cholinv = torch.linalg.inv(act_cov_mat_chol)
        W0 = U[:, :rank] @ S
        W1 = V_T[:rank, :] @ act_cov_mat_cholinv
        low_rank_linear = LowRankLinear(
            linear.weight.shape[1],
            linear.weight.shape[0],
            rank,
            bias=linear.bias is not None,
        )

        low_rank_linear.w0.data = W0
        low_rank_linear.w1.data = W1
        if b is not None and linear.bias is not None:
            low_rank_linear.bias.data = b
        else:
            low_rank_linear.bias = None
        low_rank_linear.keep_singular_values_separated = False
        return low_rank_linear

    def forward(self, x: torch.Tensor):
        # X in R^{BATCH x IN}, W0 in R^{OUT x RANK}, W1 in R^{RANK x IN}
        w0, w1 = self.w0, self.w1
        if self.keep_singular_values_separated:
            w0 = w0 @ self.S
        return torch.nn.functional.linear(x @ w1.t(), w0, bias=self.bias)

    def __repr__(self):
        return f"LowRankLinear(in_features={self.w0.shape[0]}, out_features={self.w1.shape[1]}, rank={self.w0.shape[1]}, bias={self.bias is not None})"

    def to_linear(self):
        res = nn.Linear(self.w0.shape[0], self.w1.shape[1], bias=self.bias is not None)
        res.weight = nn.Parameter(self.w0 @ self.w1)
        if self.bias is not None:
            res.bias = self.bias
        return res


class LowRankConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        rank: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super(LowRankConv2d, self).__init__()
        self.w0 = nn.Parameter(torch.randn(rank, in_channels, kernel_size, kernel_size))
        self.w1 = nn.Parameter(torch.randn(out_channels, rank, 1, 1))
        self.bias = nn.Parameter(torch.randn(out_channels)) if bias else None
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.rank = rank
        self.input_channels = in_channels
        self.kernel_size = kernel_size
        self.out_channels = out_channels

    @staticmethod
    def from_conv2d(
        conv2d: nn.Conv2d,
        ratio_to_keep: float | None = None,
        energy_to_keep: float | None = None,
        keep_singular_values_separated: bool = False,
    ):
        # Original conv2d -> O = conv2d(W, X)
        # Low rank conv2d -> O = conv2d(W1, conv2d(W0, X))
        W, b = conv2d.weight, conv2d.bias
        o, i, h, w = W.shape
        U, S, V_T = torch.linalg.svd(
            W.permute(1, 2, 3, 0).reshape(i * h * w, o), full_matrices=True
        )
        rank = _get_rank(S, ratio_to_keep=ratio_to_keep, energy_to_keep=energy_to_keep)
        W0 = (
            (
                (U[:, :rank] @ torch.diag(S[:rank]))
                .reshape(i, h, w, rank)
                .permute(3, 0, 1, 2)
            )
            if not keep_singular_values_separated
            else (U[:, :rank].reshape(i, h, w, rank).permute(3, 0, 1, 2))
        )  # shape = (rank, i, h, w)
        W1 = (
            V_T[:rank, :].reshape(rank, o, 1, 1).permute(1, 0, 2, 3)
        )  # shape = (o, rank, 1, 1)
        low_rank_conv2d = LowRankConv2d(
            conv2d.in_channels,
            conv2d.out_channels,
            conv2d.kernel_size[0],
            rank,
            stride=conv2d.stride[0],
            padding=conv2d.padding[0],
            dilation=conv2d.dilation[0],
            groups=conv2d.groups,
            bias=conv2d.bias is not None,
        )
        low_rank_conv2d.w0.data = W0
        low_rank_conv2d.w1.data = W1

        if b is not None and conv2d.bias is not None:
            low_rank_conv2d.bias.data = b
        else:
            low_rank_conv2d.bias = None

        if keep_singular_values_separated:
            low_rank_conv2d.S = nn.Parameter(S[:rank])

        low_rank_conv2d.keep_singular_values_separated = keep_singular_values_separated

        return low_rank_conv2d

    def from_conv2d_activation(
        conv2d: nn.Conv2d,
        act_cov_mat_chol: torch.Tensor,  # shape (i * h * w)^2. Result of cholesky factorization of the input activations covariance matrix (X @ X^T) = L @ L^T
        ratio_to_keep: float | None = None,
        energy_to_keep: float | None = None,
    ):
        # adapted from https://arxiv.org/abs/2403.07378
        W, b = conv2d.weight, conv2d.bias
        o, i, h, w = W.shape
        U, S, V_T = torch.linalg.svd(
            act_cov_mat_chol @ W.permute(1, 2, 3, 0).reshape(i * h * w, o),
            full_matrices=True,
        )
        rank = _get_rank(S, ratio_to_keep=ratio_to_keep, energy_to_keep=energy_to_keep)
        act_cov_mat_cholinv = torch.linalg.inv(act_cov_mat_chol)
        W0 = (
            (
                (U[:, :rank] @ torch.diag(S[:rank]))
                .reshape(i, h, w, rank)
                .permute(3, 0, 1, 2)
            ).reshape(rank, -1)
            @ act_cov_mat_cholinv.T
        ).reshape(
            rank, i, h, w
        )  # shape = (rank, i, h, w)
        W1 = (
            V_T[:rank, :].reshape(rank, o, 1, 1).permute(1, 0, 2, 3)
        )  # shape = (o, rank, 1, 1)
        low_rank_conv2d = LowRankConv2d(
            conv2d.in_channels,
            conv2d.out_channels,
            conv2d.kernel_size[0],
            rank,
            stride=conv2d.stride[0],
            padding=conv2d.padding[0],
            dilation=conv2d.dilation[0],
            groups=conv2d.groups,
            bias=conv2d.bias is not None,
        )
        low_rank_conv2d.w0.data = W0
        low_rank_conv2d.w1.data = W1

        if b is not None and conv2d.bias is not None:
            low_rank_conv2d.bias.data = b
        else:
            low_rank_conv2d.bias = None

        low_rank_conv2d.keep_singular_values_separated = False

        return low_rank_conv2d

    def get_weights_as_matrices(self, w, keyword):
        # inverse permutation of (3, 0, 1, 2) is  (1, 2, 3, 0), of (1, 0, 2, 3) is (1, 0, 2, 3)
        assert keyword in {"w0", "w1"}
        w = (
            w.permute(1, 2, 3, 0).reshape(
                self.input_channels * self.kernel_size * self.kernel_size, self.rank
            )
            if keyword == "w0"
            else w.permute(1, 0, 2, 3).reshape(self.rank, self.out_channels)
        )
        return w

    def forward(self, x: torch.Tensor):
        w0, w1 = self.w0, self.w1
        if self.keep_singular_values_separated:
            # print(w0.shape, self.S.shape)
            w0 = w0 * self.S.reshape(-1, 1, 1, 1)
        conv_out = torch.nn.functional.conv2d(
            x,
            w0,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )  # shape (batch, rank, h_out, w_out)
        h_out, w_out = conv_out.shape[2], conv_out.shape[3]
        linear_out = torch.nn.functional.linear(
            conv_out.permute(0, 2, 3, 1).reshape(-1, self.rank),
            w1.reshape(self.out_channels, self.rank),
            bias=self.bias,
        )  # shape (batch * h_out * w_out, out_channels)
        return linear_out.reshape(-1, h_out, w_out, self.out_channels).permute(
            0, 3, 1, 2
        )

    def __repr__(self):
        return f"LowRankConv2d(in_channels={self.input_channels}, out_channels={self.out_channels}, kernel_size={self.kernel_size}, rank={self.rank}, stride={self.stride}, padding={self.padding}, dilation={self.dilation}, groups={self.groups}, bias={self.bias is not None})"

    def to_conv2d(self):
        res = nn.Conv2d(
            self.input_channels,
            self.out_channels,
            self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
            bias=self.bias is not None,
        )
        w0, w1 = self.get_weights_as_matrices(
            self.w0, "w0"
        ), self.get_weights_as_matrices(self.w1, "w1")
        w = torch.reshape(
            (w0 @ w1).T,
            (
                self.out_channels,
                self.input_channels,
                self.kernel_size,
                self.kernel_size,
            ),
        )
        res.weight = nn.Parameter(w)
        if self.bias is not None:
            res.bias = self.bias
        return res
