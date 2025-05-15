import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair
from torch.nn import functional as F


def _get_rank_ratio_to_keep(S: torch.Tensor, rank_ratio_to_keep: float):
    assert 0.0 <= rank_ratio_to_keep <= 1.0, "rank_ratio_to_keep must be in [0, 1]"
    return max(int(S.shape[0] * rank_ratio_to_keep), 1)


def _get_svals_energy_ratio_to_keep(
    S: torch.Tensor, svals_energy_ratio_to_keep: float
) -> int:
    assert 0.0 <= svals_energy_ratio_to_keep <= 1.0
    sq = S.pow(2)
    cum_energy = sq.cumsum(dim=0)
    total_energy = cum_energy[-1]
    threshold = svals_energy_ratio_to_keep * total_energy
    idx = torch.searchsorted(cum_energy, threshold)
    return idx.item() + 1


def _get_params_number_ratio_to_keep(
    X: torch.Tensor,
    S: torch.Tensor,
    params_ratio_to_keep: float,
):
    assert X.ndim == 2, "X must be 2-dimensional"
    assert S.ndim == 1, "Singular values must be 1-dimensional"

    m, n = X.shape

    # A in R^{m x r}
    # B in R^{r x n}

    # So keeping a rank involves a total of m + n parameters

    params_per_rank_kept = torch.arange(0, S.shape[0] + 1).float() * (m + n)

    rel_params_per_rank_kept = params_per_rank_kept / params_per_rank_kept[-1]
    rank_to_keep = torch.searchsorted(
        rel_params_per_rank_kept, params_ratio_to_keep
    )  # rank_to_keep is the number of ranks to keep

    return rank_to_keep.item() + 1


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
        keep_metric: dict[str, float],
        keep_singular_values_separated: bool = False,
    ):
        # Original linear -> O = X @ W.T + b
        # Low rank linear -> W = U @ S @ V_T -> O = X @ (U @ S @ V_T).T + b = (X @ W1.T) @ W0.T + b
        W, b = linear.weight, linear.bias
        U, S, V_T = torch.linalg.svd(W, full_matrices=True)  # complete SVD

        rank = 0
        if keep_metric["name"] == "rank_ratio_to_keep":
            rank = _get_rank_ratio_to_keep(S, keep_metric["value"])
        elif keep_metric["name"] == "svals_energy_ratio_to_keep":
            rank = _get_svals_energy_ratio_to_keep(S, keep_metric["value"])
        elif keep_metric["name"] == "params_ratio_to_keep":
            rank = _get_params_number_ratio_to_keep(W, S, keep_metric["value"])
        else:
            raise ValueError(
                "keep_metric must be one of ['rank_ratio_to_keep', 'svals_energy_ratio_to_keep', 'params_ratio_to_keep']"
            )
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
        keep_metric: dict[str, float],
    ):
        # adapted from https://arxiv.org/abs/2403.07378
        W, b = linear.weight, linear.bias
        U, S, V_T = torch.linalg.svd(W @ act_cov_mat_chol, full_matrices=True)
        rank = 0
        if keep_metric["name"] == "rank_ratio_to_keep":
            rank = _get_rank_ratio_to_keep(S, keep_metric["value"])
        elif keep_metric["name"] == "svals_energy_ratio_to_keep":
            rank = _get_svals_energy_ratio_to_keep(S, keep_metric["value"])
        elif keep_metric["name"] == "params_ratio_to_keep":
            raise NotImplementedError(
                "params_ratio_to_keep not implemented for linear activation"
            )
        else:
            raise ValueError(
                "keep_metric must be one of ['rank_ratio_to_keep', 'svals_energy_ratio_to_keep', 'params_ratio_to_keep']"
            )
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
        keep_metric: dict[str, float],
        keep_singular_values_separated: bool = False,
    ):
        # Original conv2d -> O = conv2d(W, X)
        # Low rank conv2d -> O = conv2d(W1, conv2d(W0, X))
        W, b = conv2d.weight, conv2d.bias
        o, i, h, w = W.shape
        U, S, V_T = torch.linalg.svd(
            W.permute(1, 2, 3, 0).reshape(i * h * w, o), full_matrices=True
        )
        rank = 0
        if keep_metric["name"] == "rank_ratio_to_keep":
            rank = _get_rank_ratio_to_keep(S, keep_metric["value"])
        elif keep_metric["name"] == "svals_energy_ratio_to_keep":
            rank = _get_svals_energy_ratio_to_keep(S, keep_metric["value"])
        elif keep_metric["name"] == "params_ratio_to_keep":
            rank = _get_params_number_ratio_to_keep(
                W.permute(1, 2, 3, 0).reshape(i * h * w, o), S, keep_metric["value"]
            )
        else:
            raise ValueError(
                "keep_metric must be one of ['rank_ratio_to_keep', 'svals_energy_ratio_to_keep', 'params_ratio_to_keep']"
            )
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
        keep_metric: dict[str, float],
    ):
        # adapted from https://arxiv.org/abs/2403.07378
        W, b = conv2d.weight, conv2d.bias
        o, i, h, w = W.shape
        U, S, V_T = torch.linalg.svd(
            act_cov_mat_chol @ W.permute(1, 2, 3, 0).reshape(i * h * w, o),
            full_matrices=True,
        )
        rank = 0
        if keep_metric["name"] == "rank_ratio_to_keep":
            rank = _get_rank_ratio_to_keep(S, keep_metric["value"])
        elif keep_metric["name"] == "svals_energy_ratio_to_keep":
            rank = _get_svals_energy_ratio_to_keep(S, keep_metric["value"])
        elif keep_metric["name"] == "params_ratio_to_keep":
            raise ValueError(
                "keep_metric must be one of ['rank_ratio_to_keep', 'svals_energy_ratio_to_keep']"
            )
        S = torch.diag(S[:rank])  # in R^{MIN(IN, OUT) x MIN(IN, OUT)}

        act_cov_mat_cholinv = torch.linalg.inv(act_cov_mat_chol)
        W0 = (
            ((U[:, :rank] @ S).reshape(i, h, w, rank).permute(3, 0, 1, 2)).reshape(
                rank, -1
            )
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


class SpatialLowRankConv2d(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        rank: int,
        stride: int | tuple[int, int] = 1,
        padding: str | int | tuple[int, int] = "same",
        dilation: int | tuple[int, int] = 1,
        bias: bool = True,
    ):
        super().__init__()

        k_h, k_w = _pair(kernel_size)
        self.rank = rank
        self.stride_h, self.stride_w = _pair(stride)
        self.dil_h, self.dil_w = _pair(dilation)

        if padding == "same":
            self.pad_h = ((k_h - 1) * self.dil_h) // 2
            self.pad_w = ((k_w - 1) * self.dil_w) // 2
        else:
            self.pad_h, self.pad_w = _pair(padding)

        self.h_weight = nn.Parameter(
            torch.empty(out_channels * in_channels * rank, 1, k_h, 1)
        )  # shape = (out_channels * in_channels * rank, 1, k_h, 1)
        self.v_weight = nn.Parameter(
            torch.empty(out_channels * in_channels * rank, 1, 1, k_w)
        )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (k_h, k_w)
        self.stride = (self.stride_h, self.stride_w)
        self.padding = (self.pad_h, self.pad_w)
        self.dilation = (self.dil_h, self.dil_w)
        self.groups = 1

        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.v_weight)
        nn.init.uniform_(self.h_weight)

        if self.bias is not None:
            nn.init.zeros_(self.bias)

    @staticmethod
    def from_conv2d(
        conv: nn.Conv2d,
        keep_metric: dict[str, float],
    ) -> "SpatialLowRankConv2d":
        assert conv.groups == 1, "Grouped / depth‑wise conv not supported yet"
        W, b = conv.weight.detach(), conv.bias.detach()
        C_out, C_in, k_h, k_w = W.shape
        U, S, Vt = torch.linalg.svd(W)  # decomposes spatially each filter, so
        ranks = []
        for i in range(C_out):
            for j in range(C_in):
                rank = 0
                if keep_metric["name"] == "rank_ratio_to_keep":
                    rank = _get_rank_ratio_to_keep(S[i, j], keep_metric["value"])
                elif keep_metric["name"] == "svals_energy_ratio_to_keep":
                    rank = _get_svals_energy_ratio_to_keep(
                        S[i, j], keep_metric["value"]
                    )
                elif keep_metric["name"] == "params_ratio_to_keep":
                    raise NotImplementedError(
                        "params_ratio_to_keep not implemented for conv2d"
                    )
                ranks.append(rank)
        rank = max(ranks)  # maybe i can explore other modalities in the future
        # perm
        U_r = (
            U[:, :, :, :rank]
            .permute(1, 0, 3, 2)
            .reshape(C_in * C_out * rank, 1, k_h, 1)
        )  # shape = (C_out * rank, C_in, k_h, 1)
        S_r = S[:, :, :rank]  # shape = (C_out, C_in, rank)
        Vt_r = Vt[:, :, :rank, :]  # shape = (C_out, C_in, rank, k_w)
        Vt_r = (
            (S_r.reshape(*S_r.shape, 1) * Vt_r)
            .permute(1, 0, 2, 3)
            .reshape(C_in * C_out * rank, 1, 1, k_w)
        )
        low_rank = SpatialLowRankConv2d(
            C_in,
            C_out,
            (k_h, k_w),
            rank,
            stride=conv.stride,
            padding=conv.padding,
            dilation=conv.dilation,
            bias=b is not None,
        )
        low_rank.h_weight.data.copy_(U_r)
        low_rank.v_weight.data.copy_(Vt_r)
        if b is not None:
            low_rank.bias.data.copy_(b)
        return low_rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x.shape = (batch, C_in, H, W)
        cin = x.shape[1]
        x = F.conv2d(
            x,
            self.v_weight,
            bias=None,
            stride=(1, self.stride_w),
            padding=(0, self.pad_w),
            dilation=(1, self.dil_w),
            groups=cin,
        )
        # x.shape = (batch, C_in * C_out * rank, Hf, 1)
        x = F.conv2d(
            x,
            self.h_weight,
            bias=None,
            stride=(self.stride_h, 1),
            padding=(self.pad_h, 0),
            dilation=(self.dil_h, 1),
            groups=x.shape[1],
        )
        # x.shape = (batch, C_in * C_out * rank, Hf, Wf)
        x = x.reshape(
            x.shape[0],
            self.in_channels,
            self.out_channels,
            self.rank,
            x.shape[2],
            x.shape[3],
        ).sum(dim=(1, 3))
        if self.bias is not None:
            x = x + self.bias.view(1, self.out_channels, 1, 1)
        return x

    def to_conv2d(self) -> nn.Conv2d:
        raise NotImplementedError("to_conv2d not implemented for SpatialLowRankConv2d")
