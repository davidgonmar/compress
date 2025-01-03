import torch
import torch.nn as nn


class LowRankLinear(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, rank: int, bias: bool = True
    ):
        super(LowRankLinear, self).__init__()
        self.w0 = nn.Parameter(torch.randn(in_features, rank))
        self.w1 = nn.Parameter(torch.randn(rank, out_features))
        self.bias = nn.Parameter(torch.randn(out_features)) if bias else None

    @staticmethod
    def from_linear(linear: nn.Linear, ratio_to_keep: float = 0.1):
        # Original linear -> O = X @ W
        # Low rank linear -> W = U @ S @ V_T -> O = X @ U @ S @ V_T
        W, b = linear.weight.T, linear.bias
        U, S, V_T = torch.linalg.svd(W, full_matrices=True)  # complete SVD
        orig_rank = min(S.shape)
        rank = max(int(orig_rank * ratio_to_keep), 1)
        S = torch.diag(S[:rank])  # in R^{MIN(IN, OUT) x MIN(IN, OUT)}
        # pad S to be {IN x OUT}
        in_f, out_f = W.shape
        assert S.shape == (rank, rank)
        assert U.shape == (in_f, in_f)
        assert V_T.shape == (out_f, out_f)
        W0 = U[:, :rank] @ S  # in R^{IN x RANK}
        W1 = V_T[:rank, :]  # in R^{RANK x OUT}
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
        return low_rank_linear
    
    def forward(self, x: torch.Tensor):
        # X in R^{... x IN}
        # W0 in R^{IN x RANK} -> X @ W0 in R^{... x RANK}
        # W1 in R^{RANK x OUT} -> O = (X @ W0) @ W1 in R^{... x OUT}
        if self.bias is not None:
            return torch.matmul(torch.matmul(x, self.w0), self.w1) + self.bias
        else:
            return torch.matmul(torch.matmul(x, self.w0), self.w1)

    def __repr__(self):
        return f"LowRankLinear(in_features={self.w0.shape[0]}, out_features={self.w1.shape[1]}, rank={self.w0.shape[1]}, bias={self.bias is not None})"
