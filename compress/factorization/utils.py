import torch


def matrix_approx_rank(matrix: torch.Tensor, energy_threshold: float = 0.99) -> int:
    """
    Calculate the approximate rank of a matrix based on a given energy threshold.

    Args:
        energy_threshold (float): The energy threshold to determine the approximate rank.
        matrix (torch.Tensor): The input matrix.

    Returns:
        int: The approximate rank of the matrix.
    """
    # conv filter
    if matrix.ndim == 4:
        matrix = matrix.view(matrix.shape[0], -1)
    singular_values = torch.linalg.svdvals(matrix)
    cumulative_energy = torch.cumsum(singular_values**2, dim=0)
    energy_ratio = cumulative_energy / cumulative_energy[-1]
    approximate_rank = torch.sum(energy_ratio < energy_threshold).item()
    return approximate_rank
