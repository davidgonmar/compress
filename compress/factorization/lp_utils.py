import torch


def maximize_energy(
    cum_energy_vectors, cumulative_cost_vectors, total_cost, minimize=False
):
    import pulp

    # We are given N vectors of cumulative energies and of cumulative vectors. We want to, by selecting a (cumulative)
    # subset of indices from each vector (cumulative in the sense that if j is chosen, all(j' < j) are also chosen), maximize
    # the sum of energies at the selected indices such that the sum of the cumulative costs at the selected indices is less than or equal to the total cost.

    # Let x_{i, j} be a binary variable indicating whether the j-th index in the i-th vector is selected.
    # Then, we want to maximize sum_{i, j} x_{i, j} * cum_energy_vectors[i][j] subject to the constraints:
    # 1. sum_{j} x_{i, j} = 1 for all i
    # 2. sum_{i, j} j * x_{i, j} * cost_vectors[i][j] <= total_cost

    prob = (
        pulp.LpProblem("MaximizeEnergy", pulp.LpMaximize)
        if not minimize
        else pulp.LpProblem("MinimizeEnergy", pulp.LpMinimize)
    )

    selection_vars = {}
    for vec_idx, vec in enumerate(cum_energy_vectors):
        for idx in range(len(vec)):
            selection_vars[(vec_idx, idx)] = pulp.LpVariable(
                f"x_{vec_idx}_{idx}", cat="Binary"
            )
    prob += pulp.lpSum(
        selection_vars[(vec_idx, idx)] * cum_energy_vectors[vec_idx][idx].item()
        for vec_idx, vec in enumerate(cum_energy_vectors)
        for idx in range(len(vec))
    )

    prob += (
        pulp.lpSum(
            selection_vars[(vec_idx, idx)]
            * cumulative_cost_vectors[vec_idx][idx].item()
            for vec_idx, vec in enumerate(cum_energy_vectors)
            for idx in range(len(vec))
        )
        <= total_cost
    )
    for vec_idx, vec in enumerate(cum_energy_vectors):
        prob += (
            pulp.lpSum(selection_vars[(vec_idx, idx)] for idx in range(len(vec))) == 1
        )

    prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=60))

    selected_indices = {}
    for vec_idx, vec in enumerate(cum_energy_vectors):
        sel = [pulp.value(selection_vars[(vec_idx, idx)]) for idx in range(len(vec))]
        selected_indices[vec_idx] = torch.argmax(torch.tensor(sel)).item()

    return selected_indices
