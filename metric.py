import numpy as np

def Borda_count(ranks):
    # create weights based on the number of methods
    num_methods, num_experiments = ranks.shape
    weights = np.arange(num_methods, 0, -1)

    weighted_borda_count = np.zeros((num_methods, num_experiments), dtype=float)  # Change dtype to float

    # Calculate weighted Borda count for each seed and covariance
    for experiment_idx in range(num_experiments):
        # Sort indices based on actual ranks for the current experiment
        # tie-handling. ref: https://stackoverflow.com/questions/39059371/can-numpys-argsort-give-equal-element-the-same-rank
        def rankmin(x):
            u, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
            csum = np.zeros_like(counts)
            csum[1:] = counts[:-1].cumsum()
            return csum[inv]

        sorted_indices = rankmin(-1 * ranks[:, experiment_idx])

        # Calculate average weight for tied ranks
        average_weights = np.zeros_like(weights, dtype=float)
        unique_sorted_indices, counts = np.unique(sorted_indices, return_counts=True)
        tie_weights = weights[np.argsort(-sorted_indices)[::-1]]

        for idx, count in zip(unique_sorted_indices, counts):
            average_weights[idx] = np.sum(tie_weights[sorted_indices == idx]) / count

        # Assign weighted Borda count scores
        for method_idx, rank in enumerate(sorted_indices):
            weighted_borda_count[method_idx, experiment_idx] = average_weights[rank]

    total_weighted_borda_count = weighted_borda_count.sum(axis=1)

    return total_weighted_borda_count

def winning_rate(ranks):
    # Create weights based on the number of methods
    num_methods, num_experiments = ranks.shape
    weights = np.arange(num_methods-1, -1, -1)

    weighted_borda_count = np.zeros((num_methods, num_experiments), dtype=float)

    # Calculate weighted Borda count for each seed and covariance
    for experiment_idx in range(num_experiments):
        # Sort indices based on actual ranks for the current experiment
        def rankmin(x):
            u, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
            csum = np.zeros_like(counts)
            csum[1:] = counts[:-1].cumsum()
            return csum[inv]

        sorted_indices = rankmin(-1 * ranks[:, experiment_idx])

        # Calculate average weight for tied ranks
        average_weights = np.zeros_like(weights, dtype=float)
        unique_sorted_indices, counts = np.unique(sorted_indices, return_counts=True)
        tie_weights = weights[np.argsort(-sorted_indices)[::-1]]

        for idx, count in zip(unique_sorted_indices, counts):
            average_weights[idx] = np.sum(tie_weights[sorted_indices == idx]) / count

        # Assign weighted Borda count scores
        for method_idx, rank in enumerate(sorted_indices):
            weighted_borda_count[method_idx, experiment_idx] = average_weights[rank]

    total_weighted_borda_count = weighted_borda_count.sum(axis=1)

    # Compute winning rate
    winning_rates = total_weighted_borda_count / total_weighted_borda_count.sum()

    return winning_rates