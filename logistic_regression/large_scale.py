import os
os.environ["PYTHONWARNINGS"] = "ignore"

import argparse

import numpy as np
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

from actual import actual_effect
from IWLS import IWLS, adaptive_IWLS
from first_order import first_order, adaptive_first_order
from margin import margin

from target import target_value
from utility import data_generation

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=50, help='number of training/testing points')
    parser.add_argument('--d', type=int, default=2, help='dimension')
    parser.add_argument('--k', type=int, default=2, help='subset size')
    parser.add_argument('--job_n', type=int, default=50, help='number of workers')
    parser.add_argument('--skewed', action='store_true', help='skewed normal')
    args = parser.parse_args()
    
    return args

args = get_args()

# general parameters
n = args.n
d = args.d
k = args.k
isSkewed = args.skewed
job_n = args.job_n

seeds = range(0, 50)
covs = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]

targets = ["probability", "abs_probability", "test_loss", "abs_test_loss", "avg_abs_test_loss", "abs_avg_test_loss"]

methods = ["IWLS", "Adaptive IWLS", "Margin-based", "First-order", "Adaptive First-order"]
num_methods, num_covs, num_seeds = len(methods), len(covs), len(seeds)

def score_per_seed_cov(seed, cov, target):
    X_train, y_train, X_test, y_test = data_generation(n, d, cov, seed, isSkewed=isSkewed, target=target)
    
    original_value = target_value(X_train, y_train, X_test, y_test, target=target)
    
    ind_n, ind_p = margin(X_train, y_train)

    scores = np.array([
        actual_effect(X_train, y_train, X_test, y_test, IWLS(X_train, y_train, X_test, y_test, target=target)[:k], original_value, target=target), 
        actual_effect(X_train, y_train, X_test, y_test, adaptive_IWLS(X_train, y_train, X_test, y_test, k=k, target=target), original_value, target=target),
        max(actual_effect(X_train, y_train, X_test, y_test, ind_n[:k], original_value, target=target), actual_effect(X_train, y_train, X_test, y_test, ind_p[:k], original_value, target=target)),
        actual_effect(X_train, y_train, X_test, y_test, first_order(X_train, y_train, X_test, y_test, target=target)[:k], original_value, target=target), 
        actual_effect(X_train, y_train, X_test, y_test, adaptive_first_order(X_train, y_train, X_test, y_test, k=k, target=target), original_value, target=target)
    ])

    return scores

# ranks.shape = (num_methods, num_experiments)
def Borda_count(ranks, weights=[5, 4, 3, 2, 1]):
    num_methods, num_experiments = ranks.shape

    weighted_borda_count = np.zeros((num_methods, num_experiments), dtype=int)

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

        # Assign weighted Borda count scores
        for method_idx, rank in enumerate(sorted_indices):
            weighted_borda_count[method_idx, experiment_idx] = weights[rank]
            
    total_weighted_borda_count = weighted_borda_count.sum(axis=1)

    return total_weighted_borda_count

fig, axs = plt.subplots(2, 3, figsize=(15, 10))  # Create a 2x3 grid of subplots

for target_idx, target in enumerate(targets):
    rank_array = np.zeros((num_seeds, num_covs, num_methods), dtype=int)

    # Process each file and populate the array
    # for seed_idx, seed in enumerate(seeds):
    #     for cov_idx, cov in enumerate(covs):
    #         scores = score_per_seed_cov(seed, cov, target)
    #         for method_idx, method_name in enumerate(methods):
    #             rank_array[seed_idx, cov_idx, method_idx] = sorted(scores, key=scores.get, reverse=True).index(method_name)

    scores_array = np.array(Parallel(n_jobs=job_n)(delayed(score_per_seed_cov)(seed, cov, target) for seed in seeds for cov in covs))
    scores_array = scores_array.reshape((num_seeds, num_covs, -1))
    
    scores_method_cov_seed = scores_array.swapaxes(0, 2) # method, cov, seed
    scores_cov_method_seed = scores_method_cov_seed.swapaxes(0, 1) # cov, method, seed
    
    Borda_result = np.zeros((num_covs, num_methods), dtype=float)
    
    Borda_result = np.array(Parallel(n_jobs=50)(delayed(Borda_count)(scores_cov_method_seed[cov_idx]) for cov_idx in range(num_covs)))

    # Plot in the corresponding subplot
    row_idx, col_idx = divmod(target_idx, 3)  # Calculate subplot index
    for method_idx, method_name in enumerate(methods):
        axs[row_idx, col_idx].plot(covs, Borda_result[:, method_idx], label=method_name)
        
    axs[row_idx, col_idx].set_title(f'Target={target}')
    axs[row_idx, col_idx].set_xlabel('Covariance')
    axs[row_idx, col_idx].set_ylabel('Borda Count')
    axs[row_idx, col_idx].legend(methods)
    
    axs[row_idx, col_idx].set_xticks(covs)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])

if isSkewed:
    title = f'n={n} d={d} k={k} (Skewed).png'
else:
    title = f'n={n} d={d} k={k}.png'

plt.suptitle(title, fontsize=16)

if isSkewed:
    plt.savefig(f'figure/n={n}_d={d}_k={k}_skewed.png')
else:
    plt.savefig(f'figure/n={n}_d={d}_k={k}.png')

# Optionally, you can also close the plot if you don't want to display it
plt.close()