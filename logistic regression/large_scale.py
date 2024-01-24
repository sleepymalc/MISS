import warnings
warnings.filterwarnings("ignore")
import argparse

import numpy as np
import matplotlib.pyplot as plt

from actual import actual_effect
from IWLS import IWLS, adaptive_IWLS
from first_order import first_order
from margin import margin

from target import target_value
from utility import data_generation

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n', type=int, default=50, help='number of training/testing points')
    parser.add_argument('--d', type=int, default=2, help='dimension')
    parser.add_argument('--k', type=int, default=2, help='subset size')
    args = parser.parse_args()
    
    return args

args = get_args()

# general parameters
n = args.n
d = args.d
k = args.k

seeds = range(1, 20)
covs = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 10.0]

targets = ["probability", "abs_probability", "test_loss", "abs_test_loss", "avg_abs_test_loss", "abs_avg_test_loss"]

methods = ["IWLS", "Adaptive IWLS", "Margin-based", "First-order"]
num_methods, num_covs, num_seeds = len(methods), len(covs), len(seeds)

# ratios w.r.t. A-IWLS
def score_ratio_per_seed_cov(seed, cov, target):
    X_train, y_train, X_test, y_test = data_generation(n, d, cov, seed, target=target)
    
    original_value = target_value(X_train, y_train, X_test, y_test, target=target)
    
    ind_n, ind_p = margin(X_train, y_train)

    scores = {
        "IWLS": actual_effect(X_train, y_train, X_test, y_test, IWLS(X_train, y_train, X_test, y_test, target=target)[:k], original_value, target=target),
        "Adaptive IWLS": actual_effect(X_train, y_train, X_test, y_test, adaptive_IWLS(X_train, y_train, X_test, y_test, k=k, target=target), original_value, target=target),
        "Margin-based": max(actual_effect(X_train, y_train, X_test, y_test, ind_n[:k], original_value, target=target), actual_effect(X_train, y_train, X_test, y_test, ind_p[:k], original_value, target=target)),
        "First-order": actual_effect(X_train, y_train, X_test, y_test, first_order(X_train, y_train, X_test, y_test, target=target)[:k], original_value, target=target)
    }

    if scores["Adaptive IWLS"] == 0:
        return {
            "IWLS": 1 if scores["IWLS"] == 0 else 0,
            "Adaptive IWLS": 1 if scores["Adaptive IWLS"] == 0 else 0,
            "Margin-based": 1 if scores["Margin-based"] == 0 else 0,
            "First-order": 1 if scores["First-order"] == 0 else 0
        }
    
    ratio = {
        "IWLS": scores["IWLS"] / scores["Adaptive IWLS"],
        "Adaptive IWLS": scores["Adaptive IWLS"] / scores["Adaptive IWLS"],
        "Margin-based": scores["Margin-based"] / scores["Adaptive IWLS"],
        "First-order": scores["First-order"] / scores["Adaptive IWLS"]
    }
    
    return ratio

fig, axs = plt.subplots(2, 3, figsize=(15, 10))  # Create a 2x3 grid of subplots

for target_idx, target in enumerate(targets):
    print(f"Target: {target}")
    ratio_array = np.zeros((num_seeds, num_covs, num_methods), dtype=float)
    
    # Process each seed and covariance and populate the array
    for seed_idx, seed in enumerate(seeds):
        for cov_idx, cov in enumerate(covs):
            ratio = score_ratio_per_seed_cov(seed, cov, target)
            for method_idx, method_name in enumerate(methods):
                ratio_array[seed_idx, cov_idx, method_idx] = ratio.get(method_name, 0)

    ratio_method_cov_seed = ratio_array.swapaxes(0, 2)  # method, cov, seed
    ratio_cov_method_seed = ratio_method_cov_seed.swapaxes(0, 1)  # cov, method, seed
    ratio_result = np.zeros((num_covs, num_methods), dtype=float)

    for cov_idx in range(num_covs):
        ratio_result[cov_idx] = ratio_cov_method_seed[cov_idx].mean(axis=1)

    # Plot in the corresponding subplot
    row_idx, col_idx = divmod(target_idx, 3)  # Calculate subplot index
    for method_idx, method_name in enumerate(methods):
        axs[row_idx, col_idx].plot(covs, ratio_result[:, method_idx], label=method_name)
    # axs[row_idx, col_idx].plot(covs, ratio_result)
    axs[row_idx, col_idx].set_title(f'Target={target}')
    axs[row_idx, col_idx].set_xlabel('Covariance')
    axs[row_idx, col_idx].set_ylabel('Average Ratio of Changes w.r.t. Margin-Based')
    axs[row_idx, col_idx].legend(methods)
    
    axs[row_idx, col_idx].set_xticks(covs)
    
    # Set log-scale on the y-axis
    axs[row_idx, col_idx].set_yscale('log')

plt.tight_layout(rect=[0, 0.03, 1, 0.95])

plt.suptitle(f'n={n} d={d} k={k}', fontsize=16)

# Save the figure locally
plt.savefig(f'n={n}_d={d}_k={k}.png')

# Optionally, you can also close the plot if you don't want to display it
plt.close()