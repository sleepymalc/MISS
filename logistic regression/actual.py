import numpy as np
from joblib import Parallel, delayed
from itertools import combinations
from target import target_value

def actual_effect(X_train, y_train, X_test, y_test, subset_to_remove, original_value, target="probability"):
    reduced_X_train = np.delete(X_train, subset_to_remove, axis=0)
    reduced_y_train = np.delete(y_train, subset_to_remove, axis=0)
    reduced_value = target_value(reduced_X_train, reduced_y_train, X_test, y_test, target)

    if target == "avg_abs_test_loss":
        score = np.mean(np.abs(original_value - reduced_value))
    elif target in ["abs_probability", "abs_test_loss"]:
        score = np.abs(original_value - reduced_value)
    else:
        score = reduced_value - original_value

    return score

def actual(X_train, y_train, X_test, y_test, k=5, job_n=50, target="probability"):
    best_subset = np.full((k), None)
    score = []
    original_value = target_value(X_train, y_train, X_test, y_test, target=target)

    for subset_size in range(1, k + 1):
        # Generate all combinations of subsets of the current size
        subset_combinations = combinations(range(X_train.shape[0]), subset_size)
        combinations_list = list(combinations(range(X_train.shape[0]), subset_size))
        
        best_k_score = Parallel(n_jobs=job_n)(delayed(actual_effect)(X_train, y_train, X_test, y_test, subset_to_remove, original_value, target) for subset_to_remove in subset_combinations)
        
        sort_subset_combinations = np.array(combinations_list)[np.argsort(best_k_score)[::-1]]
        best_subset[subset_size - 1] = sort_subset_combinations[0]
        score.append(best_k_score) # TODO: Flatten this if we want to get <= k

    return [score, best_subset]