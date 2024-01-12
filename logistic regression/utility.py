import numpy as np
from target import target_value
from actual import actual_effect


def actual_rank(X_train, y_train, x_test, y_test, subset_to_remove, score, target="probability"):
    original_value = target_value(X_train, y_train, x_test, y_test, target=target)

    actual_score = actual_effect(X_train, y_train, x_test, y_test, subset_to_remove, original_value, target=target)

    # tie-handling. ref: https://stackoverflow.com/questions/39059371/can-numpys-argsort-give-equal-element-the-same-rank
    def rankmin(x):
        u, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
        csum = np.zeros_like(counts)
        csum[1:] = counts[:-1].cumsum()
        return csum[inv]+1
    
    score_rank = rankmin(-1 * np.array(score))
    return score_rank[np.where(score == actual_score)[0][0]]