import argparse
import numpy as np
from target import target_value
from actual import actual_effect

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--n', type=int, default=50, help='number of training/testing points')
    parser.add_argument('--d', type=int, default=2, help='dimension')
    parser.add_argument('--k', type=int, default=2, help='subset size')
    parser.add_argument('--cov', type=float, default=1, help='covariance')
    parser.add_argument('--job_n', type=int, default=50, help='number of workers')
    parser.add_argument('--target', type=str, default='probability', help='target function')
    args = parser.parse_args()
    
    return args

def data_generation(n, d, cov, seed, target="probability"):
    np.random.seed(seed)
    
    # generate d-dimensional data
    mean_n = np.zeros(d)
    mean_n[0] = -1

    mean_p = np.zeros(d)
    mean_p[0] = 1
    
    covariance = np.eye(d) * cov
    x_n = np.random.multivariate_normal(mean_n, covariance, int(n/2))
    x_p = np.random.multivariate_normal(mean_p, covariance, int(n/2))

    y_n = np.zeros(int(n/2)) # 0 labels
    y_p = np.ones(int(n/2))  # 1 labels

    X_train = np.vstack((x_n, x_p))
    y_train = np.hstack((y_n, y_p))

    if target in ["probability", "abs_probability", "test_loss", "abs_test_loss"]: # only one test point    
        # Choose mean_n or mean_p wp 1/2
        if np.random.rand() < 0.5:
            X_test = np.random.multivariate_normal(mean_n, covariance, 1)
            y_test = np.zeros(1)
        else:
            X_test = np.random.multivariate_normal(mean_p, covariance, 1)
            y_test = np.ones(1)
    elif target in ["avg_abs_test_loss", "abs_avg_test_loss"]: # n test points
        x_n = np.random.multivariate_normal(mean_n, covariance, int(n/2))
        x_p = np.random.multivariate_normal(mean_p, covariance, int(n/2))
        X_test = np.vstack((x_n, x_p))
        y_test = y_train
    else: # no test point needed
        X_test = None
        y_test = None

    return X_train, y_train, X_test, y_test

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