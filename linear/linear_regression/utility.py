import argparse
import numpy as np
from sklearn.datasets import make_spd_matrix

from target import target_value
from actual import actual_effect

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--n', type=int, default=50, help='number of training/testing points')
    parser.add_argument('--d', type=int, default=2, help='dimension')
    parser.add_argument('--k', type=int, default=2, help='subset size')
    parser.add_argument('--cov', type=float, default=1, help='covariance of error')
    parser.add_argument('--job_n', type=int, default=50, help='number of workers')
    parser.add_argument('--target', type=str, default='linear', help='target function')
    parser.add_argument('--skewed', action='store_true', help='skewed normal')
    args = parser.parse_args()

    return args

def data_generation(n, d, cov, seed, isSkewed=False, target="linear"):
    np.random.seed(seed)
    mean = np.zeros(d-1)

    if isSkewed:
        # Generate a random positive definite covariance matrix
        raw_covariance = make_spd_matrix(d)
        covariance = cov * (1/np.linalg.det(raw_covariance))**(1/d) * raw_covariance
    else:
        covariance = np.eye(d-1) * cov

    # generate linear regression data (X, y = θ^T X + ε) with ε ~ N(0, cov) with bias at the first dimension
    X_train = np.hstack((np.ones((n, 1)), np.random.multivariate_normal(mean, covariance, n)))
    theta = np.random.uniform(-1, 1, d)
    y_train = np.dot(X_train, theta) + np.random.normal(0, cov, n)

    if target in ["linear"]: # only one test point
        X_test = np.hstack((np.ones((1, 1)), np.random.multivariate_normal(mean, covariance, 1)))
        y_test = np.dot(X_test, theta) + np.random.normal(0, cov, 1)
    else: # no test point needed
        X_test = None
        y_test = None

    return X_train, y_train, X_test, y_test

def actual_rank(X_train, y_train, X_test, y_test, subset_to_remove, score, target="linear"):
    original_value = target_value(X_train, y_train, X_test, target=target)
    actual_score = actual_effect(X_train, y_train, X_test, subset_to_remove, original_value, target=target)

    # tie-handling. ref: https://stackoverflow.com/questions/39059371/can-numpys-argsort-give-equal-element-the-same-rank
    def rankmin(x):
        u, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
        csum = np.zeros_like(counts)
        csum[1:] = counts[:-1].cumsum()
        return csum[inv]+1

    score_rank = rankmin(-1 * np.array(score))
    return score_rank[np.where(score == actual_score)[0][0]]