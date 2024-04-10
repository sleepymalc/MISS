import numpy as np
from sklearn.linear_model import LinearRegression
from target import target_phi, target_influence
from actual import actual_effect
from target import target_value

def diagonal(X_train, y_train, X_test, y_test, target="linear"):
    n = X_train.shape[0]
    lr = LinearRegression().fit(X_train, y_train)

    phi = target_phi(X_train, y_train, X_test, y_test, target=target)

    N = X_train.T @ X_train
    H = X_train @ np.linalg.inv(N) @ X_train.T

    param_influence = np.linalg.inv(N) @ X_train.T * (lr.predict(X_train) - y_train)

    influence = target_influence(phi, param_influence, target=target) / (1 - np.diag(H))
    Diag_best = np.argsort(influence)[-n:][::-1]

    return Diag_best

def adaptive_diagonal(X_train, y_train, X_test, y_test, k=5, target="linear"):
    n = X_train.shape[0]
    lr = LinearRegression().fit(X_train, y_train)

    X_train_with_index = np.hstack((X_train, np.arange(n).reshape(-1, 1)))
    adaptive_Diag_best_k = np.zeros(k, dtype=int)

    for i in range(k):
        phi = target_phi(X_train, y_train, X_test, y_test, target=target)

        N = X_train.T @ X_train
        H = X_train @ np.linalg.inv(N) @ X_train.T

        param_influence = np.linalg.inv(N) @ X_train.T * (lr.predict(X_train) - y_train)

        influence = target_influence(phi, param_influence, target=target) / (1 - np.diag(H))

        print_size = k * 2
        top_indices = np.argsort(influence)[-(print_size):][::-1]

        actual_top_indices = X_train_with_index[:, -1][top_indices].astype(int)
        adaptive_Diag_best_k[i] = actual_top_indices[0]

        # Remove the most influential data points
        X_train = np.delete(X_train, top_indices[0], axis=0)
        X_train_with_index = np.delete(X_train_with_index, top_indices[0], axis=0)
        y_train = np.delete(y_train, top_indices[0], axis=0)


        lr = LinearRegression().fit(X_train_with_index[:, :-1], y_train)

    return adaptive_Diag_best_k

def heuristic_diagonal(X_train, y_train, X_test, y_test, k=5, target="linear"):
    n = X_train.shape[0]
    # Create n candidate subsets holder
    subsets = np.zeros((n, k), dtype=int)
    score = np.zeros(n)
    original_value = target_value(X_train, y_train, X_test, target=target)
    for i in range(n):
        subsets[i] = np.append(adaptive_diagonal(np.delete(X_train, i, axis=0), np.delete(y_train, i, axis=0), X_test, y_test, k=k-1, target=target), i)
        score[i] = actual_effect(X_train, y_train, X_test, subsets[i], original_value, target=target)

    # Get the best subset
    best_subset = subsets[np.argmax(score)]
    return best_subset