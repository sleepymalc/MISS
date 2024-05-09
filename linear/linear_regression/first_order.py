import numpy as np
from sklearn.linear_model import LinearRegression
from target import target_phi, target_influence

def first_order(X_train, y_train, X_test, y_test, target="linear"):
    n = X_train.shape[0]
    lr = LinearRegression().fit(X_train, y_train)

    phi = target_phi(X_train, y_train, X_test, y_test, target=target)

    N = X_train.T @ X_train
    param_influence = np.linalg.inv(N) @ X_train.T @ (lr.predict(X_train) - y_train)

    influence = target_influence(phi, param_influence, target=target)

    FO_best = np.argsort(influence)[-n:][::-1]

    return FO_best

def adaptive_first_order(X_train, y_train, X_test, y_test, k=5, target="linear"):
    n = X_train.shape[0]
    lr = LinearRegression().fit(X_train, y_train)

    X_train_with_index = np.hstack((X_train, np.arange(n).reshape(-1, 1)))
    adaptive_FO_best_k = np.zeros(k, dtype=int)

    for i in range(k):
        phi = target_phi(X_train, y_train, X_test, y_test, target=target)

        N = X_train.T @ X_train
        param_influence = np.linalg.inv(N) @ X_train.T @ (lr.predict(X_train) - y_train)

        influence = target_influence(phi, param_influence, target=target)

        print_size = k * 2
        top_indices = np.argsort(influence)[-(print_size):][::-1]

        actual_top_indices = X_train_with_index[:, -1][top_indices].astype(int)
        adaptive_FO_best_k[i] = actual_top_indices[0]

        # Remove the most influential data points
        X_train = np.delete(X_train, top_indices[0], axis=0)
        X_train_with_index = np.delete(X_train_with_index, top_indices[0], axis=0)
        y_train = np.delete(y_train, top_indices[0], axis=0)


        lr = LinearRegression().fit(X_train_with_index[:, :-1], y_train)

    return adaptive_FO_best_k