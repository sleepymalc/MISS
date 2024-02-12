import numpy as np
from sklearn.linear_model import LogisticRegression
from target import target_phi, target_influence

def WLS_influence(X, y, coef, W, phi, target="probability"):
    N = np.dot(W * X.T, X)
    N_inv = np.linalg.inv(N)
    r = W * (np.dot(X, coef) - y)

    param_influence = N_inv @ X.T * r

    influence = target_influence(phi, param_influence, target=target)
    influence = influence / (1 - np.diag(np.diag(W) @ X @ N_inv @ X.T)) # adjust by leverage score

    return influence

def IWLS(X_train, y_train, X_test, y_test, target="probability"):
    n = X_train.shape[0]
    lr = LogisticRegression(penalty=None).fit(X_train, y_train)
    coefficients = np.concatenate((np.array([lr.intercept_[0]]), lr.coef_[0]))
    p = lr.predict_proba(X_train)[:, 1]
    W = p * (1 - p)

    phi = target_phi(X_train, y_train, X_test, y_test, target=target)

    X_train_bar = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
    y = np.dot(X_train_bar, coefficients) + (y_train - p) / W

    influence = WLS_influence(X_train_bar, y, coefficients, W, phi, target=target)

    IWLS_best = np.argsort(influence)[-n:][::-1]

    return IWLS_best

# TODO: Currently the edge case (when k ~= n) is not handled
def adaptive_IWLS(X_train, y_train, X_test, y_test, k=5, target="probability"):
    n = X_train.shape[0]
    lr = LogisticRegression(penalty=None).fit(X_train, y_train)
    coefficients = np.concatenate((np.array([lr.intercept_[0]]), lr.coef_[0]))
    p = lr.predict_proba(X_train)[:, 1]

    X_train_bar = np.hstack((np.ones((n, 1)), X_train))
    X_train_bar_with_index = np.hstack((X_train_bar, np.arange(n).reshape(-1, 1)))
    adaptive_IWLS_best_k = np.zeros(k, dtype=int)


    for i in range(k):
        W = p * (1 - p)
        X = X_train_bar_with_index[:, :-1] # without index
        y = np.dot(X, coefficients) + (y_train - p) / W

        # Calculate phi adaptively
        phi = target_phi(X_train, y_train, X_test, y_test, target=target)

        influence = WLS_influence(X, y, coefficients, W, phi, target=target)

        print_size = k * 2
        top_indices = np.argsort(influence)[-(print_size):][::-1]

        actual_top_indices = X_train_bar_with_index[:, -1][top_indices].astype(int)
        adaptive_IWLS_best_k[i] = actual_top_indices[0]

        # Remove the most influential data points
        X = np.delete(X, top_indices[0], axis=0)
        X_train = np.delete(X_train, top_indices[0], axis=0)
        X_train_bar = np.delete(X_train_bar, top_indices[0], axis=0)
        X_train_bar_with_index = np.delete(X_train_bar_with_index, top_indices[0], axis=0)
        y_train = np.delete(y_train, top_indices[0], axis=0)

        # Train to full convergence
        lr = LogisticRegression(penalty=None).fit(X_train_bar_with_index[:, 1:-1], y_train)
        coefficients = np.concatenate((np.array([lr.intercept_[0]]), lr.coef_[0]))
        p = lr.predict_proba(X_train)[:, 1]

    return adaptive_IWLS_best_k