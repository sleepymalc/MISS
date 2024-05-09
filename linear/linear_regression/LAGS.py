import numpy as np
from sklearn.linear_model import LinearRegression

def LAGS(X_train, y_train, x_test):
    n = X_train.shape[0]
    lr = LinearRegression().fit(X_train, y_train)

    N_inv = np.linalg.inv(X_train.T @ X_train)
    H = X_train @ N_inv @ X_train.T
    influence = ((x_test @ N_inv @ X_train.T).reshape(-1, 1) * (lr.predict(X_train) - y_train)).reshape(-1) / (1 - np.diag(H))

    LAGS_best = np.argsort(influence)[-n:][::-1]

    return LAGS_best

def adaptive_LAGS(X_train, y_train, x_test, k=5):
    n = X_train.shape[0]
    index = list(range(n))
    adaptive_LAGS_best_k = np.zeros(k, dtype=int)

    for i in range(k):
        lr = LinearRegression().fit(X_train, y_train)

        N_inv = np.linalg.inv(X_train.T @ X_train)
        H = X_train @ N_inv @ X_train.T
        influence = ((x_test @ N_inv @ X_train.T).reshape(-1, 1) * (lr.predict(X_train) - y_train)).reshape(-1) / (1 - np.diag(H))

        top_index = np.argsort(influence)[-1:][::-1][0]
        adaptive_LAGS_best_k[i] = index[top_index]

        # Remove the most influential data points
        X_train = np.delete(X_train, top_index, axis=0)
        y_train = np.delete(y_train, top_index, axis=0)
        index = np.delete(index, top_index, axis=0)

    return adaptive_LAGS_best_k