import numpy as np
from sklearn.linear_model import LinearRegression

def first_order(X_train, y_train, x_test):
    n = X_train.shape[0]
    lr = LinearRegression().fit(X_train, y_train)

    N_inv = np.linalg.inv(X_train.T @ X_train)
    influence = ((x_test @ N_inv @ X_train.T).reshape(-1, 1) * (lr.predict(X_train) - y_train)).reshape(-1)

    FO_best = np.argsort(influence)[-n:][::-1]

    return FO_best

def adaptive_first_order(X_train, y_train, x_test, k=5):
    n = X_train.shape[0]
    index = list(range(n))
    adaptive_FO_best_k = np.zeros(k, dtype=int)

    for i in range(k):
        lr = LinearRegression().fit(X_train, y_train)

        N_inv = np.linalg.inv(X_train.T @ X_train)
        influence = ((x_test @ N_inv @ X_train.T).reshape(-1, 1) * (lr.predict(X_train) - y_train)).reshape(-1)

        top_index = np.argsort(influence)[-1:][::-1][0]
        adaptive_FO_best_k[i] = index[top_index]

        # Remove the most influential data points
        X_train = np.delete(X_train, top_index, axis=0)
        y_train = np.delete(y_train, top_index, axis=0)
        index = np.delete(index, top_index, axis=0)

    return adaptive_FO_best_k