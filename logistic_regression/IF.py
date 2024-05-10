import numpy as np
from sklearn.linear_model import LogisticRegression

def first_order(X_train, Y_train, x_test, y_test):
    n = X_train.shape[0]
    lr = LogisticRegression(penalty=None).fit(X_train, Y_train)

    X_train_bar = np.hstack((np.ones((X_train.shape[0], 1)), X_train))

    # Compute the Hessian w.r.t. the parameters
    Hessian_inv = np.linalg.inv(np.dot(X_train_bar.T, np.dot(np.diag(lr.predict_proba(X_train)[:, 1] * (1 - lr.predict_proba(X_train)[:, 1])), X_train_bar)) / n)

    # Compute the gradient of the logistic loss w.r.t. the parameters
    sigma_train = lr.predict_proba(X_train)[:, 1] # P(1 | x)
    grad_loss_train = (sigma_train - Y_train) * X_train_bar.T
    param_influence = Hessian_inv @ grad_loss_train / n

    x_test_bar = np.hstack((1, x_test))
    sigma_test = lr.predict_proba(x_test.reshape(1, -1))[0][y_test]
    phi = (1 - sigma_test) * sigma_test * x_test_bar

    influence = (phi @ param_influence).reshape(-1)

    FO_best = np.argsort(influence)[-n:][::-1]

    return FO_best

def adaptive_first_order(X_train, Y_train, x_test, y_test, k=5):
    n = X_train.shape[0]
    X_train_bar = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
    index = list(range(n))
    adaptive_FO_best_k = np.zeros(k, dtype=int)

    for i in range(k):
        lr = LogisticRegression(penalty=None).fit(X_train, Y_train)

        Hessian_inv = np.linalg.inv(np.dot(X_train_bar.T, np.dot(np.diag(lr.predict_proba(X_train)[:, 1] * (1 - lr.predict_proba(X_train)[:, 1])), X_train_bar)) / n)

        # Compute the gradient of the logistic loss w.r.t. the parameters
        sigma_train = lr.predict_proba(X_train)[:, 1] # P(1 | x)
        grad_loss_train = (sigma_train - Y_train) * X_train_bar.T
        param_influence = Hessian_inv @ grad_loss_train / n

        x_test_bar = np.hstack((1, x_test))
        sigma_test = lr.predict_proba(x_test.reshape(1, -1))[0][y_test]
        phi = (1 - sigma_test) * sigma_test * x_test_bar

        influence = (phi @ param_influence).reshape(-1)

        top_index = np.argsort(influence)[-1:][::-1][0]
        adaptive_FO_best_k[i] = index[top_index]

        # Remove the most influential data points
        X_train = np.delete(X_train, top_index, axis=0)
        X_train_bar = np.delete(X_train_bar, top_index, axis=0)
        Y_train = np.delete(Y_train, top_index, axis=0)
        index = np.delete(index, top_index, axis=0)

    return adaptive_FO_best_k