import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

def target_value(X_train, y_train, x_test, y_test, target="probability"):
    lr = LogisticRegression(penalty=None).fit(X_train, y_train)

    if target == "probability":
        value = lr.predict_proba(x_test.reshape(1, -1))[0][1] # The predicted probability of the positive class
    elif target == "train_loss":
        value = log_loss(y_train, lr.predict_proba(X_train), labels=[0, 1])
    elif target == "test_loss":
        value = log_loss([y_test], lr.predict_proba(x_test.reshape(1, -1)), labels=[0, 1])

    return value

def target_phi(X_train, y_train, x_test, y_test, target="probability"):
    lr = LogisticRegression(penalty=None).fit(X_train, y_train)

    X_train_bar = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
    x_test_bar = np.hstack((1, x_test))

    if target == "probability":
        sigma_test = lr.predict_proba(x_test.reshape(1, -1))[0][1]
        phi = (1 - sigma_test) * sigma_test * x_test_bar
    elif target == "train_loss":
        sigma_train = lr.predict_proba(X_train)[:, 1]
        grad_loss_train = (sigma_train - y_train) * X_train_bar.T
        phi = grad_loss_train.T
    elif target == "test_loss":
        sigma_test = lr.predict_proba(x_test.reshape(1, -1))[0][1]
        grad_loss_test = (sigma_test - y_test) * x_test_bar
        phi = grad_loss_test.T

    return phi

def target_influence(phi, param_influence, target="probability"):
    if target == "probability":
        influence = phi @ param_influence
    elif target == "train_loss":
        influence = np.sum(phi @ param_influence, axis=0)
    elif target == "test_loss":
        influence = phi @ param_influence
    
    return influence
    