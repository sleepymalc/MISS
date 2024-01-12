import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

def target_value(X_train, y_train, X_test, y_test, target="probability"):
    lr = LogisticRegression(penalty=None).fit(X_train, y_train)

    if target == "probability":
        value = lr.predict_proba(X_test)[0][1] # The predicted probability of the positive class
    elif target == "test_loss":
        value = log_loss(y_test, lr.predict_proba(X_test), labels=[0, 1])
    elif target == "avg_train_loss":
        value = log_loss(y_train, lr.predict_proba(X_train), labels=[0, 1])
    elif target == "avg_abs_test_loss":
        value = np.array([log_loss([y_true], [y_pred_prob], labels=[0, 1]) for y_true, y_pred_prob in zip(y_test, lr.predict_proba(X_test))]) # Due to this special target, we return the list of every individual test loss instead of an aggregated average test loss

    return value

def target_phi(X_train, y_train, X_test, y_test, target="probability"):
    lr = LogisticRegression(penalty=None).fit(X_train, y_train)

    if target == "probability":
        X_test_bar = np.hstack((np.ones((X_test.shape[0], 1)), X_test))
        sigma_test = lr.predict_proba(X_test)[0][1]
        phi = (1 - sigma_test) * sigma_test * X_test_bar
    elif target == "test_loss":
        X_test_bar = np.hstack((np.ones((X_test.shape[0], 1)), X_test))
        sigma_test = lr.predict_proba(X_test)[0][1]
        phi = (sigma_test - y_test) * X_test_bar
    elif target == "avg_train_loss":
        X_train_bar = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
        sigma_train = lr.predict_proba(X_train)[:, 1]
        phi = ((sigma_train - y_train) * X_train_bar.T).T
    elif target == "avg_abs_test_loss":
        X_test_bar = np.hstack((np.ones((X_test.shape[0], 1)), X_test))
        sigma_test = lr.predict_proba(X_test)[:, 1]
        phi = ((sigma_test - y_test) * X_test_bar.T).T

    return phi

def target_influence(phi, param_influence, target="probability"):
    if target in ["probability", "test_loss"]:
        influence = (phi @ param_influence).reshape(-1)
    elif target == "avg_train_loss":
        influence = np.sum(phi @ param_influence, axis=0)
    elif target == "avg_abs_test_loss":
        influence = np.sum(np.abs(phi @ param_influence), axis=0)
    
    return influence