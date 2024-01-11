import numpy as np
from sklearn.linear_model import LogisticRegression

def first_order(X_train, y_train, x_test, y_test, target="probability"):
    n = X_train.shape[0]
    lr = LogisticRegression(penalty=None).fit(X_train, y_train)

    # Compute the gradient of the logistic loss w.r.t. the parameters
    sigma_train = lr.predict_proba(X_train)[:, 1] # P(1 | x)
    sigma_test = lr.predict_proba(x_test.reshape(1, -1))[0][1]
    grad_loss_train = (sigma_train - y_train) * X_train.T
    grad_loss_test = (sigma_test - y_test) * x_test

    # Compute the Hessian w.r.t. the parameters
    Hessian = np.dot(X_train.T, np.dot(np.diag(lr.predict_proba(X_train)[:, 1] * (1 - lr.predict_proba(X_train)[:, 1])), X_train)) / n

    Hessian_inv = np.linalg.inv(Hessian)
    param_influences = Hessian_inv @ grad_loss_train
 
    if target == "probability":
        phi = (1 - sigma_test) * sigma_test * x_test * grad_loss_test.T
        influences = - phi @ param_influences
    elif target == "train_loss":
        phi = grad_loss_train.T
        influences = - np.sum(phi @ param_influences, axis=0)
    elif target == "test_loss":
        phi = grad_loss_test.T
        influences = - phi @ param_influences
  
    FO_best = np.argsort(influences)[-n:][::-1]
    return FO_best