import numpy as np
from sklearn.linear_model import LogisticRegression
from target import target_phi, target_influence

def first_order(X_train, y_train, X_test, y_test, target="probability"):
    n = X_train.shape[0]
    lr = LogisticRegression(penalty=None).fit(X_train, y_train)

    X_train_bar = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
    
    # Compute the Hessian w.r.t. the parameters
    Hessian = np.dot(X_train_bar.T, np.dot(np.diag(lr.predict_proba(X_train)[:, 1] * (1 - lr.predict_proba(X_train)[:, 1])), X_train_bar)) / n

    Hessian_inv = np.linalg.inv(Hessian)

    # Compute the gradient of the logistic loss w.r.t. the parameters
    sigma_train = lr.predict_proba(X_train)[:, 1] # P(1 | x)
    grad_loss_train = (sigma_train - y_train) * X_train_bar.T
    param_influence = Hessian_inv @ grad_loss_train / n

    phi = target_phi(X_train, y_train, X_test, y_test, target=target)
    
    influence = target_influence(phi, param_influence, target=target)
  
    FO_best = np.argsort(influence)[-n:][::-1]

    return FO_best

def adaptive_first_order(X_train, y_train, X_test, y_test, k=5, target="probability"):
    n = X_train.shape[0]
    lr = LogisticRegression(penalty=None).fit(X_train, y_train)

    X_train_bar = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
    X_train_bar_with_index = np.hstack((X_train_bar, np.arange(n).reshape(-1, 1)))
    adaptive_FO_best_k = np.zeros(k, dtype=int)

    for i in range(k):
        Hessian = np.dot(X_train_bar.T, np.dot(np.diag(lr.predict_proba(X_train)[:, 1] * (1 - lr.predict_proba(X_train)[:, 1])), X_train_bar)) / n

        Hessian_inv = np.linalg.inv(Hessian)

        # Compute the gradient of the logistic loss w.r.t. the parameters
        sigma_train = lr.predict_proba(X_train)[:, 1] # P(1 | x)
        grad_loss_train = (sigma_train - y_train) * X_train_bar.T
        param_influence = Hessian_inv @ grad_loss_train / n

        phi = target_phi(X_train, y_train, X_test, y_test, target=target)
        
        influence = target_influence(phi, param_influence, target=target)    
          
        print_size = k * 2
        top_indices = np.argsort(influence)[-(print_size):][::-1]
        
        actual_top_indices = X_train_bar_with_index[:, -1][top_indices].astype(int)
        adaptive_FO_best_k[i] = actual_top_indices[0]

        # Remove the most influential data points
        X_train = np.delete(X_train, top_indices[0], axis=0)
        X_train_bar = np.delete(X_train_bar, top_indices[0], axis=0)
        X_train_bar_with_index = np.delete(X_train_bar_with_index, top_indices[0], axis=0)
        y_train = np.delete(y_train, top_indices[0], axis=0)
        

        # Train to full convergence
        lr = LogisticRegression(penalty=None).fit(X_train_bar_with_index[:, 1:-1], y_train)

    return adaptive_FO_best_k