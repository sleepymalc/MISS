import numpy as np
from sklearn.linear_model import LogisticRegression

def WLS_influence(X, y, coef, W, phi, target="probability"):
    n = X.shape[0]
    influences = np.zeros(n)
 
    N = np.dot(W * X.T, X)
    N_inv = np.linalg.inv(N)
    r = W * (np.dot(X, coef) - y)
    
    param_influences = N_inv @ X.T * r

    if target == "probability":
        influences = (phi @ param_influences) / (1 - np.diag(np.diag(W) @ X @ N_inv @ X.T))    
    elif target == "train_loss":
        influences = np.sum((phi @ param_influences) / (1 - np.diag(np.diag(W) @ X @ N_inv @ X.T)), axis=0)
    elif target == "test_loss":
        influences = (phi @ param_influences) / (1 - np.diag(np.diag(W) @ X @ N_inv @ X.T))
    
    return influences

def IWLS(X_train, y_train, x_test, y_test, target="probability"):
    n = X_train.shape[0]

    lr = LogisticRegression(penalty=None).fit(X_train, y_train)
    coefficients = np.concatenate((np.array([lr.intercept_[0]]), lr.coef_[0]))
    p = lr.predict_proba(X_train)[:, 1]
    
    W = p * (1 - p)
    X_train_bar = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
    x_test_bar = np.hstack((1, x_test))
    y = np.dot(X_train_bar, coefficients) + (y_train - p) / W
    
    # Calculate phi
    if target == "probability":
        sigma = lr.predict_proba(x_test.reshape(1, -1))[0][1]
        phi = (1 - sigma) * sigma * x_test_bar
    elif target == "train_loss":
        sigma_train = lr.predict_proba(X_train)[:, 1]
        grad_loss_train = (sigma_train - y_train) * X_train_bar.T
        phi = grad_loss_train.T
    elif target == "test_loss":
        sigma = lr.predict_proba(x_test.reshape(1, -1))[0][1]
        grad_loss_test = (sigma - y_test) * x_test_bar
        phi = grad_loss_test.T

    influences = WLS_influence(X_train_bar, y, coefficients, W, phi, target=target)  

    IWLS_best = np.argsort(influences)[-n:][::-1]
 
    return IWLS_best

# TODO: Currently the edge case (when k ~= n) is not handled
def adaptive_IWLS(X_train, y_train, x_test, y_test, k=5, target="probability"):
    n = X_train.shape[0]
    
    lr = LogisticRegression(penalty=None).fit(X_train, y_train)
    coefficients = np.concatenate((np.array([lr.intercept_[0]]), lr.coef_[0]))
    p = lr.predict_proba(X_train)[:, 1]

    X_train_bar = np.hstack((np.ones((n, 1)), X_train))
    x_test_bar = np.hstack((1, x_test))
    X_train_bar_with_index = np.hstack((X_train_bar, np.arange(n).reshape(-1, 1)))
    adaptive_IWLS_best_k = np.zeros(k, dtype=int)

        
    for i in range(k):
        W = p * (1 - p)
        X = X_train_bar_with_index[:, :-1] # without index
        y = np.dot(X, coefficients) + (y_train - p) / W
        
        # Calculate phi adaptively
        if target == "probability":
            sigma = lr.predict_proba(x_test.reshape(1, -1))[0][1]
            phi = (1 - sigma) * sigma * x_test_bar
        elif target == "train_loss":
            sigma_train = lr.predict_proba(X_train)[:, 1]
            grad_loss_train = (sigma_train - y_train) * X_train_bar.T
            phi = grad_loss_train.T
        elif target == "test_loss":
            sigma = lr.predict_proba(x_test.reshape(1, -1))[0][1]
            grad_loss_test = (sigma - y_test) * x_test_bar
            phi = grad_loss_test.T
            
        # Calculate influences
        influences = WLS_influence(X, y, coefficients, W, phi, target=target)
          
        print_size = k * 2
        top_indices = np.argsort(influences)[-(print_size):][::-1]
        
        actual_top_indices = X_train_bar_with_index[:, -1][top_indices].astype(int)
        adaptive_IWLS_best_k[i] = actual_top_indices[0]

        # Remove the most influential data points
        X = np.delete(X, top_indices[0], axis=0)
        X_train = np.delete(X_train, top_indices[0], axis=0)
        X_train_bar = np.delete(X_train_bar, top_indices[0], axis=0)
        X_train_bar_with_index = np.delete(X_train_bar_with_index, top_indices[0], axis=0)
        y_train = np.delete(y_train, top_indices[0], axis=0)
        
        
        # # One step IWLS update
        # X_weighted = X.T * W
        # Hessian = np.dot(X_weighted, X)
        # gradient = np.dot(X.T, y_train - p)
        # coefficients += np.linalg.solve(Hessian, gradient)

        # def sigmoid(z):
        #     return 1 / (1 + np.exp(-z))

        # p = sigmoid(np.dot(X, coefficients))

        # Train to full convergence
        lr = LogisticRegression(penalty=None).fit(X_train_bar_with_index[:, 1:-1], y_train)
        coefficients = np.concatenate((np.array([lr.intercept_[0]]), lr.coef_[0]))
        p = lr.predict_proba(X_train)[:, 1]
    return adaptive_IWLS_best_k