import argparse
from joblib import Parallel, delayed
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from itertools import combinations

def actual_effect(X_train, y_train, x_test, y_test, subset_to_remove, original_score, target="probability"):
    # Train a Logistic Regression classifier on the reduced training set
    reduced_X_train = np.delete(X_train, subset_to_remove, axis=0)
    reduced_y_train = np.delete(y_train, subset_to_remove, axis=0)
    reduced_lr = LogisticRegression(penalty=None).fit(reduced_X_train, reduced_y_train)

    # Make inference
    if target == "probability":
        reduced_score = reduced_lr.predict_proba(x_test.reshape(1, -1))[0][1]
    elif target == "train_loss":
        reduced_score = log_loss(reduced_y_train, reduced_lr.predict_proba(reduced_X_train), labels=[0, 1])
    elif target == "test_loss":
        reduced_score = log_loss([y_test], reduced_lr.predict_proba(x_test.reshape(1, -1)), labels=[0, 1])
  
    # Calculate the difference in predicted probabilities
    score_difference = reduced_score - original_score

    return score_difference

# TODO: The actual effect of a specific k, not <= k
def actual(X_train, y_train, x_test, y_test, k=10, job_n=50, target="probability"):
    # Create a Logistic Regression classifier
    original_lr = LogisticRegression(penalty=None).fit(X_train, y_train)
 
    # Initialize variables to keep track of the best subset and loss difference for parameter changes
    best_subset = np.full((k), None)
    score = []
    
    if target == "probability":
        original_score = original_lr.predict_proba(x_test.reshape(1, -1))[0][1] # We're looking at the predicted probability of the positive class
    elif target == "train_loss":
        original_score = log_loss(y_train, original_lr.predict_proba(X_train), labels=[0, 1])
    elif target == "test_loss":
        original_score = log_loss([y_test], original_lr.predict_proba(x_test.reshape(1, -1)), labels=[0, 1])

    # Loop over different subset sizes from 1 to k
    for subset_size in range(1, k + 1):
        # Generate all combinations of subsets of the current size
        subset_combinations = combinations(range(X_train.shape[0]), subset_size)
        combinations_list = list(combinations(range(X_train.shape[0]), subset_size))
        
        best_k_score = Parallel(n_jobs=job_n)(delayed(actual_effect)(X_train, y_train, x_test, y_test, subset_to_remove, original_score, target) for subset_to_remove in subset_combinations)
        
        sort_subset_combinations = np.array(combinations_list)[np.argsort(best_k_score)[::-1]]
        best_subset[subset_size - 1] = sort_subset_combinations[0]
        score.append(best_k_score) # TODO: Flatten this if we want to get <= k

    return [score, best_subset]

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

# Calculate adaptive influences TODO: Currently the edge case (when k ~= n) is not handled
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
  
# Margin-based approach
def margin(X_train, y_train):
    n = X_train.shape[0]
    lr = LogisticRegression(penalty=None).fit(X_train, y_train)
    param = np.concatenate(([lr.intercept_[0]], lr.coef_[0]))

    X_train_bar = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
    margin = (X_train_bar @ param) * (1 - 2 * y_train)

    margin_n = margin[:int(n/2)]
    margin_p = margin[int(n/2):]
 
    ind_n = np.argsort(margin_n)[-int(n/2):][::-1]
    ind_p = np.argsort(margin_p)[-int(n/2):][::-1] + len(margin_p)
 
    return ind_n, ind_p

# First-order method
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

# Result
def actual_rank(X_train, y_train, x_test, y_test, subset_to_remove, score, target="probability"):
    original_lr = LogisticRegression(penalty=None).fit(X_train, y_train)

    if target == "probability":
            original_score = original_lr.predict_proba(x_test.reshape(1, -1))[0][1] # We're looking at the predicted probability of the positive class
    elif target == "train_loss":
        original_score = log_loss(y_train, original_lr.predict_proba(X_train), labels=[0, 1])
    elif target == "test_loss":
        original_score = log_loss([y_test], original_lr.predict_proba(x_test.reshape(1, -1)), labels=[0, 1])

    actual_score = actual_effect(X_train, y_train, x_test, y_test, subset_to_remove, original_score, target=target)

    # tie-handling. ref: https://stackoverflow.com/questions/39059371/can-numpys-argsort-give-equal-element-the-same-rank
    def rankmin(x):
        u, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
        csum = np.zeros_like(counts)
        csum[1:] = counts[:-1].cumsum()
        return csum[inv]+1
    
    score_rank = rankmin(-1 * np.array(score))
    return score_rank[np.where(score == actual_score)[0][0]]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=20)
    parser.add_argument('--n', type=int, default=50)
    parser.add_argument('--k', type=int, default=2)
    parser.add_argument('--cov', type=float, default=1)
    parser.add_argument('--job_n', type=int, default=50)
    parser.add_argument('--target', type=str, default='probability')
    args = parser.parse_args()

    # general parameters
    n = args.n
    k = args.k
    job_n = args.job_n
    cov = args.cov
    target = args.target
    seed = args.seed
    
    np.random.seed(seed)

    out_file = f"results/target={target}/s={seed}_n={n}_k={k}_cov={cov}.txt"

    # generate data
    mean_n = np.array([-1, 0])
    mean_p = np.array([1, 0])
    covariance = np.eye(2) * cov  
    x_n = np.random.multivariate_normal(mean_n, covariance, int(n/2))
    x_p = np.random.multivariate_normal(mean_p, covariance, int(n/2))

    y_n = np.zeros(int(n/2)) # 0 labels
    y_p = np.ones(int(n/2))  # 1 labels

    X_train = np.vstack((x_n, x_p))
    y_train = np.hstack((y_n, y_p))

    # Choose mean_n or mean_p wp 1/2
    if np.random.rand() < 0.5:
        x_test = np.random.multivariate_normal(mean_n, covariance)
        y_test = 0
    else:
        x_test = np.random.multivariate_normal(mean_p, covariance)
        y_test = 1

    print_size = k * 2

    # Best Subset
    score, best_subset = actual(X_train, y_train, x_test, y_test, k=k, job_n=job_n, target=target)
    best_k_score = score[-1]

    with open(out_file, 'w') as f:
        f.write('Best Subset\n')
        for subset_size in range(1, k + 1):
            f.write(f"\tsize {subset_size}: {best_subset[subset_size-1]}\n")
        f.write('\n')

    # IWLS
    IWLS_best = IWLS(X_train, y_train, x_test, y_test, target=target)
    with open(out_file, 'a') as f:
        f.write('IWLS Best Subset\n')
        f.write(f"\ttop {print_size}: {IWLS_best[:print_size]}\n\n")
    
    # Adaptive IWLS
    adaptive_IWLS_best_k = adaptive_IWLS(X_train, y_train, x_test, y_test, k=k, target=target)
    with open(out_file, 'a') as f:
        f.write('Adaptive IWLS Best Subset\n')
        f.write(f"\ttop {k}: {adaptive_IWLS_best_k}\n\n")

    # Margin-based approach
    ind_n, ind_p = margin(X_train, y_train)
    with open(out_file, 'a') as f:
        f.write('Margin-based Best Subset\n')
        f.write(f"\tpositive group:\ttop {print_size}: {ind_p[:print_size]}\n")
        f.write(f"\tnegative group:\ttop {print_size}: {ind_n[:print_size]}\n\n")

    # First-order method
    FO_best = first_order(X_train, y_train, x_test, y_test, target=target)
    with open(out_file, 'a') as f:
        f.write('First-order Best Subset\n')
        f.write(f"\ttop {print_size}: {FO_best[:print_size]}\n\n")

    with open(out_file, 'a') as f:
        f.write('IWLS Best Subset v.s. Best Subset (size=k)\n')
        rank = actual_rank(X_train, y_train, x_test, y_test, IWLS_best[:k], best_k_score, target=target)
        f.write(f'\tActual rank: {rank}\n\n')

    
        f.write('Adaptive IWLS Best Subset v.s. Best Subset (size=k)\n')
        rank = actual_rank(X_train, y_train, x_test, y_test, adaptive_IWLS_best_k, best_k_score, target=target)
        f.write(f'\tActual rank: {rank}\n\n')


        f.write('Margin-based Best Subset v.s. Best Subset (size=k)\n')
        f.write(f'P Group\n')
        rank = actual_rank(X_train, y_train, x_test, y_test, ind_p[:k], best_k_score, target=target)
        f.write(f'\tActual rank: {rank}\n')
    
        f.write(f'N Group\n')
        rank = actual_rank(X_train, y_train, x_test, y_test, ind_n[:k], best_k_score, target=target)
        f.write(f'\tActual rank: {rank}\n\n')
        
        f.write('First-order Best Subset v.s. Best Subset (size=k)\n')
        rank = actual_rank(X_train, y_train, x_test, y_test, FO_best[:k], best_k_score, target=target)
        f.write(f'\tActual rank: {rank}\n')