import argparse
import numpy as np

from actual import actual, actual_effect
from IWLS import IWLS, adaptive_IWLS
from first_order import first_order
from margin import margin

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