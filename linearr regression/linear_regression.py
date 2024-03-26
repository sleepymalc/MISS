import warnings
warnings.filterwarnings("ignore")

from actual import actual
from first_order import first_order, adaptive_first_order
from diagonal import diagonal, adaptive_diagonal
from utility import data_generation, actual_rank, get_args

if __name__ == '__main__':
    args = get_args()

    # general parameters
    n = args.n
    d = args.d
    k = args.k
    job_n = args.job_n
    cov = args.cov
    isSkewed = args.skewed
    target = args.target
    seed = args.seed

    assert target in ["linear"], f'Invalid target: {target}'

    out_file = f"results/target={target}/n={n}_d={d}_k={k}/s={seed}_cov={cov}.txt"

    X_train, y_train, X_test, y_test = data_generation(n, d, cov, seed, isSkewed=isSkewed, target=target)

    print_size = k * 2

    # Best Subset
    score, best_subset = actual(X_train, y_train, X_test, k=k, job_n=job_n, target=target)
    best_k_score = score[-1]

    import os

    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    with open(out_file, 'w') as f:
        f.write('Best Subset\n')
        for subset_size in range(1, k + 1):
            f.write(f"\tsize {subset_size}: {best_subset[subset_size-1]}\n")
        f.write('\n')

    # Diagonal
    Diag_best = diagonal(X_train, y_train, X_test, y_test, target=target)
    with open(out_file, 'a') as f:
        f.write('Diagonal Best Subset\n')
        f.write(f"\ttop {print_size}: {Diag_best[:print_size]}\n\n")

    # Adaptive Diagonal
    adaptive_Diag_best_k = adaptive_diagonal(X_train, y_train, X_test, y_test, k=k, target=target)
    with open(out_file, 'a') as f:
        f.write('Adaptive Diagonal Best Subset\n')
        f.write(f"\ttop {k}: {adaptive_Diag_best_k}\n\n")


    # First-order method
    FO_best = first_order(X_train, y_train, X_test, y_test, target=target)
    with open(out_file, 'a') as f:
        f.write('First-order Best Subset\n')
        f.write(f"\ttop {print_size}: {FO_best[:print_size]}\n\n")

    # Adaptive First-order method
    adaptive_FO_best_k = adaptive_first_order(X_train, y_train, X_test, y_test, k=k, target=target)
    with open(out_file, 'a') as f:
        f.write('Adaptive First-order Best Subset\n')
        f.write(f"\ttop {k}: {adaptive_FO_best_k}\n\n")

    with open(out_file, 'a') as f:
        f.write('Diagonal Best Subset v.s. Best Subset (size=k)\n')
        rank = actual_rank(X_train, y_train, X_test, y_test, Diag_best[:k], best_k_score, target=target)
        f.write(f'\tActual rank: {rank}\n\n')

        f.write('Adaptive Diagonal Best Subset v.s. Best Subset (size=k)\n')
        rank = actual_rank(X_train, y_train, X_test, y_test, adaptive_Diag_best_k, best_k_score, target=target)
        f.write(f'\tActual rank: {rank}\n\n')


        f.write('First-order Best Subset v.s. Best Subset (size=k)\n')
        rank = actual_rank(X_train, y_train, X_test, y_test, FO_best[:k], best_k_score, target=target)
        f.write(f'\tActual rank: {rank}\n\n')

        f.write('Adaptive First-order Best Subset v.s. Best Subset (size=k)\n')
        rank = actual_rank(X_train, y_train, X_test, y_test, adaptive_FO_best_k, best_k_score, target=target)
        f.write(f'\tActual rank: {rank}\n')