import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from itertools import combinations


def actual_effect(X_train, y_train, x_test, y_test, subset_to_remove, original_score, target="probability"):
    reduced_X_train = np.delete(X_train, subset_to_remove, axis=0)
    reduced_y_train = np.delete(y_train, subset_to_remove, axis=0)
    reduced_lr = LogisticRegression(penalty=None).fit(reduced_X_train, reduced_y_train)

    if target == "probability":
        reduced_score = reduced_lr.predict_proba(x_test.reshape(1, -1))[0][1]
    elif target == "train_loss":
        reduced_score = log_loss(reduced_y_train, reduced_lr.predict_proba(reduced_X_train), labels=[0, 1])
    elif target == "test_loss":
        reduced_score = log_loss([y_test], reduced_lr.predict_proba(x_test.reshape(1, -1)), labels=[0, 1])

    score_difference = reduced_score - original_score

    return score_difference

def actual(X_train, y_train, x_test, y_test, k=10, job_n=50, target="probability"):
    original_lr = LogisticRegression(penalty=None).fit(X_train, y_train)
 
    best_subset = np.full((k), None)
    score = []
    
    if target == "probability":
        original_score = original_lr.predict_proba(x_test.reshape(1, -1))[0][1] # The predicted probability of the positive class
    elif target == "train_loss":
        original_score = log_loss(y_train, original_lr.predict_proba(X_train), labels=[0, 1])
    elif target == "test_loss":
        original_score = log_loss([y_test], original_lr.predict_proba(x_test.reshape(1, -1)), labels=[0, 1])

    for subset_size in range(1, k + 1):
        # Generate all combinations of subsets of the current size
        subset_combinations = combinations(range(X_train.shape[0]), subset_size)
        combinations_list = list(combinations(range(X_train.shape[0]), subset_size))
        
        best_k_score = Parallel(n_jobs=job_n)(delayed(actual_effect)(X_train, y_train, x_test, y_test, subset_to_remove, original_score, target) for subset_to_remove in subset_combinations)
        
        sort_subset_combinations = np.array(combinations_list)[np.argsort(best_k_score)[::-1]]
        best_subset[subset_size - 1] = sort_subset_combinations[0]
        score.append(best_k_score) # TODO: Flatten this if we want to get <= k

    return [score, best_subset]