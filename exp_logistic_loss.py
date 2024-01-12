import argparse
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import copy
from itertools import combinations
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=20)
parser.add_argument('--n', type=int, default=50)
parser.add_argument('--b', type=int, default=1)
parser.add_argument('--cov_str', type=float, default=1)
parser.add_argument('--out-file', type=str, default='output.txt')
parser.add_argument('--out-file-np', type=str, default='output-np.npz')
args = parser.parse_args()

np.random.seed(args.seed)
n = args.n
cov_str = args.cov_str
b = args.b

# generate data
mean_n = np.array([-1, 0])
mean_p = np.array([1, 0])
cov = np.eye(2) * cov_str

def gen_data(mean_n, mean_p, cov, n):
    x_n = np.random.multivariate_normal(mean_n, cov, n)
    x_p = np.random.multivariate_normal(mean_p, cov, n)

    y_n = -np.ones(n)  # -1 labels
    y_p = np.ones(n)  # 1 labels

    X = np.vstack((x_n, x_p))
    y = np.hstack((y_n, y_p))
    k = len(X)
    return X, y, k

X_train, y_train, k = gen_data(mean_n, mean_p, cov, n)
X_test, y_test, _ = gen_data(mean_n, mean_p, cov, 25)


def obtain_param(X, y):
    clf = LogisticRegression().fit(X, y)
    ic = clf.intercept_[0]
    coef = clf.coef_[0]
    param = np.concatenate(([ic], coef))
    return (param, clf, ic, coef) 

comp = obtain_param(X_train, y_train)
clf = comp[1]
param = comp[0]

X_one = np.hstack([np.ones(len(X_train)).reshape(len(X_train), 1), X_train])
margin = (X_one @ comp[0]) * y_train * (-1)
margin_n = margin[:n]
margin_p = margin[n:]

def top_k_indices_n(margin_n, k):
    sorted_indices = np.argsort(margin_n)
    top_k = sorted_indices[-k:][::-1]
    return list(top_k)

def top_k_indices_p(margin_p, k):
    sorted_indices = np.argsort(margin_p)
    top_k = sorted_indices[-k:][::-1] + len(margin_p)
    return list(top_k)

def calc_loss(clf_ori, clf_new):
    diff_sum = 0
    y_pred_ori_l = clf_ori.predict_proba(X_test)
    y_pred_new_l = clf_new.predict_proba(X_test)
    for x_test_i, y_test_i, y_pred_ori, y_pred_new in zip(X_test, y_test, y_pred_ori_l, y_pred_new_l):
        y_true = y_test_i
        
        assert clf_ori.classes_[0] == -1, f'class mismatch!'
        assert clf_new.classes_[0] == -1, f'class mismatch!'
        
        def calc_log_loss(y_true, y_pred):
            y_true_alt = 1 if y_true == 1 else 0
            y_pred_alt = y_pred[1]
            return -(y_true_alt * np.log(y_pred_alt) + (1 - y_true_alt) * np.log(1 - y_pred_alt))
        
        diff_sum += abs(calc_log_loss(y_true, y_pred_ori) - calc_log_loss(y_true, y_pred_new))
    return diff_sum / len(X_test)


def get_rank(comb, ind_S):
    mask = np.ones(k, dtype=bool)
    mask[ind_S] = False
    comp_S = obtain_param(X_train[mask], y_train[mask])
    val_S = calc_loss(clf, comp_S[1])
    sorted_list = sorted(comb, reverse=True)
    return sorted_list.index(val_S) + 1

def get_all(b):
    comb = []
    S_list = []            
    
    for r in range(1, b + 1):
        for S in tqdm(combinations(range(k), r)):
            list_S = list(S)
            mask = np.ones(k, dtype=bool)
            mask[list_S] = False
            comp_S = obtain_param(X_train[mask], y_train[mask])
            comb.append(calc_loss(clf, comp_S[1]))
            # comb.append(np.linalg.norm(comp_S[0]-param))
            S_list.append(list_S)
    return comb, S_list

comb, S_list = get_all(b)
ind_n = top_k_indices_n(margin_n, 10)
ind_p = top_k_indices_p(margin_p, 10)
ind_n_true = top_k_indices_n(margin_n, b)
ind_p_true = top_k_indices_p(margin_p, b)

with open(args.out_file, 'w') as f:
    f.write(str(ind_n)+'\n')
    f.write(str(ind_p)+'\n')
    f.write(str([S_list[top_k_indices_n(comb, 10)[i]] for i in range(10)])+'\n')
    f.write(str(get_rank(comb, ind_n_true))+'\n')
    f.write(str(get_rank(comb, ind_p_true))+'\n')

