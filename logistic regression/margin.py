import argparse
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import copy
from itertools import combinations

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=20)
parser.add_argument('--n', type=int, default=50)
parser.add_argument('--b', type=int, default=2)
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
x_n = np.random.multivariate_normal(mean_n, cov, n)
x_p = np.random.multivariate_normal(mean_p, cov, n)

y_n = -np.ones(n)  # -1 labels
y_p = np.ones(n)  # 1 labels

X = np.vstack((x_n, x_p))
y = np.hstack((y_n, y_p))
# X = np.zeros((2 * n, 2))
# for i in range(1, n):
#     X[i] = np.asarray([-1, 0])
#     X[i+n] = np.asarray([1, 0])
# X[0] = np.asarray([-1, h])
# X[n] = np.asarray([1, h]) 
# y_n = -np.ones(n)  
# y_p = np.ones(n)
# y = np.concatenate([y_n, y_p])

# X = np.asarray([[-1, 100], [-1, 0], [1, 0], [1, 100]])
# y = np.asarray([-1, -1, 1, 1])
k = len(X)

x_test = np.random.multivariate_normal(mean_n, cov)
x_test = x_test.reshape(-1, len(x_test))

def obtain_param(X, y):
	clf = LogisticRegression().fit(X, y)
	ic = clf.intercept_[0]
	coef = clf.coef_[0]
	param = np.concatenate(([ic], coef))
	return (param, clf, ic, coef) 

comp = obtain_param(X, y)
param = comp[0]

X_one = np.hstack([np.ones(len(X)).reshape(len(X), 1), X])
margin = (X_one @ comp[0]) * y * (-1)
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

def get_rank(comb, ind_S):
	mask = np.ones(k, dtype=bool)
	mask[ind_S] = False
	comp_S = obtain_param(X[mask], y[mask])
	val_S = np.linalg.norm(comp_S[0]-param)
	sorted_list = sorted(comb, reverse=True)
	return sorted_list.index(val_S) + 1

def get_all(b):
	comb = []
	S_list = []
	for r in range(1, b + 1):
		for S in combinations(range(k), r):
			list_S = list(S)
			mask = np.ones(k, dtype=bool)
			mask[list_S] = False
			comp_S = obtain_param(X[mask], y[mask])
			comb.append(np.linalg.norm(comp_S[0]-param))
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