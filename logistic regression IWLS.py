import argparse
import numpy as np
from sklearn.linear_model import LogisticRegression
from itertools import combinations, permutations
from scipy import stats
from sklearn.metrics import ndcg_score
from rbo import RankingSimilarity

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=20)
parser.add_argument('--n', type=int, default=50)
parser.add_argument('--k', type=int, default=2)
parser.add_argument('--cov_str', type=float, default=1)
parser.add_argument('--mode', type=str, default='probability')
args = parser.parse_args()

# general parameters
n = args.n
k = args.k
cov_str = args.cov_str
mode = args.mode
seed = args.seed
 
np.random.seed(seed)

out_file = f"results/mode={mode}_seed={seed}_n={n}_k={k}_cov={cov_str}.txt"

# generate data
mean_n = np.array([-1, 0])
mean_p = np.array([1, 0])
cov = np.eye(2) * cov_str  
x_n = np.random.multivariate_normal(mean_n, cov, int(n/2))
x_p = np.random.multivariate_normal(mean_p, cov, int(n/2))

y_n = np.zeros(int(n/2)) # 0 labels
y_p = np.ones(int(n/2))  # 1 labels

X_train = np.vstack((x_n, x_p))
y_train = np.hstack((y_n, y_p))

x_test = np.random.multivariate_normal(mean_n, cov)
# x_test = x_test.reshape(-1, len(x_test))

# Create a Logistic Regression classifier
logistic_classifier = LogisticRegression(penalty=None)
logistic_classifier.fit(X_train, y_train)

def sigmoid(z):
	return 1 / (1 + np.exp(-z))

def logistic_regression_IWLS(X, y, max_iters=500, tolerance=1e-6):
	n_samples, n_features = X.shape
	X = np.hstack((np.ones((n_samples, 1)), X))  # Add a column of ones for the intercept

	# Initialize coefficients, including the intercept
	coef = np.zeros(n_features + 1)
	p = np.zeros(n_samples)
	prev_coef = coef.copy()
	
	
	for _ in range(max_iters):
		p = sigmoid(np.dot(X, coef))
		W = p * (1 - p)  # Calculate weights based on the current predictions

		# Perform a weighted least squares update
		X_weighted = X.T * W
		Hessian = np.dot(X_weighted, X)
		gradient = np.dot(X.T, y - p)
		coef += np.linalg.solve(Hessian, gradient)

		# Check for convergence
		if np.allclose(coef, prev_coef, atol=tolerance):
			break

		prev_coef = coef.copy()
	
	return coef, p

# TODO: Store the best 3 subsets for each size
def brute_force_removal(original_logistic_classifier, X_train, y_train, x_test, k=10, mode=mode):
	# Initialize variables to keep track of the best subset and loss difference for parameter changes
	best_subset_fix_test = np.full((k), None)
	best_reduced_Z_fix_test = np.full((k), None)

	## Fixed test point
	x_test = np.hstack((1, x_test))
	
	if mode == "linear":
		original_score = np.dot(np.hstack((original_logistic_classifier.intercept_, original_logistic_classifier.coef_[0])), x_test)
	elif mode == "probability":
		original_score = original_logistic_classifier.predict_proba(x_test[1:].reshape(1, -1))[0][1]
	
	# Loop over different subset sizes from 1 to k
	for subset_size in range(1, k + 1):
		# Generate all combinations of subsets of the current size
		subset_combinations = combinations(range(X_train.shape[0]), subset_size)

		max_score_difference = -float("inf")

		for subset_to_remove in subset_combinations:
			# Create a new training set without the selected data points
			reduced_X_train = np.delete(X_train, subset_to_remove, axis=0)
			reduced_y_train = np.delete(y_train, subset_to_remove, axis=0)

			# Train a Logistic Regression classifier on the reduced training set
			reduced_logistic_classifier = LogisticRegression(penalty=None)

			reduced_logistic_classifier.fit(reduced_X_train, reduced_y_train)

			# Make inference
			if mode == "linear":
				reduced_score = np.dot(np.hstack((reduced_logistic_classifier.intercept_, reduced_logistic_classifier.coef_[0])), x_test)
			elif mode == "probability":
				reduced_score = reduced_logistic_classifier.predict_proba(x_test[1:].reshape(1, -1))[0][1]

			# Calculate the difference in predicted probabilities
			score_difference = reduced_score - original_score

			# Update if the current subset induces the maximum change in test loss
			if score_difference > max_score_difference:
				max_score_difference = score_difference
				best_subset_fix_test[subset_size-1] = subset_to_remove

	return [best_subset_fix_test, best_reduced_Z_fix_test]
		
def calculate_influence(X, x_test, y, coef, W, leverage=True, mode=mode):
	n_samples = X.shape[0]
	influences = np.zeros(n_samples)
 
	N = np.dot(W * X.T, X)
	N_inv = np.linalg.inv(N)
	r = W * (np.dot(X, coef) - y)
	
	if mode == "linear":
		influences = np.dot(np.dot(x_test, N_inv), X.T * r)
	elif mode == "probability":
		sigma = sigmoid(np.dot(x_test, coef))
		phi = (1 - sigma) * sigma * x_test
		influences = np.dot(np.dot(phi, N_inv), X.T * r)
		
	if leverage:
		for i in range(n_samples):
			# Calculate the influence using the provided formula
			influences[i] = influences[i] / (1 - W[i] * np.dot(np.dot(X[i], N_inv), X[i]))

	return influences

parameter = brute_force_removal(logistic_classifier, X_train, y_train, x_test, k)

best_k_subset = parameter[0][-1]

# print ground truth
with open(out_file, 'w') as f:
	f.write('Best Subset\n')
	for subset_size in range(1, k + 1):
		f.write(f"\tsize {subset_size}: {parameter[0][subset_size-1]}\n")
	f.write('\n')

# Create the IWLS logistic regression model and fit it
# TODO: this might be avoided since all we need is a converged probability to construct W
coefficients, p = logistic_regression_IWLS(X_train, y_train)
W = p * (1 - p)
X_train_bar = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
x_test_bar = np.hstack((1, x_test))
y = np.dot(X_train_bar, coefficients) + (y_train - p) / W

# Calculate influences
influences = calculate_influence(X_train_bar, x_test_bar, y, coefficients, W)

print_size = k * 2

IWLS_best = np.argsort(influences)[-n:][::-1]

with open(out_file, 'a') as f:
	f.write('IWLS Best Subset\n')
	f.write(f"\ttop {print_size}: {IWLS_best[:print_size]}\n\n")

# Calculate adaptive influences
coef = coefficients.copy()
X_train_bar = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
x_test_bar = np.hstack((1, x_test))
X_train_bar_with_index = np.hstack((X_train_bar, np.arange(X_train_bar.shape[0]).reshape(-1, 1)))
y_copy = y_train.copy()
p_copy = p.copy()
adaptive_IWLS_best_k = np.zeros(k, dtype=int)
 
for i in range(k):
	W = p_copy * (1 - p_copy)
	X = X_train_bar_with_index[:, :-1] # without index
	y = np.dot(X, coef) + (y_copy - p_copy) / W
	# Calculate influences
	influences = calculate_influence(X, x_test_bar, y, coef, W)
	top_indices = np.argsort(influences)[-(print_size):][::-1]
	
	actual_top_indices = X_train_bar_with_index[:, -1][top_indices].astype(int)
	adaptive_IWLS_best_k[i] = actual_top_indices[0]
	# Remove the most influential data points
	X = np.delete(X, top_indices[0], axis=0)
	X_train_bar_with_index = np.delete(X_train_bar_with_index, top_indices[0], axis=0)
	y_copy = np.delete(y_copy, top_indices[0], axis=0)
	W = np.delete(W, top_indices[0], axis=0)
	p_copy = np.delete(p_copy, top_indices[0], axis=0)
	
	if i > 0:
		# Perform a weighted least squares update
		X_weighted = X.T * W
		Hessian = np.dot(X_weighted, X)
		gradient = np.dot(X.T, y_copy - p_copy)
		coef += np.linalg.solve(Hessian, gradient)
		p_copy = sigmoid(np.dot(X, coef))
  
with open(out_file, 'a') as f:
    f.write('Adaptive IWLS Best Subset\n')
    f.write(f"\ttop {k}: {adaptive_IWLS_best_k}\n\n")
  
# Margin-based approach

def obtain_param(X, y):
	clf = LogisticRegression().fit(X, y)
	ic = clf.intercept_[0]
	coef = clf.coef_[0]
	param = np.concatenate(([ic], coef))
	return (param, clf, ic, coef) 

X = np.vstack((x_n, x_p))
y = np.hstack((-np.ones(int(n/2)), y_p))

comp = obtain_param(X, y)

X_one = np.hstack([np.ones(len(X)).reshape(len(X), 1), X])
margin = (X_one @ comp[0]) * y * (-1)

def top_k_indices_n(margin_n, k):
	sorted_indices = np.argsort(margin_n)
	top_k = sorted_indices[-k:][::-1]
	return top_k

def top_k_indices_p(margin_p, k):
	sorted_indices = np.argsort(margin_p)
	top_k = sorted_indices[-k:][::-1] + len(margin_p)
	return top_k

margin_n = margin[:int(n/2)]
margin_p = margin[int(n/2):]

ind_n = top_k_indices_n(margin_n, int(n/2))
ind_p = top_k_indices_p(margin_p, int(n/2))
 
with open(out_file, 'a') as f:
	f.write('Margin-based Best Subset\n')
	f.write(f"\tpositive group:\ttop {print_size}: {ind_p[:print_size]}\n")
	f.write(f"\tnegative group:\ttop {print_size}: {ind_n[:print_size]}\n\n")
 
# Calculate the maximum possible NDCG and rbo score by shuffling the order
def max_score(set_list, fixed_order_list):
    # Generate all possible permutations of the set_list
    all_permutations = permutations(set_list)
    max_ndcg_score = 0
    max_rbo_score = 0
    
    # Iterate over all permutations and calculate NDCG and rbo score for each
    for permuted_set in all_permutations:
        ndcg = ndcg_score([fixed_order_list], [list(permuted_set)])
        rbo = RankingSimilarity(fixed_order_list, list(permuted_set)).rbo()
        if ndcg > max_ndcg_score:
            max_ndcg_score = ndcg
        if rbo > max_rbo_score:
            max_rbo_score = rbo
    return max_ndcg_score, max_rbo_score

# Calculate the average possible NDCG and rbo score by shuffling the order
def average_score(set_list, fixed_order_list):
    # Generate all possible permutations of the set_list
    all_permutations = permutations(set_list)
    length = 0
    avg_ndcg_score = 0
    avg_rbo_score = 0
    
    # Iterate over all permutations and calculate NDCG and rbo score for each
    for permuted_set in all_permutations:
        avg_ndcg_score += ndcg_score([fixed_order_list], [list(permuted_set)])
        avg_rbo_score += RankingSimilarity(fixed_order_list, list(permuted_set)).rbo()
        length += 1
    
    return avg_ndcg_score/length, avg_rbo_score/length

# Result
with open(out_file, 'a') as f:
	f.write('Margin-based v.s. IWLS-based\n')
	f.write(f'-size=n/2\n')
	f.write(f'\tP Group\tK: {stats.kendalltau(ind_p[:int(n/2)], IWLS_best[:int(n/2)]).statistic:.5f} | P: {stats.pearsonr(ind_p[:int(n/2)], IWLS_best[:int(n/2)]).statistic:.5f}\n')
	f.write(f'\tN Group\tK: {stats.kendalltau(ind_n[:int(n/2)], IWLS_best[:int(n/2)]).statistic:.5f} | P: {stats.pearsonr(ind_n[:int(n/2)], IWLS_best[:int(n/2)]).statistic:.5f}\n')
	f.write(f'-size=k\n')
	f.write(f'\tP Group\tK: {stats.kendalltau(ind_p[:k], IWLS_best[:k]).statistic:.5f} | P: {stats.pearsonr(ind_p[:k], IWLS_best[:k]).statistic:.5f}\n')
	f.write(f'\tN Group\tK: {stats.kendalltau(ind_n[:k], IWLS_best[:k]).statistic:.5f} | P: {stats.pearsonr(ind_n[:k], IWLS_best[:k]).statistic:.5f}\n')
	f.write(f'-size=2k\n')
	f.write(f'\tP Group\tK: {stats.kendalltau(ind_p[:2*k], IWLS_best[:2*k]).statistic:.5f} | P: {stats.pearsonr(ind_p[:2*k], IWLS_best[:2*k]).statistic:.5f}\n')
	f.write(f'\tN Group\tK: {stats.kendalltau(ind_n[:2*k], IWLS_best[:2*k]).statistic:.5f} | P: {stats.pearsonr(ind_n[:2*k], IWLS_best[:2*k]).statistic:.5f}\n\n')
  
	f.write('Approximated Best Subset v.s. Best Subset (size=k)\n')
	ndcg, rbo = max_score(best_k_subset, IWLS_best[:k])
	f.write(f'\tmax NDCG: {ndcg:.5f} | max rbo: {rbo:.5f}\n')
	ndcg, rbo = average_score(best_k_subset, IWLS_best[:k])
	f.write(f'\tavg NDCG: {ndcg:.5f} | avg rbo: {rbo:.5f}\n\n')
 
	f.write('Approximated Adaptive Best Subset v.s. Best Subset (size=k)\n')
	ndcg, rbo = max_score(best_k_subset, adaptive_IWLS_best_k)
	f.write(f'\tmax NDCG: {ndcg:.5f} | max rbo: {rbo:.5f}\n')
	ndcg, rbo = average_score(best_k_subset, adaptive_IWLS_best_k)
	f.write(f'\tavg NDCG: {ndcg:.5f} | avg rbo: {rbo:.5f}\n\n')

	f.write('Margin-based Best Subset v.s. Best Subset (size=k)\n')
	f.write(f'P Group\n')
	ndcg, rbo = max_score(best_k_subset, ind_p[:k])
	f.write(f'\tmax NDCG: {ndcg:.5f} | max rbo: {rbo:.5f}\n')
	ndcg, rbo = average_score(best_k_subset, ind_p[:k])
	f.write(f'\tavg NDCG: {ndcg:.5f} | avg rbo: {rbo:.5f}\n')
 
	f.write(f'N Group\n')
	ndcg, rbo = max_score(best_k_subset, ind_n[:k])
	f.write(f'\tmax NDCG: {ndcg:.5f} | max rbo: {rbo:.5f}\n')
	ndcg, rbo = average_score(best_k_subset, ind_n[:k])
	f.write(f'\tavg NDCG: {ndcg:.5f} | avg rbo: {rbo:.5f}\n')
	