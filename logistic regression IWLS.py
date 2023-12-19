import argparse
import numpy as np
from sklearn.linear_model import LogisticRegression
from itertools import combinations
from scipy import stats

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
test_point_index = 0

# generate data
mean_n = np.array([-1, 0])
mean_p = np.array([1, 0])
cov = np.eye(2) * cov_str  
x_n = np.random.multivariate_normal(mean_n, cov, n)
x_p = np.random.multivariate_normal(mean_p, cov, n)

y_n = np.zeros(n) # 0 labels
y_p = np.ones(n)  # 1 labels

X_train = np.vstack((x_n, x_p))
y_train = np.hstack((y_n, y_p))

X_test = np.random.multivariate_normal(mean_n, cov)
X_test = X_test.reshape(-1, len(X_test))

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
def brute_force_removal(original_logistic_classifier, X_train, y_train, X_test, fixed_test_point_index=0, k=10, mode=mode):
	# Initialize variables to keep track of the best subset and loss difference for parameter changes
	best_subset_fix_test = np.full((k), None)
	best_reduced_Z_fix_test = np.full((k), None)

	## Fixed test point
	fixed_test_point = np.hstack((1, X_test[fixed_test_point_index]))
	
	if mode == "linear":
		original_score = np.dot(np.hstack((original_logistic_classifier.intercept_, original_logistic_classifier.coef_[0])), fixed_test_point)
	elif mode == "probability":
		original_score = original_logistic_classifier.predict_proba(fixed_test_point[1:].reshape(1, -1))[0][1]
	
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
				reduced_score = np.dot(np.hstack((reduced_logistic_classifier.intercept_, reduced_logistic_classifier.coef_[0])), fixed_test_point)
			elif mode == "probability":
				reduced_score = reduced_logistic_classifier.predict_proba(fixed_test_point[1:].reshape(1, -1))[0][1]

			# Calculate the difference in predicted probabilities
			score_difference = reduced_score - original_score

			# Update if the current subset induces the maximum change in test loss
			if score_difference > max_score_difference:
				max_score_difference = score_difference
				best_subset_fix_test[subset_size-1] = subset_to_remove

	return [best_subset_fix_test, best_reduced_Z_fix_test]
		
def calculate_influence(X, X_test, y, coef, W, test_point_index, leverage=True, mode=mode):
	n_samples = X.shape[0]
	influences = np.zeros(n_samples)

	# Fixed test point with intercept feature
	fixed_test_point = X_test[test_point_index]
	N = np.dot(W * X.T, X)
	N_inv = np.linalg.inv(N)
	r = W * (np.dot(X, coef) - y)
	
	if mode == "linear":
		influences = np.dot(np.dot(fixed_test_point, N_inv), X.T * r)
	elif mode == "probability":
		sigma = sigmoid(np.dot(fixed_test_point, coef))
		phi = (1 - sigma) * sigma * fixed_test_point
		influences = np.dot(np.dot(phi, N_inv), X.T * r)
		
	if leverage:
		for i in range(n_samples):
			# Calculate the influence using the provided formula
			influences[i] = influences[i] / (1 - W[i] * np.dot(np.dot(X[i], N_inv), X[i]))

	return influences

parameter = brute_force_removal(logistic_classifier, X_train, y_train, X_test, test_point_index, k)

best_subset = parameter[0][-1]

# print ground truth
with open(out_file, 'w') as f:
	f.write('Best Subset\n')
	for subset_size in range(1, k + 1):
		f.write(f"\tsize {subset_size}: {parameter[0][subset_size-1]}\n")

# Create the IWLS logistic regression model and fit it
# TODO: this might be avoided since all we need is a converged probability to construct W
coefficients, p = logistic_regression_IWLS(X_train, y_train)
W = p * (1 - p)
X_train_bar = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
X_test_bar = np.hstack((np.ones((X_test.shape[0], 1)), X_test))
y = np.dot(X_train_bar, coefficients) + (y_train - p) / W

# Calculate influences
influences = calculate_influence(X_train_bar, X_test_bar, y, coefficients, W, test_point_index)

print_size = k * 2

top_indices = np.argsort(influences)[-(print_size):][::-1]
appx_best_subset = top_indices[:k]

with open(out_file, 'a') as f:
	f.write('Approximated Best Subset\n')
	f.write(f"\ttop {print_size}: {top_indices}\n")

# Calculate adaptive influences
coef = coefficients.copy()
X_train_bar = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
X_test_bar = np.hstack((np.ones((X_test.shape[0], 1)), X_test))
X_train_bar_with_index = np.hstack((X_train_bar, np.arange(X_train_bar.shape[0]).reshape(-1, 1)))
y_copy = y_train.copy()
p_copy = p.copy()
appx_adaptive_best_subset = np.zeros(k)

with open(out_file, 'a') as f:
	f.write('Approximated Adaptive Best Subset\n')
 
for i in range(k):
	W = p_copy * (1 - p_copy)
	X = X_train_bar_with_index[:, :-1] # without index
	y = np.dot(X, coef) + (y_copy - p_copy) / W
	# Calculate influences
	influences = calculate_influence(X, X_test_bar, y, coef, W, test_point_index)
	top_indices = np.argsort(influences)[-(print_size):][::-1]
	
	actual_top_indices = X_train_bar_with_index[:, -1][top_indices].astype(int)
	appx_adaptive_best_subset[i] = actual_top_indices[0]
	with open(out_file, 'a') as f:
		f.write(f"\titer {i+1}:\ttop {print_size}: {actual_top_indices}\n")
	
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
  
# Result
# with open(out_file, 'a') as f:
# 	f.write('Approximated Best Subset v.s. Best Subset\n')
# 	f.write(f'\tK: {stats.kendalltau(best_subset, appx_best_subset).statistic:.3f} | P: {stats.pearsonr(best_subset, appx_best_subset).statistic:.3f}\n')
 
# 	f.write('Approximated Adaptive Best Subset v.s. Best Subset\n')
# 	f.write(f'\tK: {stats.kendalltau(best_subset, appx_adaptive_best_subset).statistic:.3f} | P: {stats.pearsonr(best_subset, appx_adaptive_best_subset).statistic:.3f}\n')