import numpy as np
from sklearn.linear_model import LogisticRegression

def margin(X_train, y_train):
    # Fit logistic regression model
    lr = LogisticRegression(penalty=None).fit(X_train, y_train)
    param = np.concatenate(([lr.intercept_[0]], lr.coef_[0]))

    X_train_bar = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
    margin = (X_train_bar @ param) * (1 - 2 * y_train)

    # Sort margins and get corresponding indices
    sorted_indices = np.argsort(margin)[::-1]

    # Separate positive and negative margins based on sorted indices
    ind_p = sorted_indices[y_train[sorted_indices] == 1]
    ind_n = sorted_indices[y_train[sorted_indices] == 0]

    return ind_n, ind_p