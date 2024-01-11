import numpy as np
from sklearn.linear_model import LogisticRegression

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