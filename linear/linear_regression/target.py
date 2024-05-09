from sklearn.linear_model import LinearRegression

def target_value(X_train, y_train, X_test, target="linear"):
    lr = LinearRegression().fit(X_train, y_train)

    if target == "linear":
        value = lr.predict(X_test.reshape(1, -1))

    return value

def target_phi(X_train, y_train, X_test, y_test, target="linear"):
    if target == "linear":
        phi = X_test

    return phi

def target_influence(phi, param_influence, target="linear"):
    if target == "linear":
        influence = (phi @ param_influence).reshape(-1)

    return influence