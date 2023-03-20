import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_diabetes, make_regression
from sklearn.model_selection import train_test_split
from skranger.ensemble import RangerForestRegressor
from acpi import ACPI
from acpi.utils import mean_score, quantile_score, compute_coverage


def test_predict_wreject():
    # Parameters
    random_state = 2022
    valid_ratio = 0.5
    alpha = 0.1
    eps = 0.1

    # UCI datasets
    sklearn_data = load_diabetes()
    X, y = sklearn_data.data, sklearn_data.target
    # X, y = make_regression(n_samples=1000, n_features=10, noise=0.05, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    x_train, x_test = X_train.astype(np.double), X_test.astype(np.double)
    y_train, y_test = np.asarray(y_train).astype(np.double).reshape(-1), np.asarray(y_test).astype(np.double).reshape(
        -1)

    in_shape = x_train.shape[1]
    n_train = x_train.shape[0]
    idx = np.random.permutation(n_train)
    n_half = int(np.floor(n_train * valid_ratio))
    idx_train, idx_cal = idx[:n_half], idx[n_half:]

    # Fit classifier
    reg = RandomForestRegressor(random_state=random_state)
    reg.fit(x_train[idx_train], y_train[idx_train])

    # Fit ACPI
    acp = ACPI(model_cali=reg, n_estimators=300, mtry=0, max_depth=20, min_node_size=10, seed=random_state,
               estimator='ref')

    level = 1 - alpha
    acp.fit_calibration(x_train[idx_cal], y_train[idx_cal], quantile=level, only_qrf=True, n_iter_qrf=10)

    # Fit predict reject option
    residual_reg = RangerForestRegressor()
    acp.fit_reject(residual_reg, split=False)

    level_fdp = 0.2
    delta_fdp = 0.1
    v_star = 50
    rej_indices = acp.predict_wreject(x_test, v_star=v_star, method='marginal', level=level_fdp, delta=delta_fdp)

    fdp = np.mean(y_test[rej_indices] <= v_star)
    tol = np.sum(y_test >= v_star)
    found = np.sum(y_test[rej_indices] >= v_star)
    power = tol / found
    assert fdp <= level
