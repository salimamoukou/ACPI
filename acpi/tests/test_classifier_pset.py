import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import fetch_california_housing, fetch_covtype, fetch_rcv1, load_iris
from sklearn.model_selection import train_test_split
from acpi import ACPI
from acpi.utils import mean_score, quantile_score, compute_coverage_classification


def test_prediction_set_qrf():
    # Parameters
    random_state = 2022
    valid_ratio = 0.5
    alpha = 0.1
    eps = 0.1

    # SIMULATED DATA
    # centers = [(0, 3.5), (-2, 0), (2, 0)]
    # covs = [np.eye(2), np.eye(2)*2, np.diag([5, 1])]
    # x_min, x_max, y_min, y_max, step = -6, 8, -6, 8, 0.1
    # n_samples = 1000
    # n_classes = 3
    # np.random.seed(42)
    # X = np.vstack([
    #     np.random.multivariate_normal(center, cov, n_samples)
    #     for center, cov in zip(centers, covs)
    # ])
    # y = np.hstack([np.full(n_samples, i) for i in range(0, n_classes)])

    # UCI datasets
    sklearn_data = load_iris()
    X, y = sklearn_data.data, sklearn_data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    x_train, x_test = X_train.astype(np.double), X_test.astype(np.double)
    y_train, y_test = np.asarray(y_train).astype(np.int).reshape(-1), np.asarray(y_test).astype(np.int).reshape(-1)

    in_shape = x_train.shape[1]
    n_train = x_train.shape[0]
    idx = np.random.permutation(n_train)
    n_half = int(np.floor(n_train * valid_ratio))
    idx_train, idx_cal = idx[:n_half], idx[n_half:]

    # Fit classifier
    clf = RandomForestClassifier(random_state=random_state)
    clf.fit(x_train[idx_train], y_train[idx_train])

    # Fit ACPI
    acp = ACPI(model_cali=clf, n_estimators=300, mtry=0, max_depth=20, min_node_size=10,
               seed=random_state, estimator='clf')

    level = 1 - alpha
    acp.fit_calibration(x_train[idx_cal], y_train[idx_cal], quantile=level, only_qrf=True, n_iter_qrf=50)
    y_pred_set_qrf = acp.predict_qrf_pi(x_test)

    coverage = compute_coverage_classification(y_pred_set_qrf, y_test)
    assert coverage >= 1 - alpha - eps


def test_prediction_set_all():
    # Parameters
    random_state = 2022
    valid_ratio = 0.5
    alpha = 0.1
    eps = 0.1

    # SIMULATED DATA
    # centers = [(0, 3.5), (-2, 0), (2, 0)]
    # covs = [np.eye(2), np.eye(2)*2, np.diag([5, 1])]
    # x_min, x_max, y_min, y_max, step = -6, 8, -6, 8, 0.1
    # n_samples = 1000
    # n_classes = 3
    # np.random.seed(42)
    # X = np.vstack([
    #     np.random.multivariate_normal(center, cov, n_samples)
    #     for center, cov in zip(centers, covs)
    # ])
    # y = np.hstack([np.full(n_samples, i) for i in range(0, n_classes)])

    # UCI datasets
    sklearn_data = load_iris()
    X, y = sklearn_data.data, sklearn_data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    x_train, x_test = X_train.astype(np.double), X_test.astype(np.double)
    y_train, y_test = np.asarray(y_train).astype(int).reshape(-1), np.asarray(y_test).astype(int).reshape(-1)

    in_shape = x_train.shape[1]
    n_train = x_train.shape[0]
    idx = np.random.permutation(n_train)
    n_half = int(np.floor(n_train * valid_ratio))
    idx_train, idx_cal = idx[:n_half], idx[n_half:]

    # Fit classifier
    clf = RandomForestClassifier(random_state=random_state)
    clf.fit(x_train[idx_train], y_train[idx_train])

    # Fit ACPI
    acp = ACPI(model_cali=clf, n_estimators=300, mtry=0, max_depth=20, min_node_size=10,
               seed=random_state, estimator='clf')

    level = 1 - alpha
    acp.fit_calibration(x_train[idx_cal], y_train[idx_cal], quantile=level, training_conditional=True,
                    training_conditional_one=True,
                    bygroup=True,
                    only_qrf=False,
                    n_iter_qrf=50)
    y_pred_set_qrf = acp.predict_pi(x_test, method='qrf')
    y_pred_set_lcprf = acp.predict_pi(x_test, method='lcp-rf')
    y_pred_set_lcprfgroup = acp.predict_pi(x_test, method='lcp-rf-group')

    coverage_qrf = compute_coverage_classification(y_pred_set_qrf, y_test)
    coverage_lcprf = compute_coverage_classification(y_pred_set_lcprf, y_test)
    coverage_lcprfgroup = compute_coverage_classification(y_pred_set_lcprfgroup, y_test)
    assert coverage_qrf >= 1 - alpha - eps and coverage_lcprf >= 1 - alpha - eps and coverage_lcprfgroup >= 1 - alpha - eps
