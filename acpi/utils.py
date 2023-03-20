import sys
import numpy as np
import pickle


def safe_isinstance(obj, class_path_str):
    """
    Acts as a safe version of isinstance without having to explicitly
    import packages which may not exist in the users environment.

    Checks if obj is an instance of type specified by class_path_str.

    Parameters
    ----------
    obj: Any
        Some object you want to test against
    class_path_str: str or list
        A string or list of strings specifying full class paths
        Example: `sklearn.ensemble.RandomForestRegressor`

    Returns
    --------
    bool: True if isinstance is true and the package exists, False otherwise
    """
    if isinstance(class_path_str, str):
        class_path_strs = [class_path_str]
    elif isinstance(class_path_str, list) or isinstance(class_path_str, tuple):
        class_path_strs = class_path_str
    else:
        class_path_strs = ['']

    # try each module path in order
    for class_path_str in class_path_strs:
        if "." not in class_path_str:
            raise ValueError("class_path_str must be a string or list of strings specifying a full \
                module path to a class. Eg, 'sklearn.ensemble.RandomForestRegressor'")

        # Splits on last occurence of "."
        module_name, class_name = class_path_str.rsplit(".", 1)

        # here we don't check further if the model is not imported, since we shouldn't have
        # an object of that types passed to us if the model the type is from has never been
        # imported. (and we don't want to import lots of new modules for no reason)
        if module_name not in sys.modules:
            continue

        module = sys.modules[module_name]

        # Get class
        _class = getattr(module, class_name, None)

        if _class is None:
            continue

        if isinstance(obj, _class):
            return True

    return False


def weighted_percentile(a, q, weights=None, sorter=None):
    """Returns the weighted percentile of a at q given weights.

    Parameters
    ----------
    a: array-like, shape=(n_samples,)
        samples at which the quantile.
    q: int
        quantile.
    weights: array-like, shape=(n_samples,)
        weights[i] is the weight given to point a[i] while computing the
        quantile. If weights[i] is zero, a[i] is simply ignored during the
        percentile computation.
    sorter: array-like, shape=(n_samples,)
        If provided, assume that a[sorter] is sorted.
    Returns
    -------
    percentile: float
        Weighted percentile of a at q.
    References
    ----------
    1. https://en.wikipedia.org/wiki/Percentile#The_Weighted_Percentile_method
    Notes
    -----
    Note that weighted_percentile(a, q) is not equivalent to
    np.percentile(a, q). This is because in np.percentile
    sorted(a)[i] is assumed to be at quantile 0.0, while here we assume
    sorted(a)[i] is given a weight of 1.0 / len(a), hence it is at the
    1.0 / len(a)th quantile.
    """
    if weights is None:
        weights = np.ones_like(a)
    if q > 100 or q < 0:
        raise ValueError("q should be in-between 0 and 100, "
                         "got %d" % q)

    a = np.asarray(a, dtype=np.float32)
    weights = np.asarray(weights, dtype=np.float32)
    if len(a) != len(weights):
        raise ValueError("a and weights should have the same length.")

    if sorter is not None:
        a = a[sorter]
        weights = weights[sorter]

    nz = weights != 0
    a = a[nz]
    weights = weights[nz]

    if sorter is None:
        sorted_indices = np.argsort(a)
        sorted_a = a[sorted_indices]
        sorted_weights = weights[sorted_indices]
    else:
        sorted_a = a
        sorted_weights = weights

    # Step 1
    sorted_cum_weights = np.cumsum(sorted_weights)
    total = sorted_cum_weights[-1]

    # Step 2
    partial_sum = 100.0 / total * (sorted_cum_weights - sorted_weights / 2.0)
    start = np.searchsorted(partial_sum, q) - 1
    if start == len(sorted_cum_weights) - 1:
        return sorted_a[-1]
    if start == -1:
        return sorted_a[0]

    # Step 3.
    fraction = (q - partial_sum[start]) / (partial_sum[start + 1] - partial_sum[start])
    return sorted_a[start] + fraction * (sorted_a[start + 1] - sorted_a[start])


def quantile_score(prediction, y):
    """Nonconformity score for quantile regression estimates.

    score(predictions, y) = max(predictions[:,0] - y, y - predictions[:, 1])

    Parameters
    ----------
    prediction : numpy.array
        model's predictions
    y : numpy.array
        true labels
    Returns
    -------
    numpy.array
        It returns the nonconformity scores
    """
    y_lower = prediction[:, 0]
    y_upper = prediction[:, 1]
    score_low = y_lower - np.squeeze(y)
    score_high = np.squeeze(y) - y_upper
    score = np.maximum(score_high, score_low)
    return score


def mean_score(prediction, y):
    """Nonconformity score for mean regression estimates.

    score(predictions, y) = abs(predictions - y).

    Parameters
    ----------
    prediction : numpy.array
        model's predictions
    y : numpy.array
        true labels
    Returns
    -------
    numpy.array
        It returns the nonconformity scores
    """
    return np.abs(prediction - y)


def classifier_score(pred_proba, y_label):
    """Transforms predict probabilities into scores.

    score(pred_proba, y)_i = 1 - pred_proba_{y_i}.

    Parameters
    ----------
    pred_proba : numpy.arrau
        model's predict_proba
    y_label :
        true label
    Returns
    -------
    numpy.array
        It returns the nonconformity scores
    """
    return 1 - np.take_along_axis(pred_proba, y_label.reshape(-1, 1), axis=1).reshape(-1)


def compute_coverage(y_test, y_lower, y_upper):
    """Compute average coverage and length of prediction intervals.

    Parameters
    ----------
    y_test : numpy array, true labels (n)
    y_lower : numpy array, estimated lower bound for the labels (n)
    y_upper : numpy array, estimated upper bound for the labels (n)

    Returns
    -------
    coverage : float, average coverage
    avg_length : float, average length

    """
    in_the_range = np.sum((y_test >= y_lower) & (y_test <= y_upper))
    coverage = in_the_range / len(y_test)
    avg_length = np.mean(abs(y_upper - y_lower))
    return coverage


def compute_coverage_classification(y_pred_set, y_label):
    """Coverage score obtained by the prediction sets.

    Parameters
    ----------
    y_pred_set : numpy.array
        ndarray of shape (n_samples, n_class)
    y_label : numpy.array
        True labels

    Returns
    -------
    float
        Coverage score obtained by the prediction sets.
    """
    coverage = np.take_along_axis(
        y_pred_set, y_label.reshape(-1, 1), axis=1
    ).mean()
    return float(coverage)


def get_partition(leaf_id, part, node_id, children_left, children_right, feature, threshold):
    """Get the partition of the tree.
    """
    left = np.where(children_left == leaf_id)[0]
    right = np.where(children_right == leaf_id)[0]

    if (len(left) == 0) * (len(right) == 0):
        return part, node_id

    else:
        if len(right) != 0:
            right = int(right[0])
            node_id.append(feature[right])

            part[feature[right]] = np.concatenate((part[feature[right]], np.array([[threshold[right], np.inf]])))
            part[feature[right]] = np.array([[np.max(part[feature[right]][:, 0]), np.min(part[feature[right]][:, 1])]])
            return get_partition(right, part, node_id, children_left, children_right, feature, threshold)
        else:
            left = int(left[0])
            node_id.append(feature[left])

            part[feature[left]] = np.concatenate((part[feature[left]], np.array([[-np.inf, threshold[left]]])))
            part[feature[left]] = np.array([[np.max(part[feature[left]][:, 0]), np.min(part[feature[left]][:, 1])]])

            return get_partition(left, part, node_id, children_left, children_right, feature, threshold)


def get_values_greater_than(lst, k):
    """
    Get values greater than k in lst
    Parameters
    ----------
    lst : list
    k : int or float

    Returns
    -------
    list
        A list that contains all the values greater than k in lst
    """
    values = []
    for i in range(len(lst)):
        if lst[i] > k:
            values.append(lst[i])
    values.sort()
    return values


def save_model(model, name):
    """Save model.
    """
    with open('{}.pickle'.format(name), 'wb') as f:
        pickle.dump(model, f)


def load_model(name):
    """Load model.
    """
    with open('{}.pickle'.format(name), 'rb') as f:
        loaded_obj = pickle.load(f)
    return loaded_obj


def calibrate_ccv(pval, n_cal, delta=0.01, method="MC", simes_kden=2, fs_correction=None, a=None, two_sided=False):
    if method == "Simes":
        k = int(n_cal / simes_kden)
        output = betainv_simes(pval, n_cal, k, delta)
        two_sided = False

    elif method == "DKWM":
        epsilon = np.sqrt(np.log(2.0 / delta) / (2.0 * n_cal))
        if two_sided == True:
            output = np.minimum(1.0, 2.0 * np.minimum(pval + epsilon, 1 - pval + epsilon))
        else:
            output = np.minimum(1.0, pval + epsilon)

    elif method == "Linear":
        a = 10.0 / n_cal  # 0.005
        b = find_slope_EB(n_cal, alpha=a, prob=1.0 - delta)
        output_1 = np.minimum((pval + a) / (1.0 - b), (pval + a + b) / (1.0 + b))
        output_2 = np.maximum((1 - pval + a + b) / (1.0 + b), (1 - pval + a) / (1.0 - b))
        if two_sided == True:
            output = np.minimum(1.0, 2.0 * np.minimum(output_1, output_2))
        else:
            output = np.minimum(1.0, output_1)

    elif method == "MC":
        if fs_correction is None:
            fs_correction = estimate_fs_correction(delta, n_cal)
        output = betainv_mc(pval, n_cal, delta, fs_correction=fs_correction)
        two_sided = False

    elif method == "Asymptotic":
        k = int(n_cal / simes_kden)
        output = betainv_asymptotic(pval, n_cal, k, delta)
        two_sided = False

    else:
        raise ValueError('Invalid calibration method.')

    return output


def BH(pvalues, level):
    """
    Benjamini-Hochberg procedure.
    """
    n = len(pvalues)
    pvalues_sort_ind = np.argsort(pvalues)
    pvalues_sort = np.sort(pvalues)  # p(1) < p(2) < .... < p(n)

    comp = pvalues_sort <= (level * np.arange(1, n + 1) / n)
    # get first location i0 at which p(k) <= level * k / n
    comp = comp[::-1]
    comp_true_ind = np.nonzero(comp)[0]
    i0 = comp_true_ind[0] if comp_true_ind.size > 0 else n
    nb_rej = n - i0

    return pvalues_sort_ind[:nb_rej]


def adaptiveEmpBH(null_statistics, test_statistics, level, correction_type, storey_threshold=0.5):
    pvalues = np.array([compute_pvalue(x, null_statistics) for x in test_statistics])

    if correction_type == "storey":
        null_prop = storey_estimator(pvalues=pvalues, threshold=storey_threshold)
    elif correction_type == "quantile":
        null_prop = quantile_estimator(pvalues=pvalues, k0=len(pvalues) // 2)
    else:
        raise ValueError("correction_type is mis-specified")

    lvl_corr = level / null_prop
    return BH(pvalues=pvalues, level=lvl_corr)


def compute_pvalue(test_statistic, null_statistics):
    return (1 + np.sum(null_statistics > test_statistic)) / (len(null_statistics) + 1)


def compute_pvalue_candes(test_statistic, null_statistics):
    return (1 + np.sum(null_statistics <= test_statistic)) / (len(null_statistics) + 1)


def storey_estimator(pvalues, threshold):
    return (1 + np.sum(pvalues <= threshold)) / (len(pvalues) * (1 - threshold))


def quantile_estimator(pvalues, k0):  # eg k0=m/2
    m = len(pvalues)
    pvalues_sorted = np.sort(pvalues)
    return (m - k0 + 1) / (m * (1 - pvalues_sorted[k0]))


def prediction_wreject(model_residuals, x_test, x_cal, v_cal, v_star, level=0.2, delta=0.1, method='marginal'):
    null_statistics = model_residuals.predict(x_cal)
    test_statistics = model_residuals.predict(x_test)

    null_statistics = null_statistics[v_cal <= v_star]
    pvalues = np.array([compute_pvalue(x, null_statistics) for x in test_statistics])

    n_cal = len(null_statistics)
    simes_kden = 2
    fs_correction = estimate_fs_correction(delta, n_cal)

    if method != 'marginal':
        if method == "Simes":
            pvalues = calibrate_ccv(pvalues, n_cal, delta=delta, method=method, simes_kden=simes_kden, two_sided=False)
        elif method == 'MC':
            pvalues = calibrate_ccv(pvalues, n_cal, delta=delta, method=method, fs_correction=fs_correction,
                                    two_sided=False)
        elif method == 'Asymptotic':
            pvalues = calibrate_ccv(pvalues, n_cal, delta=delta, method=method, simes_kden=simes_kden)
        elif method == 'DKWM':
            pvalues = calibrate_ccv(pvalues, n_cal, delta=delta, method=method)
        else:
            raise ValueError('The available methods are: Simes, MC, Asymptotic, DKWM')

    rej_set = BH(pvalues, level)
    return rej_set


def compute_cn(delta, n):
    cn = -np.log(-np.log(1 - delta)) + 2 * np.log(np.log(n)) + 0.5 * np.log(np.log(np.log(n))) - 0.5 * np.log(np.pi)
    cn /= np.sqrt(2 * np.log(np.log(n)))
    return cn


def compute_hybrid_bound(delta, n, gamma):
    i = np.arange(1, n + 1)
    cna = compute_cn(delta - gamma, n)
    bound = i / n + cna * np.sqrt(i * (n - i)) / (n * np.sqrt(n))
    k_linear = int(n / 2)
    slope = (bound[k_linear - 1] - bound[k_linear - 2])
    bound[k_linear:] = bound[k_linear - 1] + slope * (i[k_linear:] - k_linear)
    k_simes = int(n / 2)
    bound_s = 1.0 - compute_aseq(n, k_simes, delta)[::-1]
    bound_h = np.minimum(bound_s, bound)
    return bound_h


def estimate_fs_correction(delta, n):
    n_mc = 10000
    U = np.random.uniform(size=(n_mc, n))
    U = np.sort(U, axis=1)
    cna = compute_cn(delta, n)
    i = np.arange(1, n + 1)
    bound_a = i / n + cna * np.sqrt(i * (n - i)) / (n * np.sqrt(n))
    k_simes = int(n / 2)
    bound_s = 1.0 - compute_aseq(n, k_simes, delta)[::-1]

    def estimate_prob_crossing(gamma):
        bound = compute_hybrid_bound(delta, n, gamma)
        crossings = np.sum(U > bound, 1)
        prob_crossing = np.mean(crossings > 0)
        return prob_crossing

    # Binary search
    gamma0 = -(1 - 1e-6 - delta)
    f0 = estimate_prob_crossing(gamma0)
    gamma1 = delta - 1e-6
    f1 = estimate_prob_crossing(gamma1)
    while np.abs(gamma1 - gamma0) > 1e-6:
        gamma = (gamma0 + gamma1) / 2
        f = estimate_prob_crossing(gamma)
        if f > delta:
            gamma0 = gamma
            f0 = f
        else:
            gamma1 = gamma
            f1 = f
    return gamma


def betainv_mc(pvals, n, delta, fs_correction=1):
    iseq = np.arange(1, n + 1)
    cn = compute_cn(delta, n)
    bound = compute_hybrid_bound(delta, n, fs_correction)
    aseq = 1 - np.minimum(1, bound[::-1])
    out = betainv_generic(pvals, aseq)
    return out


def betainv_asymptotic(pvals, n, k, delta):
    k = int(k)
    iseq = np.arange(1, n + 1)
    cn = compute_cn(delta, n)
    aseq = iseq / n + cn * np.sqrt(iseq * (n - iseq)) / (n * np.sqrt(n))
    aseq = 1 - np.minimum(1, aseq[::-1])
    out = betainv_generic(pvals, aseq)
    return out


def betainv_generic(pvals, aseq):
    n = len(aseq)
    idx = np.maximum(1, np.floor((n + 1) * (1 - pvals))).astype(int)
    out = 1 - aseq[idx - 1]
    return out


def compute_aseq(n, k, delta):
    def movingaverage(values, window):
        weights = np.repeat(1.0, window) / window
        sma = np.convolve(values, weights, 'valid')
        return sma

    k = int(k)
    fac1 = np.log(delta) / k - np.mean(np.log(np.arange(n - k + 1, n + 1)))
    fac2 = movingaverage(np.log(np.arange(1, n + 1)), k)
    aseq = np.concatenate([np.zeros((k - 1,)), np.exp(fac2 + fac1)])
    return aseq


def betainv_simes(pvals, n, k, delta):
    aseq = compute_aseq(n, k, delta)
    out = betainv_generic(pvals, aseq)
    return out


def cdf_bound(x, x_cal, aseq):
    n_cal = len(x_cal)
    x = np.reshape(np.array(x), [len(x), 1])
    jseq = np.maximum(0, np.sum(x > x_cal, 1) - 1)
    g_hat = 1 - aseq[n_cal - 1 - jseq]
    return g_hat


# Empirical bound
def find_slope_EB(n, alpha=None, prob=0.9, n_sim=5000):
    if alpha is None:
        alpha = 10.0 / n
    U = np.random.uniform(size=(n_sim, n))
    U = np.sort(U, 1)

    def get_a(n, beta, alpha=None):
        # function on bottom of page 13 of Shorack and Wellner
        # for symmetric, affine linear bands
        if alpha is None:
            alpha = 1.0 / n
        a1 = (np.arange(1, n + 1) / n - alpha) / (1.0 + beta)  # curve to the left of 1/2
        a2 = 1.0 + (np.arange(1, n + 1) / n - 1.0 - alpha) / (1.0 - beta)
        return np.maximum(a1, a2)

    def get_b(n, beta, alpha=None):
        # function on bottom of page 13 of Shorack and Wellner
        # for symmetric, affine linear bands
        if alpha is None:
            alpha = 1.0 / n
        b1 = (np.arange(1, n + 1) / n - 1.0 / n + alpha) / (1.0 - beta)
        b2 = 1.0 + (np.arange(1, n + 1) / n - 1.0 / n + alpha - 1.0) / (1.0 + beta)
        return np.minimum(b1, b2)

    def compute_coverage(U, alpha, beta):
        n = U.shape[1]
        a = get_a(n, beta, alpha=alpha)
        b = get_b(n, beta, alpha=alpha)
        Uo = np.max(np.maximum(U > b, U < a), 1)
        covg = 1.0 - np.mean(Uo)
        return covg

    # Search for beta parameter that gives the right coverage
    n2 = n * n
    beta_max = 1.0 - 1.0 / n2
    beta_min = 1.0 / n2
    while beta_max - beta_min > 1e-6:
        beta = 0.5 * (beta_min + beta_max)
        covg = compute_coverage(U, alpha, beta)
        if covg > prob:
            beta_max = beta
        else:
            beta_min = beta

    return beta


def compute_metrics(rej_set, v_test, v_star):
    fdr = np.mean(v_test[rej_set] <= v_star)
    power = np.sum(v_test[rej_set] > v_star) / np.sum(v_test > v_star)
    return fdr, power


def compute_pi(pred, r):
    if pred.ndim > 1:
        y_lower = pred[:, 0] - r
        y_upper = pred[:, 1] + r
    else:
        y_lower = pred - r
        y_upper = pred + r

    return y_lower, y_upper

def check_is_qrf_calibration(estimator):
    if not estimator.check_is_qrf_calibration:
        raise ValueError('You need to fit the QRF calibration before')


def check_is_calibrate(estimator):
    if not estimator.check_is_calibrate:
        raise ValueError('You need to fit the calibration before')


def check_is_training_conditional(estimator):
    if not estimator.check_is_training_conditional:
        raise ValueError('You need to fit the calibration with training_conditional=True '
                         'before')


def check_is_training_conditional_one(estimator):
    if not estimator.check_is_training_conditional * estimator.check_is_training_conditional_one:
        raise ValueError('You need to fit the calibration with training_conditional/one=True '
                         'before')


def check_is_calibrate_bygroup(estimator):
    if not estimator.check_is_calibrate_bygroup:
        raise ValueError('You need to fit the calibration with bygroup=True before')


def check_is_training_conditional_bygroup(estimator):
    if not estimator.check_is_training_conditional_bygroup:
        raise ValueError('You need to fit the calibration with bygroup=True and training_conditional =True before')

def check_is_reject_fitted(estimator):
    if not estimator.check_is_reject_fitted:
        raise ValueError('You need to fit the reject option by calling fit_reject')
