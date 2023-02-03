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
