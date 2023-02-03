# distutils: language = c++

from libcpp.vector cimport vector
from libcpp.set cimport set
import numpy as np
cimport numpy as np
ctypedef np.float64_t double
cimport cython
from scipy.special import comb
import itertools
from tqdm import tqdm
from acpi.utils import weighted_percentile
from cython.parallel cimport prange, parallel, threadid
from cython.operator cimport dereference as deref, preincrement as inc
cimport openmp

cdef extern from "<algorithm>" namespace "std" nogil:
     iter std_remove "std::remove" [iter, T](iter first, iter last, const T& val)
     iter std_find "std::find" [iter, T](iter first, iter last, const T& val)

cdef extern from "limits.h":
    unsigned long ULONG_MAX

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double single_compute_forest_weights(const double[:] & x, const double & y_x, const double[:, :] & data, const double[::1] & y_data,
         const int[:, :] & features, const double[:, :] & thresholds, const int[:, :] & children_left,  const int[:, :] & children_right,
         int & min_node_size, double[:] & w) nogil:

    cdef unsigned int n_trees = features.shape[0]
    cdef unsigned int N = data.shape[0]
    cdef double s, sdp
    cdef int o
    sdp = 0

    cdef int b, level, i, it_node
    cdef set[int] in_data, in_data_b
    cdef set[int].iterator it

    for b in range(n_trees):
        for i in range(N):
            in_data.insert(i)
            in_data_b.insert(i)

        it_node = 0
        while(children_left[b, it_node] >= 0 or children_right[b, it_node] >= 0):
            if x[features[b, it_node]] <= thresholds[b, it_node]:

                it = in_data.begin()
                while(it != in_data.end()):
                    if data[deref(it), features[b, it_node]] > thresholds[b, it_node]:
                        in_data_b.erase(deref(it))
                    inc(it)
                in_data = in_data_b

                it_node = children_left[b, it_node]
            else:

                it = in_data.begin()
                while(it != in_data.end()):
                    if data[deref(it), features[b, it_node]] <= thresholds[b, it_node]:
                        in_data_b.erase(deref(it))
                    inc(it)
                in_data = in_data_b

                it_node = children_right[b, it_node]

            if in_data.size() < min_node_size:
                break

        it = in_data.begin()
        while(it != in_data.end()):
            w[deref(it)] += (1./(n_trees*in_data.size()))
            #w[deref(it)] += 1
            inc(it)

        in_data.clear()
        in_data_b.clear()

    return 0


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef compute_forest_weights_verbose(const double[:, :] & X, const double[::1] & y_X, const double[:, :] & data,
                                     const double[::1] & y_data, const int[:, :] & features, const double[:, :] & thresholds,
                                       const int[:, :] & children_left, const int[:, :] & children_right, int min_node_size):

        cdef int i
        cdef int N = X.shape[0]
        cdef double[:, :] weights = np.zeros(shape=(N, data.shape[0]))
        cdef double[::1] sdp = np.zeros(N)

        for i in prange(N, nogil=True, schedule='dynamic'):
            sdp[i] = single_compute_forest_weights(X[i], y_X[i], data, y_data,
                        features, thresholds, children_left, children_right,
                        min_node_size, weights[i])

        return np.array(weights)



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double single_compute_forest_weights_cali(const double[:] & x, const double[:, :] & data, const double[::1] & y_data,
         const int[:, :] & features,  const double[:, :] & thresholds, const int[:, :] & children_left, const int[:, :] & children_right,
         int & min_node_size, double[:] & w, double[:, :] & weights) nogil:

    cdef unsigned int n_trees = features.shape[0]
    cdef unsigned int N = data.shape[0]
    cdef double s, sdp
    cdef int o
    sdp = 0

    cdef int b, level, i, it_node
    cdef set[int] in_data, in_data_b
    cdef set[int].iterator it, it_j

    for b in range(n_trees):
        for i in range(N):
            in_data.insert(i)
            in_data_b.insert(i)

        it_node = 0
        while(children_left[b, it_node] >= 0 or children_right[b, it_node] >= 0):
            if x[features[b, it_node]] <= thresholds[b, it_node]:

                it = in_data.begin()
                while(it != in_data.end()):
                    if data[deref(it), features[b, it_node]] > thresholds[b, it_node]:
                        in_data_b.erase(deref(it))
                    inc(it)
                in_data = in_data_b

                it_node = children_left[b, it_node]
            else:

                it = in_data.begin()
                while(it != in_data.end()):
                    if data[deref(it), features[b, it_node]] <= thresholds[b, it_node]:
                        in_data_b.erase(deref(it))
                    inc(it)
                in_data = in_data_b

                it_node = children_right[b, it_node]

            if in_data.size() < min_node_size:
                break

        it = in_data.begin()
        while(it != in_data.end()):
            w[deref(it)] += (1./(n_trees*in_data.size()))
            it_j = in_data.begin()
            while(it_j != in_data.end()):
                weights[deref(it), deref(it_j)] += (1./(n_trees*in_data.size())) - (1./(n_trees*(in_data.size() - 1)))
                inc(it_j)
            inc(it)

        in_data.clear()
        in_data_b.clear()

    return 0


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cpdef compute_forest_weights_cali_verbose(const double[:] & X, const double[:, :] & data, const double[::1] & y_data,
        const int[:, :] & features, const double[:, :] & thresholds, const int[:, :] & children_left,
        const int[:, :] & children_right, int min_node_size, const double[:, :] & weights):

        cdef int i
        cdef double[:] w = np.zeros(shape=(data.shape[0]))
        cdef double[:, :] weights_copy = weights.copy()

        single_compute_forest_weights_cali(X, data, y_data,
                    features, thresholds, children_left, children_right,
                    min_node_size, w, weights_copy)

        for i in prange(data.shape[0], nogil=True):
            weights_copy[data.shape[0]-1, i] = w[i]
            weights_copy[i, data.shape[0]-1] = w[i]

        return np.array(weights_copy)

cdef int l(int i) nogil:
    if i==0:
        return 0
    return i

cdef return_w_cali(const double[:] & x, const double[:, :] & x_cali,
                    const double[:] & r_cali, const double[:, :] & w_cali, acpi_instance):

    cdef np.ndarray[np.float64_t, ndim=2] w_o = np.zeros(shape=(x_cali.shape[0] + 1, x_cali.shape[0] + 1))
    w_o[:-1, :-1] = w_cali
    cdef np.ndarray[np.float64_t, ndim=2] x_cali_new = np.concatenate([x_cali, np.reshape(x, (1, -1))])
    cdef np.ndarray[np.float64_t, ndim=1] r_cali_new = np.concatenate([r_cali, [1e+30]])
    cdef np.ndarray[np.float64_t, ndim=2] w_corrected = acpi_instance.compute_forest_weights_cali(x, x_cali_new, r_cali_new, w_o)
    return w_corrected, x_cali_new, r_cali_new

cpdef return_w_cali_py(const double[:] & x, const double[:, :] & x_cali,
                    const double[:] & r_cali, const double[:, :] & w_cali, acpi_instance):

    cdef np.ndarray[np.float64_t, ndim=2] w_o = np.zeros(shape=(x_cali.shape[0] + 1, x_cali.shape[0] + 1))
    w_o[:-1, :-1] = w_cali
    cdef np.ndarray[np.float64_t, ndim=2] x_cali_new = np.concatenate([x_cali, np.reshape(x, (1, -1))])
    cdef np.ndarray[np.float64_t, ndim=1] r_cali_new = np.concatenate([r_cali, [1e+30]])
    cdef np.ndarray[np.float64_t, ndim=2] w_corrected = acpi_instance.compute_forest_weights_cali(x, x_cali_new, r_cali_new, w_o)
    return w_corrected, x_cali_new, r_cali_new


cdef int binary_search(const double[:] & S, const long[:] & delta, double alpha) nogil:
    cdef int left = 0
    cdef int right = delta.shape[0] - 1
    cdef int length_s = S.shape[0]
    cdef int mid

    while left < right:
        mid = (left + right + 1) // 2

        if delta[mid] < length_s and S[delta[mid]] < alpha:
            left = mid
        else:
            right = mid - 1

    if delta[left] < length_s and S[delta[left]] < alpha:
        return delta[left]
    else:
        return delta[-1]

cdef int binary_search_w(const double[:] & S, const double[:] & theta, double alpha) nogil:
    # On commence par définir les bornes de la recherche
    cdef int left = 0
    cdef int right = S.shape[0] - 1
    cdef int mid
    while left < right:
        mid = (left + right + 1) // 2

        if S[mid] < alpha and theta[mid] > alpha:
            left = mid
        # Sinon, on cherche à droite de mid
        else:
            right = mid - 1

    if S[left] < alpha and theta[left] > alpha:
        return left
    # Sinon, on renvoie None pour indiquer qu'aucun index ne correspond à la condition
    else:
        return -1


cdef sorted_w(const double[:, :] & w_cali, const double[:] & r_cali):

    cdef double[:, :] w_cali_sorted = np.zeros(shape=(w_cali.shape[0], w_cali.shape[1]))
    cdef long[:] argsort_ycali = np.argsort(r_cali)

    for i in range(w_cali_sorted.shape[0]):
        for j in range(w_cali_sorted.shape[0]):
            w_cali_sorted[i, j] = w_cali[argsort_ycali[i], argsort_ycali[j]]
    return w_cali_sorted, argsort_ycali


cdef compute_partition(const double[:, :] & w_corrected, const double[:] & r_cali_new):

    cdef double[:, :] w_cali_sorted
    cdef long[:] argsort_ycali
    cdef int c_1, c_2, c_3, L_1, L_2, L_3, k, i, s
    cdef double[:] S
    w_cali_sorted, argsort_ycali = sorted_w(w_corrected, r_cali_new)

    cdef double[:, :] w_cali_aug  = np.zeros(shape=(w_cali_sorted.shape[0], w_cali_sorted.shape[1]+1))
    for i in range(w_cali_aug.shape[0]):
        for j in range(1, w_cali_aug.shape[1]):
            w_cali_aug[i, j] = w_cali_sorted[i, j-1]
    cdef double[:, :] theta = np.cumsum(w_cali_aug, axis=1)

    cdef list a_1 = []
    cdef list p_1 = []
    cdef list a_2 = []
    cdef list p_2 = []
    cdef list a_3 = []
    cdef list p_3 = []

    for i in range(r_cali_new.shape[0]):

        if theta[i, l(i)] + w_cali_sorted[i, -1] < theta[-1, l(i)]:
            a_1.append(i)
            p_1.append(theta[i, l(i)] + w_cali_sorted[i, -1])

        elif theta[i, l(i)] >= theta[-1, l(i)]:
            a_2.append(i)
            p_2.append(theta[i, l(i)])
        else:
            a_3.append(i)
            p_3.append(l(i))

    cdef double[:] p_1_sort = np.sort(p_1)
    cdef double[:] p_2_sort = np.sort(p_2)
    cdef long[:] p_3_sort = np.sort(p_3).astype(np.int64)

    c_1, c_2, c_3 = 0, 0, 0
    L_1, L_2, L_3 = len(a_1), len(a_2), len(a_3)

    S = np.zeros(shape=r_cali_new.shape[0])

    for k in range(r_cali_new.shape[0]):
        while c_1 < L_1 and p_1_sort[c_1] < theta[-1, k]:
            c_1 = c_1 + 1
        while c_2 < L_2 and p_2_sort[c_2] < theta[-1, k]:
            c_2 = c_2 + 1
        while c_3 < L_3 and p_3_sort[c_3] < l(k):
            c_3 = c_3 + 1
        S[k] = (1.*(c_1 + c_2 + c_3))/(r_cali_new.shape[0])
    return S, theta[-1], np.nonzero(w_cali_aug[-1])[0]


cpdef compute_rf_lcp(const double[:, :] & x_test, const double[:, :] & x_cali,
                     const double[:] & r_cali, const double[:, :] & w_cali,
                     const float & quantile, acpi_instance):

    cdef double[:] r_lcp = np.zeros(shape=x_test.shape[0])
    cdef long[:] s_lcp = np.zeros(shape=x_test.shape[0], dtype=np.int64)
    cdef double[:, :] w_corrected
    cdef double[:, :] x_cali_new
    cdef double[:] r_cali_new
    cdef double[:] S
    cdef int k_star, i
    cdef double[:] theta
    cdef long[:] delta

    for i in tqdm(range(x_test.shape[0])):
        w_corrected, x_cali_new, r_cali_new = return_w_cali(x_test[i], x_cali, r_cali, w_cali, acpi_instance)
        S, theta, delta = compute_partition(w_corrected, r_cali_new)

        # k_star = find_kwd(S, theta, delta, quantile)
        # k_star = find_kw(S, theta, quantile)
        k_star = binary_search(S, delta, quantile)
        s_lcp[i] = k_star

        if  k_star <= r_cali.shape[0] and k_star > 0:
            r_lcp[i] = np.sort(r_cali)[k_star-1]
        elif k_star == 0:
            r_lcp[i] = np.min(r_cali)
        else:
            r_lcp[i] = np.max(r_cali)
    return np.asarray(r_lcp), np.asarray(s_lcp)

cpdef compute_rf_lcp_train_one(const double[:, :] & x_test, const double[:, :] & x_cali,
                     const double[:] & r_cali, const double[:, :] & w_cali,
                     const float & quantile, acpi_instance, const long & k):

    cdef double[:] r_lcp = np.zeros(shape=x_test.shape[0])
    cdef long[:] s_lcp = np.zeros(shape=x_test.shape[0], dtype=np.int64)
    cdef double[:, :] w_corrected
    cdef double[:, :] x_cali_new
    cdef double[:] r_cali_new
    cdef double[:] S
    cdef int k_star, i
    cdef double[:] theta
    cdef long[:] delta

    for i in tqdm(range(x_test.shape[0])):
        w_corrected, x_cali_new, r_cali_new = return_w_cali(x_test[i], x_cali, r_cali, w_cali, acpi_instance)
        S, theta, delta = compute_partition(w_corrected, r_cali_new)

        # k_star = find_kwd(S, theta, delta, quantile) + k
        # k_star = find_kw(S, theta, quantile) + k
        k_star = binary_search(S, delta, quantile) + k
        s_lcp[i] = k_star

        if  k_star <= r_cali.shape[0] and k_star > 0:
            r_lcp[i] = np.sort(r_cali)[k_star-1]
        elif k_star == 0:
            r_lcp[i] = np.min(r_cali)
        else:
            r_lcp[i] = np.max(r_cali)
    return np.asarray(r_lcp), np.asarray(s_lcp)

cpdef compute_rf_lcp_bygroup(const double[:, :] & x_test,  x_cali,
                     r_cali,  w_cali,
                     const float & quantile, acpi_instance, const long[:] & group_test):

    cdef double[:] r_lcp = np.zeros(shape=x_test.shape[0])
    cdef long[:] s_lcp = np.zeros(shape=x_test.shape[0], dtype=np.int64)
    cdef double[:, :] w_corrected
    cdef double[:, :] x_cali_new
    cdef double[:] r_cali_new
    cdef double[:] S
    cdef int k_star, i
    cdef double[:] theta
    cdef long[:] delta

    for i in tqdm(range(x_test.shape[0])):
        w_corrected, x_cali_new, r_cali_new = return_w_cali(x_test[i], x_cali[group_test[i]],
                                                            r_cali[group_test[i]], w_cali[group_test[i]],
                                                            acpi_instance)
        S, theta, delta = compute_partition(w_corrected, r_cali_new)

        k_star = binary_search(S, delta, quantile)
        s_lcp[i] = k_star

        r_cali_test = r_cali[group_test[i]]
        if  k_star <= r_cali_test.shape[0] and k_star > 0:
            r_lcp[i] = np.sort(r_cali_test)[k_star-1]
        elif k_star == 0:
            r_lcp[i] = np.min(r_cali_test)
        else:
            r_lcp[i] = np.max(r_cali_test)
    return np.asarray(r_lcp), np.asarray(s_lcp)


cpdef compute_rf_lcp_bygroup_train(const double[:, :] & x_test,  x_cali,
                     r_cali,  w_cali,
                     const float & quantile, acpi_instance, const long[:] & group_test,
                     const long & k):

    cdef double[:] r_lcp = np.zeros(shape=x_test.shape[0])
    cdef long[:] s_lcp = np.zeros(shape=x_test.shape[0], dtype=np.int64)
    cdef double[:, :] w_corrected
    cdef double[:, :] x_cali_new
    cdef double[:] r_cali_new
    cdef double[:] S
    cdef int k_star, i
    cdef double[:] theta
    cdef long[:] delta

    for i in tqdm(range(x_test.shape[0])):
        w_corrected, x_cali_new, r_cali_new = return_w_cali(x_test[i], x_cali[group_test[i]],
                                                            r_cali[group_test[i]], w_cali[group_test[i]],
                                                            acpi_instance)
        S, theta, delta =compute_partition(w_corrected, r_cali_new)

        k_star = binary_search(S, delta, quantile) + k
        s_lcp[i] = k_star

        r_cali_test = r_cali[group_test[i]]
        if  k_star <= r_cali_test.shape[0] and k_star > 0:
            r_lcp[i] = np.sort(r_cali_test)[k_star-1]
        elif k_star == 0:
            r_lcp[i] = np.min(r_cali_test)
        else:
            r_lcp[i] = np.max(r_cali_test)
    return np.asarray(r_lcp), np.asarray(s_lcp)

cpdef compute_rf_lcp_train(const double[:, :] & x_test, const double[:, :] & x_cali,
                     const double[:] & r_cali, const double[:, :] & w_cali,
                     const float & quantile, acpi_instance, const long & k):

    cdef double[:] r_lcp = np.zeros(shape=x_test.shape[0])
    cdef long[:] s_lcp = np.zeros(shape=x_test.shape[0], dtype=np.int64)
    cdef double[:, :] w_corrected
    cdef double[:, :] x_cali_new
    cdef double[:] r_cali_new
    cdef double[:] S
    cdef int k_star, i
    cdef list support, values_greater
    cdef double[:] theta
    cdef long[:] delta

    for i in tqdm(range(x_test.shape[0])):
        w_corrected, x_cali_new, r_cali_new = return_w_cali(x_test[i], x_cali, r_cali, w_cali, acpi_instance)
        S, theta, delta = compute_partition(w_corrected, r_cali_new)

        k_star = binary_search(S, delta, quantile)
        s_lcp[i] = k_star

        if  k_star <= r_cali.shape[0] and k_star > 0:
            r_lcp[i] = np.sort(r_cali)[k_star-1]
        elif k_star == 0:
            r_lcp[i] = np.min(r_cali)
        else:
            r_lcp[i] = np.max(r_cali)

        support = []
        for j in range(r_cali_new.shape[0]):
            if w_corrected[-1, j] > 0:
                support.append(r_cali_new[j])

        values_greater = get_values_greater_than(support, r_lcp[i])
        if k < len(values_greater) and k > 0:
            r_lcp[i] = values_greater[k-1]
        elif k >= len(values_greater):
            r_lcp[i] = np.max(r_cali)
    return np.asarray(r_lcp), np.asarray(s_lcp)


cpdef compute_rf_lcp_support(const double[:, :] & x_test, const double[:, :] & x_cali,
                     const double[:] & r_cali, const double[:, :] & w_cali,
                     const float & quantile, acpi_instance):

    cdef double[:] r_lcp = np.zeros(shape=x_test.shape[0])
    cdef long[:] s_lcp = np.zeros(shape=x_test.shape[0], dtype=np.int64)
    cdef double[:, :] w_corrected
    cdef double[:, :] x_cali_new
    cdef double[:] r_cali_new
    cdef double[:] S
    cdef int k_star, i, k
    cdef double[:] theta
    cdef long[:] delta

    cdef list supports, support
    supports = []
    for i in tqdm(range(x_test.shape[0])):
        w_corrected, x_cali_new, r_cali_new = return_w_cali(x_test[i], x_cali, r_cali, w_cali, acpi_instance)
        S, theta, delta =compute_partition(w_corrected, r_cali_new)

        support = []
        for k in range(r_cali_new.shape[0]):
            if w_corrected[-1, k] > 0:
                support.append(r_cali_new[k])
        supports.append(support)

        k_star = binary_search(S, delta, quantile)
        s_lcp[i] = k_star

        if  k_star <= r_cali.shape[0] and k_star > 0:
            r_lcp[i] = np.sort(r_cali)[k_star-1]
        elif k_star == 0:
            r_lcp[i] = np.min(r_cali)
        else:
            r_lcp[i] = np.max(r_cali)
    return np.asarray(r_lcp), np.asarray(s_lcp), supports


cpdef get_changed_values(lst):
    changed_values = {}
    for i, value in enumerate(lst):
        if i == 0 or lst[i-1] != value:
            changed_values[i] = value
    return changed_values

cpdef find_index(lst, k):
  # On parcourt les éléments de la liste
  for i, x in enumerate(lst):
    # Si l'élément courant est égal à k, on renvoie son index
    if x == k:
      return i
  # Si aucun élément n'est égal à k, on renvoie None
  return None

cdef get_values_greater_than(list lst, double k):
  cdef list values = []

  for i in range(len(lst)):
    if lst[i] > k:
      values.append(lst[i])

  values.sort()
  return values

cpdef find_kw(const double[:] & S, const double[:] & theta, double alpha):
    for k in range(S.shape[0]):
        if S[k] >= alpha and theta[k] >= alpha:
            return k
    return -1

cpdef find_kwd(const double[:] & S, const double[:] & theta,
              const long[:] delta, double alpha):
    for k in range(delta.shape[0]):
        if S[delta[k]] >= alpha and theta[delta[k]] >= alpha:
            return delta[k]
    return -1