from .base_forest import *
from .utils import classifier_score, quantile_score, mean_score, compute_coverage, compute_coverage_classification, \
    get_values_greater_than
import numpy as np
import cyext_acv
from skranger.ensemble import RangerForestRegressor
from sklearn.utils.validation import check_is_fitted, check_X_y, column_or_1d, check_array, as_float_array, \
    check_consistent_length
from scipy.sparse import csr_matrix
import scipy.sparse as sp
import pygenstability
from tqdm import tqdm


class ACPI:
    def __init__(
            self,
            model_cali,
            estimator='reg',
            n_estimators=200,
            mtry=0,
            min_node_size=10,
            max_depth=15,
            replace=True,
            sample_fraction=None,
            keep_inbag=False,
            inbag=None,
            split_rule="variance",
            num_random_splits=1,
            seed=2021,
            verbose=False,
            importance="impurity",
    ):
        """Adaptive Conformal Prediction Interval* (ACPI).

        Adaptive Conformal Prediction Interval* (ACPI) is a Python package that aims to provide
        Adaptive Predictive Interval (PI) that better represent the uncertainty of the
        model by reweighting the NonConformal Score with the learned weights of a Random Forest while guaranteeing
        marginal, training-conditional and assymptotic conditiona coverage.


        Parameters
        ----------
        model_cali : model object
            It can be any model (regression, quantile regression or classification model) with scikit-learn API, i.e.

            . "predict" method for regression (mean or quantile estimates)

            . "predict" and "predict_proba" methos for classification.
         estimator : string
            Set "clf" if model_cali is a classifer, or "reg" if it is a regressor (mean or quantile).
        n_estimators : int
            Number of trees used in the forest
        mtry : int or function
            The number of features to split on each node. When a  callable is passed, the function must accept a
            single parameter which is the number of features passed, and return some value between 1 and the number of
            features.
        min_node_size : int
            The minimal node size.
        max_depth : int
            The maximal tree depth; 0 means unlimited.
        replace : bool
            Sample with replacement.
        sample_fraction : float
            The fraction of observations to sample. The default is 1 when sampling with
            replacement, and 0.632 otherwise. This can be a list of class specific values.
        keep_inbag : bool
            If true, save how often observations are in-bag in each tree.
        inbag : list
            A list of size ``n_estimators``, containing inbag counts for each observation.
            Can be used for stratified sampling.
        split_rule : string
            one of 'variance', 'extratrees', 'maxstat', 'beta' for regression; default 'variance'
        num_random_splits : int
            The number of random splits to consider for the ``extratrees`` splitrule.
        seed : int
            Random seed value
        verbose : bool
            Enable ranger's verbose logging
        importance : string
            It compute feature importance when learning the RF.
            One of one of ``none``, ``impurity``, ``impurity_corrected``, ``permutation``.
        """
        self.n_estimators = n_estimators
        self.verbose = verbose
        self.mtry = mtry
        self.importance = importance
        self.min_node_size = min_node_size
        self.max_depth = max_depth
        self.replace = replace
        self.sample_fraction = sample_fraction
        self.keep_inbag = keep_inbag
        self.inbag = inbag
        self.split_rule = split_rule
        self.num_random_splits = num_random_splits
        self.check_is_acpi_fit = False
        self.ACPI = None
        self.seed = seed
        self.d = None
        self.depth = max_depth * np.ones(shape=n_estimators, dtype=np.int32)
        self.x_cali = None
        self.r_cali = None
        self.w_cali = None
        self.w_cali_bygroup = None
        self.check_is_calibrate = False
        self.communities = None
        self.all_results = None
        self.x_cali_bygroup = None
        self.r_cali_bygroup = None
        self.x_cali_train = None
        self.r_cali_train = None
        self.check_is_calibrate_bygroup = True
        self.r_lcp_train_cali = None
        self.s_lcp_train_cali = None
        self.pred_cali_train = None
        self.k_cali = None
        self.k_cali_bygroup = None
        self.check_is_training_conditional = False
        self.coverage_cali = 0
        self.coverage_cali_bygroup = 0
        self.r_lcp_bygroup = None
        self.s_lcp_bygroup = None
        self.check_is_training_conditional_bygroup = False
        self.w_cali_train = None
        self.support_train_cali = None
        self.y_cali = None
        self.check_is_qrf_calibration = False
        self.alpha_star = None
        self.coverage_qrf = None
        self.k_cali_one = None
        self.coverage_cali_one = None
        self.check_is_training_conditional_one = False
        self.pred_cali = None
        self.model_cali = model_cali
        self.score_cali = None
        self.score_type = None
        self.idx_marg = None
        self.idx_train = None
        self.only_qrf = None
        self.nonconformity_score = None
        self.estimator = estimator
        self.pred_cali_proba = None
        self.pred_cali_proba_train = None
        self.quantile = None

        self.model = RangerForestRegressor(n_estimators=self.n_estimators,
                                           verbose=self.verbose,
                                           mtry=self.mtry,
                                           importance=self.importance,
                                           min_node_size=self.min_node_size,
                                           max_depth=self.max_depth,
                                           replace=self.replace,
                                           sample_fraction=self.sample_fraction,
                                           keep_inbag=self.keep_inbag,
                                           inbag=self.inbag,
                                           split_rule=self.split_rule,
                                           num_random_splits=self.num_random_splits,
                                           seed=self.seed,
                                           enable_tree_details=True,
                                           quantiles=True)

        self.qrf = RangerForestRegressor(n_estimators=self.n_estimators,
                                         verbose=self.verbose,
                                         mtry=self.mtry,
                                         importance=self.importance,
                                         min_node_size=self.min_node_size,
                                         max_depth=self.max_depth,
                                         replace=self.replace,
                                         sample_fraction=self.sample_fraction,
                                         keep_inbag=self.keep_inbag,
                                         inbag=self.inbag,
                                         split_rule=self.split_rule,
                                         num_random_splits=self.num_random_splits,
                                         seed=self.seed,
                                         enable_tree_details=True,
                                         quantiles=True)

    def fit(self, X, y, nonconformity_func=None, sample_weight=None, split_select_weights=None,
            always_split_features=None, categorical_features=None):
        """Fit a Random Forest estimator to predict the nonconformity score of the calibration samples.

        It should be used to find the good parameters of the Random Forest to predict the nonconformity score of the
        calibration samples, before using the method 'fit_calibration'.

        Parameters
        ----------
        X : numpy.array or pandas.DataFrame
            Calibration samples of shape (n_samples, n_features)
        y : numpy.array
            Calibration labels of shape (n_samples,)
        nonconformity_func : function, optional
            A callable that returns the nonconformity scores given the predictions of the model and the labels.
            By default,
            . if predictions.ndim == 1 it uses the absolute residuals |Y-predictions|
            . elif predictions.ndim==2, it use the classic quantile score max(predictions[:, 0]-Y, Y-predictions[:, 1]).
        sample_weight : numpy.array, optional
            optional weights for input samples
        split_select_weights : list, optional
            Vector of weights between 0 and 1 of probabilities to select features for splitting. Can be a single vector
            or a vector of vectors with one vector per tree.
        always_split_features : list, optional
            Features which should always be selected for splitting. A list of column index values
        categorical_features : list, optional
            A list of column index values which should be considered categorical, or unordered.

        Returns
        -------
        ACPI
            ACPI estimator instance
        """
        X, y = check_X_y(X, y, dtype=np.double)
        predictions = self.model_cali.predict(X)

        if self.estimator == 'clf':
            pred_cali_proba = self.model_cali.predict_proba(X)
            r_fit = classifier_score(pred_cali_proba, y)
        elif nonconformity_func is not None:
            r_fit = nonconformity_func(predictions, y)
        elif predictions.ndim > 1:
            r_fit = quantile_score(predictions, y)
        else:
            r_fit = mean_score(predictions, y)

        self.d = X.shape[1]
        self.model.fit(X=X, y=r_fit, sample_weight=sample_weight, split_select_weights=split_select_weights,
                       always_split_features=always_split_features, categorical_features=categorical_features)

        for i in range(len(self.model.estimators_)):
            self.depth[i] = self.model.estimators_[i].get_depth()
        return self

    def fit_calibration(self, x_cali, y_cali, nonconformity_func=None, quantile=0.9, only_qrf=True, n_steps_qrf=40,
                        n_iter_qrf=50, training_conditional=False, bygroup=False, training_conditional_one=False,
                        method='leiden', min_scale=-1, max_scale=0.5, constructor='continuous_normalized', n_scale=100,
                        n_tries=50):
        """Calibrate the Quantile Regression Forest (QRF) for marginal and training-conditional coverage.

         By default, it only calibrates the QRF for training-conditional coverage, but you can also compute only
         marginal coverage by setting (only_qrf=False), have both marginal and training-conditional by setting
         (only_qrf=False, training_conditional=True), and finally you can compute the cluster/partition defined by
         the forest weights to allow groupwise-conformalization by setting (only_qrf=False, training_conditional=True,
         bygroup=True).

        Parameters
        ----------
        x_cali : numpy.array or pandas.DataFrame
            Calibration samples of shape (n_samples, n_features)
        y_cali : numpy.array
            Calibration labels of shape (n_samples,)
        nonconformity_func : function, optional
            A callable that returns the nonconformity scores given the predictions of the model and the labels.
            By default,
            . if predictions.ndim == 1 it uses the absolute residuals |Y-predictions|
            . elif predictions.ndim==2, it use the classic quantile score max(predictions[0]-Y, Y-predictions[1]).
        quantile : float
            The desired coverage level, by default quantile=0.9
        only_qrf : bool, optional
            If True, only compute QRF training-conditional calibration, otherwise also compute LCP-RF.
            By default, it only computes QRF training-conditional.
        n_steps_qrf : int
            It is uses to discretize the interval (0 ,1) to generate the candidate values for \hat{\alpha}. The latter
            is the correction term used in the quantile of LCP-RF/QRF to achieve training-conditional coverage.
             n_steps_qrf is the number of time
        n_iter_qrf : int
            It is the number of time that the calibration sample is randomly split to compute training-conditional
            calibration. Indeed, we split the calibration n_iter_qrf times and choose the split that lead to the best
            performance for training-conditional coverage/width.
        training_conditional : bool, optional
            If True, it also calibrate LCP-RF for training-conditional. By default, it is False.
        bygroup : bool, optional
            If True, it computes the cluster/partition of the calibration data using the weights of the Forest.
            Hence, allowing groupwise conformalization.
        training_conditional_one : bool,
            If False, it calibrate LCP-RF for training-conditional using the cumulative probabilities of the QRF as
            candidate for \hat{\alpha}, otherwise we just arbitrary values by discretizing the interval (0, 1).
        method : string
            Parameters of the clustering algorithm (see the Package PyGenStability): optimiation method,
             louvain or leiden. By default, leiden.
        constructor : str or function
            Parameters of the clustering algorithm (see the Package PyGenStability): name of the quality constructor,
            or custom constructor function. It must have two arguments, graph and scale.
        min_scale : float
            Parameters of the clustering algorithm (see the Package PyGenStability): minimum Markov scale
        max_scale : float
            Parameters of the clustering algorithm (see the Package PyGenStability): maximum Markov scale
        n_scale : numpy.array
            Parameters of the clustering algorithm (see the Package PyGenStability): custom scale vector, if provided,
            it will override the other scale arguments
        n_tries : int
            Parameters of the clustering algorithm (see the Package PyGenStability): number of modularity optimisation evaluations

        Returns
        -------
        ACPI
            ACPI estimator instance
        """
        self.x_cali, self.y_cali = check_X_y(x_cali, y_cali, dtype=np.double)
        self.pred_cali = self.model_cali.predict(self.x_cali).astype(np.double)
        self.only_qrf = only_qrf
        self.quantile = quantile

        if self.estimator == 'clf':
            self.pred_cali_proba = self.model_cali.predict_proba(self.x_cali).astype(np.double)
            self.r_cali = classifier_score(self.pred_cali_proba, self.y_cali)
            self.nonconformity_score = classifier_score
        elif nonconformity_func is not None:
            self.r_cali = nonconformity_func(self.pred_cali, self.y_cali)
            self.nonconformity_score = nonconformity_func
        elif self.pred_cali.ndim > 1:
            self.r_cali = quantile_score(self.pred_cali, self.y_cali)
            self.nonconformity_score = quantile_score
        else:
            self.r_cali = mean_score(self.pred_cali, self.y_cali)
            self.nonconformity_score = mean_score

        self.r_cali = as_float_array(self.r_cali).astype(np.double)

        print('Training calibration of QRF')
        self.alpha_star, self.coverage_qrf, self.idx_marg, self.idx_train = self.fit_qrf_calibration(steps=n_steps_qrf,
                                                                                                     n_iter=n_iter_qrf)
        if self.only_qrf:
            return self

        if training_conditional:
            r_cali = self.r_cali.copy()
            # n_half = int(np.floor(x_cali.shape[0] * 0.5))
            self.x_cali = x_cali[self.idx_marg]
            self.r_cali = r_cali[self.idx_marg]
            self.y_cali = y_cali[self.idx_marg]

            self.x_cali_train = x_cali[self.idx_train]
            self.r_cali_train = r_cali[self.idx_train]
            self.y_cali_train = y_cali[self.idx_train]
            self.pred_cali_train = self.pred_cali[self.idx_train]
            if self.estimator == 'clf':
                self.pred_cali_proba_train = self.pred_cali_proba[self.idx_train]

        print('Marginal calibration of RF-LCP')

        self.d = self.x_cali.shape[1]
        self.model.fit(self.x_cali, self.r_cali)
        self.ACPI = BaseAgnosTree(self.model, self.d)
        self.check_is_acpi_fit = True

        self.w_cali = self.compute_forest_weights(self.x_cali, self.r_cali, self.x_cali, self.r_cali)
        self.check_is_calibrate = True

        if training_conditional:
            print('Training-conditional calibration of LCP-RF')
            self.r_lcp_train_cali, self.s_lcp_train_cali, self.support_train_cali = self.predict_rf_lcp_support(
                self.x_cali_train, quantile)
            # self.support_train_cali = np.sort(self.support_train_cali)

            self.k_cali, self.coverage_cali = self.train_conditional_calibration()
            self.check_is_training_conditional = True

        if training_conditional_one:
            print('Training-conditional-one calibration of RF-LCP')

            self.k_cali_one, self.coverage_cali_one = self.train_conditional_calibration_one()
            self.check_is_training_conditional_one = True

        if bygroup:
            print('Computing communities using the RF weights')
            weights_csr = csr_matrix(self.w_cali)
            s, communities = sp.csgraph.connected_components(weights_csr, directed=False)
            if s > 1:
                self.communities = communities
            else:
                self.all_results = pygenstability.run(weights_csr, constructor=constructor,
                                                      min_scale=min_scale, max_scale=max_scale,
                                                      n_scale=n_scale, n_tries=n_tries,
                                                      method=method)
                self.communities = self.all_results['community_id'][-1]

            self.x_cali_bygroup = []
            self.r_cali_bygroup = []
            self.w_cali_bygroup = []
            print('Marginal calibration of Group-wise RF-LCP')
            for group in np.unique(self.communities):
                self.x_cali_bygroup.append(self.x_cali[self.communities == group])
                self.r_cali_bygroup.append(self.r_cali[self.communities == group])
                self.w_cali_bygroup.append(
                    self.compute_forest_weights(self.x_cali_bygroup[-1],
                                                self.r_cali_bygroup[-1],
                                                self.x_cali_bygroup[-1],
                                                self.r_cali_bygroup[-1])
                )

            self.check_is_calibrate_bygroup = True

            if training_conditional:
                print('Training-conditional calibration of Group-wise LCP')
                self.r_lcp_bygroup, self.s_lcp_bygroup = self.predict_rf_lcp_bygroup(self.x_cali_train, quantile)
                self.k_cali_bygroup, self.coverage_cali_bygroup = \
                    self.train_conditional_calibration_bygroup()
                self.check_is_training_conditional_bygroup = True
        return self

    def fit_qrf_calibration(self, steps=40, n_iter=20):
        """Calibrate the Quantile Regression Forest for training-conditional coverage.

        Parameters
        ----------
        steps : int
            It is uses to discretize the interval (0 ,1) to generate the candidate values for \hat{\alpha}. The latter
            is the correction term used in the quantile of LCP-RF/QRF to achieve training-conditional coverage.
             n_steps_qrf is the number of time
        n_iter : int
            It is the number of time that the calibration sample is randomly split to compute training-conditional
            calibration. Indeed, we split the calibration n_iter_qrf times and choose the split that lead to the best
            performance for training-conditional coverage/width.

        Returns
        -------
        best_alpha : float
            The quantile level of the QRF that lead to training-conditional coverage
        best_coverage : float
            The empiracal coverage rate attains on the second calibration data.
        idx_cali : numpy.array
            Indices of the calibration data used for training the QRF.
        idx_train : numpy.array
            Indices of the calibration data used to find best_alpha for training-conditional coverage.
        """
        best_idx_marg = []
        best_idx_train = []
        best_alpha = []
        best_coverage = []
        for z in tqdm(range(n_iter)):
            idx = np.random.permutation(self.x_cali.shape[0])
            n_half = int(np.floor(self.x_cali.shape[0] * 0.5))
            x_cali = self.x_cali[idx[:n_half]]
            r_cali = self.r_cali[idx[:n_half]]
            y_cali = self.y_cali[idx[:n_half]]

            x_cali_train = self.x_cali[idx[n_half:]]
            r_cali_train = self.r_cali[idx[n_half:]]
            y_cali_train = self.y_cali[idx[n_half:]]
            pred_cali_train = self.pred_cali[idx[n_half:]]
            if self.estimator == 'clf':
                pred_cali_proba_train = self.pred_cali_proba[idx[n_half:]]

            self.qrf.fit(x_cali, r_cali)

            alpha_sets = np.linspace(self.quantile, 1, steps)
            for i in range(alpha_sets.shape[0]):
                alpha_star = alpha_sets[i]
                r = self.qrf.predict_quantiles(x_cali_train, quantiles=[alpha_star])

                if self.estimator == 'clf':
                    y_pred_set = np.greater_equal(pred_cali_proba_train, 1 - r.reshape(-1, 1))
                    coverage = compute_coverage_classification(y_pred_set, y_cali_train)
                else:
                    if self.pred_cali.ndim == 1:
                        y_lower_qrft = pred_cali_train - r
                        y_upper_qrft = pred_cali_train + r
                    else:
                        y_lower_qrft = pred_cali_train[:, 0] - r
                        y_upper_qrft = pred_cali_train[:, 1] + r

                    coverage = compute_coverage(y_cali_train, y_lower_qrft, y_upper_qrft)

                if coverage >= self.quantile:
                    break
            best_idx_marg.append(idx[:n_half])
            best_idx_train.append(idx[n_half:])
            best_coverage.append(coverage)
            best_alpha.append(alpha_star)

        best_id = np.argmin(best_alpha)
        x_cali = self.x_cali[best_idx_marg[best_id]]
        r_cali = self.r_cali[best_idx_marg[best_id]]
        self.qrf.fit(x_cali, r_cali)

        self.check_is_qrf_calibration = True

        return best_alpha[best_id], best_coverage[best_id], best_idx_marg[best_id], best_idx_train[best_id]

    def predict_pi(self, x_test, method='qrf'):
        """Return the Prediction Intervals/Sets using QRF, LCP-RF, or LCP-RF-G with training-conditional calibration.

        Parameters
        ----------
        x_test : numpy.array
            Test samples
        method : string
            Method to choose for Predictions Intervals/Sets estimates.
            Choose among:
            . "qrf": QRF with training-conditional calibration (recommended).

            . "lcp-rf": LCP-RF with training-conditional calibration.

            . "lcp-rf-g": LCP-RF with groupwise training-conditional calibration.

            By default, method="qrf".
        Returns
        -------
        numpy.array
            . For regression, it returns the Predictive Intervals as a ndarray of shape (n_samples, 2).
            The first dimension is the lower bound and the last is the upper bound interval.

            . For classification, it returns the Predictive sets as a boolean ndarray of shape (n_samples, n_class).
        """
        if method == 'qrf':
            if not self.check_is_qrf_calibration:
                raise ValueError('You need to fit the QRF calibration before')
            r = self.predict_qrf_r(x_test)
        elif method == 'lcp-rf':
            if not self.check_is_qrf_calibration:
                raise ValueError('You need to fit the QRF calibration before')
            r, _ = self.predict_rf_lcp_train_one(x_test, self.quantile)
        elif method == 'lcp-rf-group':
            r, _ = self.predict_rf_lcp_bygroup_train(x_test, self.quantile)
        else:
            raise ValueError("Available methods are: 'qrf', 'lcp-rf', 'lcp-rf-group'")

        pred = self.model_cali.predict(x_test)
        if self.estimator == 'clf':
            pred_proba = self.model_cali.predict_proba(x_test)
            y_pred_set = np.greater_equal(pred_proba, 1 - r.reshape(-1, 1))
            return y_pred_set
        else:
            if self.pred_cali.ndim > 1:
                y_lower = pred[:, 0] - r
                y_upper = pred[:, 1] + r
            else:
                y_lower = pred - r
                y_upper = pred + r
        return y_lower, y_upper

    def predict_qrf_pi(self, x_test):
        """Return the Prediction Intervals/Sets using training-conditional calibrate QRF.

        Parameters
        ----------
        x_test : numpy.array
            Test samples
        Returns
        -------
        numpy.array
            For regression, it returns the Predictive Intervals as a ndarray of shape (n_samples, 2).
            The first dimension is the lower bound and the last is the upper bound interval.

            For classification, it returns the Predictive sets as a boolean ndarray of shape (n_samples, n_class).
        """
        if not self.check_is_qrf_calibration:
            raise ValueError('You need to fit the QRF calibration before')
        r = self.predict_qrf_r(x_test)
        pred = self.model_cali.predict(x_test)
        if self.estimator == 'clf':
            pred_proba = self.model_cali.predict_proba(x_test)
            y_pred_set = np.greater_equal(pred_proba, 1 - r.reshape(-1, 1))
            return y_pred_set
        else:
            if self.pred_cali.ndim > 1:
                y_lower = pred[:, 0] - r
                y_upper = pred[:, 1] + r
            else:
                y_lower = pred - r
                y_upper = pred + r
        return y_lower, y_upper

    def predict_qrf_r(self, x_test, quantile=None):
        """Compute the correction term for training-conditional of QRF.

        Parameters
        ----------
        x_test : numpy.array
            Test samples
        quantile : By default, it used the estimated quantile to achieve training-conditional but you can overwrite
        the value usint this parameter.

        Returns
        -------
        numpy.array
            ndarray of shape (n_samples,) that corresponds the corrected terms or the adapted quantiles.
        """
        if not self.check_is_qrf_calibration:
            raise ValueError('You need to fit the QRF calibration before')
        if quantile is None:
            return self.qrf.predict_quantiles(x_test, quantiles=[self.alpha_star])
        return self.qrf.predict_quantiles(x_test, quantiles=[quantile]), None

    def predict_rf_lcp(self, x_test, quantile):
        """Compute the correction term for LCP-RF (marginal coverage).

        Parameters
        ----------
        x_test : numpy.array
            Test samples
        quantile : By default, it used the estimated quantile to achieve training-conditional but you can overwrite
        the value usint this parameter.

        Returns
        -------
        numpy.array
            ndarray of shape (n_samples,) that corresponds the corrected terms or the adapted quantiles.
        """
        if not self.check_is_calibrate:
            raise ValueError('You need to fit the calibration before')
        return cyext_acv.compute_rf_lcp(x_test, self.x_cali, self.r_cali, self.w_cali, quantile, self)

    def predict_rf_lcp_train(self, x_test, quantile, k=None):
        """Compute the correction term for training-conditional LCP-RF.

        Parameters
        ----------
        x_test : numpy.array
            Test samples
        quantile : By default, it used the estimated quantile to achieve training-conditional but you can overwrite
        the value usint this parameter.

        Returns
        -------
        numpy.array
            ndarray of shape (n_samples,) that corresponds the corrected terms or the adapted quantiles.
        """
        if not self.check_is_training_conditional:
            raise ValueError('You need to fit the calibration with training_conditional=True '
                             'before')
        if k is None:
            return cyext_acv.compute_rf_lcp_train(x_test, self.x_cali, self.r_cali,
                                                  self.w_cali, quantile, self, self.k_cali)
        return cyext_acv.compute_rf_lcp_train(x_test, self.x_cali, self.r_cali,
                                              self.w_cali, quantile, self, k)

    def predict_rf_lcp_train_one(self, x_test, quantile, k=None):
        """Compute the correction term for training-conditional LCP-RF using discretization of (0, 1).

        Parameters
        ----------
        x_test : numpy.array
            Test samples
        quantile : By default, it used the estimated quantile to achieve training-conditional but you can overwrite
        the value usint this parameter.

        Returns
        -------
        numpy.array
            ndarray of shape (n_samples,) that corresponds the corrected terms or the adapted quantiles.
        """
        if not self.check_is_training_conditional * self.check_is_training_conditional_one:
            raise ValueError('You need to fit the calibration with training_conditional/one=True '
                             'before')
        if k is None:
            return cyext_acv.compute_rf_lcp_train_one(x_test, self.x_cali, self.r_cali,
                                                      self.w_cali, quantile, self, self.k_cali_one)
        return cyext_acv.compute_rf_lcp_train_one(x_test, self.x_cali, self.r_cali,
                                                  self.w_cali, quantile, self, k)

    def predict_rf_lcp_bygroup(self, x_test, quantile):
        """Compute the correction term for groupwise LCP-RF.

        Parameters
        ----------
        x_test : numpy.array
            Test samples
        quantile : By default, it used the estimated quantile to achieve training-conditional but you can overwrite
        the value usint this parameter.

        Returns
        -------
        numpy.array
            ndarray of shape (n_samples,) that corresponds the corrected terms or the adapted quantiles.
        """
        if not self.check_is_calibrate_bygroup:
            raise ValueError('You need to fit the calibration with bygroup=True before')

        y_test_base = np.zeros(x_test.shape[0])
        w_test = self.compute_forest_weights(x_test, y_test_base, self.x_cali, self.r_cali)
        groups = np.unique(self.communities)
        p_test = np.zeros(shape=(x_test.shape[0], len(groups)))
        for group in groups:
            p_test[:, group] = np.sum(w_test[:, self.communities == group], axis=1)

        group_test = np.argmax(p_test, axis=1)
        return cyext_acv.compute_rf_lcp_bygroup(x_test, self.x_cali_bygroup, self.r_cali_bygroup,
                                                self.w_cali_bygroup, quantile, self, group_test)

    def predict_rf_lcp_bygroup_train(self, x_test, quantile, k=None):
        """Compute the correction term for training-conditional groupwsize LCP-RF.

        Parameters
        ----------
        x_test : numpy.array
            Test samples
        quantile : By default, it used the estimated quantile to achieve training-conditional but you can overwrite
        the value usint this parameter.

        Returns
        -------
        numpy.array
            ndarray of shape (n_samples,) that corresponds the corrected terms or the adapted quantiles.
        """

        if not self.check_is_training_conditional_bygroup:
            raise ValueError('You need to fit the calibration with bygroup=True and training_conditional =True before')

        y_test_base = np.zeros(x_test.shape[0])
        w_test = self.compute_forest_weights(x_test, y_test_base, self.x_cali, self.r_cali)
        groups = np.unique(self.communities)
        p_test = np.zeros(shape=(x_test.shape[0], len(groups)))
        for group in groups:
            p_test[:, group] = np.sum(w_test[:, self.communities == group], axis=1)

        group_test = np.argmax(p_test, axis=1)
        if k is None:
            return cyext_acv.compute_rf_lcp_bygroup_train(x_test, self.x_cali_bygroup, self.r_cali_bygroup,
                                                          self.w_cali_bygroup, quantile, self, group_test,
                                                          self.k_cali_bygroup)
        else:
            return cyext_acv.compute_rf_lcp_bygroup_train(x_test, self.x_cali_bygroup, self.r_cali_bygroup,
                                                          self.w_cali_bygroup, quantile, self, group_test,
                                                          k)

    def predict_rf_lcp_support(self, x_test, quantile):
        if not self.check_is_calibrate:
            raise ValueError('You need to fit the calibration before')
        return cyext_acv.compute_rf_lcp_support(x_test, self.x_cali, self.r_cali, self.w_cali, quantile, self)

    def train_conditional_calibration(self):
        """Training-conditional calibration of LCP-RF.

        Returns
        -------
        int
            Index of the correction term using the values of the QRF
        coverage
            The coverage reached with the second calibration data.
        """
        if self.estimator == 'clf':
            y_pred_set = np.greater_equal(self.pred_cali_proba_train, 1 - self.r_lcp_train_cali.reshape(-1, 1))
            coverage = compute_coverage_classification(y_pred_set, self.y_cali_train)
        else:
            if self.pred_cali.ndim == 1:
                y_lower = self.pred_cali_train - self.r_lcp_train_cali
                y_upper = self.pred_cali_train + self.r_lcp_train_cali
            else:
                y_lower = self.pred_cali_train[:, 0] - self.r_lcp_train_cali
                y_upper = self.pred_cali_train[:, 1] + self.r_lcp_train_cali

            coverage = compute_coverage(self.y_cali_train, y_lower, y_upper)

        possible_values = []
        for i in range(self.x_cali_train.shape[0]):
            possible_values.append(get_values_greater_than(self.support_train_cali[i], self.r_lcp_train_cali[i]))
            # print('values', possible_values[-1])

        r_lcp_train = np.zeros(shape=self.r_lcp_train_cali.shape)

        k = 0
        while coverage < self.quantile:
            for i in range(self.x_cali_train.shape[0]):
                if k < len(possible_values[i]):
                    r_lcp_train[i] = possible_values[i][k]
                else:
                    r_lcp_train[i] = possible_values[i][len(possible_values[i]) - 1]

            if self.estimator == 'clf':
                y_pred_set = np.greater_equal(self.pred_cali_proba_train, 1 - r_lcp_train.reshape(-1, 1))
                coverage = compute_coverage_classification(y_pred_set, self.y_cali_train)
            else:
                if self.pred_cali.ndim == 1:
                    y_lower = self.pred_cali_train - r_lcp_train
                    y_upper = self.pred_cali_train + r_lcp_train
                else:
                    y_lower = self.pred_cali_train[:, 0] - r_lcp_train
                    y_upper = self.pred_cali_train[:, 1] + r_lcp_train
                coverage = compute_coverage(self.y_cali_train, y_lower, y_upper)
            k += 1
            # print(k, r_lcp_train[i], r_lcp[i], coverage)
        return k, coverage

    def train_conditional_calibration_one(self):
        """Training-conditional calibration of LCP-RF.

        Returns
        -------
        int
            Index of the correction term using discretization of (0,1)
        coverage
            The coverage reached with the second calibration data.
        """
        if self.estimator == 'clf':
            y_pred_set = np.greater_equal(self.pred_cali_proba_train, 1 - self.r_lcp_train_cali.reshape(-1, 1))
            coverage = compute_coverage_classification(y_pred_set, self.y_cali_train)
        else:
            if self.pred_cali.ndim == 1:
                y_lower = self.pred_cali_train - self.r_lcp_train_cali
                y_upper = self.pred_cali_train + self.r_lcp_train_cali
            else:
                y_lower = self.pred_cali_train[:, 0] - self.r_lcp_train_cali
                y_upper = self.pred_cali_train[:, 1] + self.r_lcp_train_cali
            coverage = compute_coverage(self.y_cali_train, y_lower, y_upper)

        r_lcp_train = np.zeros(shape=self.r_lcp_train_cali.shape)
        r_sort = np.sort(self.r_cali)

        k = 0
        while coverage < self.quantile:
            k += 1
            for i in range(self.x_cali_train.shape[0]):
                k_star = self.s_lcp_train_cali[i] + k
                if r_sort.shape[0] >= k_star > 0:
                    r_lcp_train[i] = r_sort[k_star - 1]
                elif k_star == 0:
                    r_lcp_train[i] = np.min(r_sort)
                else:
                    r_lcp_train[i] = np.max(r_sort)
            if self.estimator == 'clf':
                y_pred_set = np.greater_equal(self.pred_cali_proba_train, 1 - r_lcp_train.reshape(-1, 1))
                coverage = compute_coverage_classification(y_pred_set, self.y_cali_train)
            else:
                if self.pred_cali.ndim == 1:
                    y_lower = self.pred_cali_train - r_lcp_train
                    y_upper = self.pred_cali_train + r_lcp_train
                else:
                    y_lower = self.pred_cali_train[:, 0] - r_lcp_train
                    y_upper = self.pred_cali_train[:, 1] + r_lcp_train
                coverage = compute_coverage(self.y_cali_train, y_lower, y_upper)
        return k, coverage

    def train_conditional_calibration_bygroup(self):
        """Training-conditional calibration of groupwise LCP-RF.

        Returns
        -------
        int
            Index of the correction term using discretization of (0,1)
        coverage
            The coverage reached with the second calibration data.
        """
        y_cali_train_base = np.zeros(self.x_cali_train.shape[0])
        w_test = self.compute_forest_weights(self.x_cali_train, y_cali_train_base, self.x_cali, self.r_cali)
        groups = np.unique(self.communities)
        p_test = np.zeros(shape=(self.x_cali_train.shape[0], len(groups)))
        for group in groups:
            p_test[:, group] = np.sum(w_test[:, self.communities == group], axis=1)
        group_test = np.argmax(p_test, axis=1)

        if self.estimator == 'clf':
            y_pred_set = np.greater_equal(self.pred_cali_proba_train, 1 - self.r_lcp_bygroup.reshape(-1, 1))
            coverage = compute_coverage_classification(y_pred_set, self.y_cali_train)
        else:
            if self.pred_cali.ndim == 1:
                y_lower = self.pred_cali_train - self.r_lcp_bygroup
                y_upper = self.pred_cali_train + self.r_lcp_bygroup
            else:
                y_lower = self.pred_cali_train[:, 0] - self.r_lcp_bygroup
                y_upper = self.pred_cali_train[:, 1] - self.r_lcp_bygroup

            coverage = compute_coverage(self.y_cali_train, y_lower, y_upper)

        r_lcp_train = np.zeros(shape=self.r_lcp_bygroup.shape)

        k = 0
        while coverage < self.quantile:
            k += 1
            for i in range(self.x_cali_train.shape[0]):
                k_star = self.s_lcp_bygroup[i] + k
                r_cali_test = self.r_cali_bygroup[group_test[i]]
                if r_cali_test.shape[0] >= k_star > 0:
                    r_lcp_train[i] = np.sort(r_cali_test)[k_star - 1]
                elif k_star == 0:
                    r_lcp_train[i] = np.min(self.r_cali)
                else:
                    r_lcp_train[i] = np.max(self.r_cali)

            if self.estimator == 'clf':
                y_pred_set = np.greater_equal(self.pred_cali_proba_train, 1 - r_lcp_train.reshape(-1, 1))
                coverage = compute_coverage_classification(y_pred_set, self.y_cali_train)
            else:
                if self.pred_cali.ndim == 1:
                    y_lower = self.pred_cali_train - r_lcp_train
                    y_upper = self.pred_cali_train + r_lcp_train
                else:
                    y_lower = self.pred_cali_train[:, 0] - r_lcp_train
                    y_upper = self.pred_cali_train[:, 1] + r_lcp_train
                coverage = compute_coverage(self.y_cali_train, y_lower, y_upper)
        return k, coverage

    def compute_forest_weights(self, X, y, data, y_data):
        """Compute the weights of the RF.

        It compute the weights of X using "data" as background data.
        Parameters
        ----------
        X : numpy.array
            Input data
        y : numpy.array
            Input labels
        data : numpy.array
            Background data of the RF
        y_data : numpy.array
            Labels of the background data

        Returns
        -------
        numpy.array
            It return numpy.array w of shape (X.shape[0], data.shape[0]). w[i, j] corresponds to how many
            times data[j] falls in the same leaves as X[i].
        """
        X, y = check_X_y(X, y, dtype=np.double)
        data, y_data = check_X_y(data, y_data, dtype=np.double)
        y, y_data = as_float_array(y).astype(np.double), as_float_array(y_data).astype(np.double)
        self.check_is_acpi_fitted()
        w = cyext_acv.compute_forest_weights_verbose(X, y, data, y_data,
                                                     self.ACPI.features,
                                                     self.ACPI.thresholds,
                                                     self.ACPI.children_left,
                                                     self.ACPI.children_right,
                                                     self.min_node_size)

        return w

    def compute_forest_weights_cali(self, X, data, y_data, weights):
        """Compute the weights of the RF with X as background data.

        It compute the weights of X using "data" and X as background data.
        Parameters
        ----------
        X : numpy.array
            Input data
        y : numpy.array
            Input labels
        data : numpy.array
            Background data of the RF
        y_data : numpy.array
            Labels of the background data

        Returns
        -------
        numpy.array
            It return numpy.array w of shape (X.shape[0], data.shape[0]+1). w[i, j] corresponds to how many
            times data[j] falls in the same leaves as X[i] using data+X[i] as background data. w[-1, i] is the weights
            associate to X[i].
        """
        w = cyext_acv.compute_forest_weights_cali_verbose(X, data, y_data,
                                                          self.ACPI.features,
                                                          self.ACPI.thresholds,
                                                          self.ACPI.children_left,
                                                          self.ACPI.children_right,
                                                          self.min_node_size, weights)
        return w

    def predict(self, X):
        """
        Predict nonconformity score of the QRF.

        Parameters
        ----------
        X : numpy.array

        Returns
        -------
        numpy.array
            It returns mean estimates of the nonconformity score of X.
        """
        X = check_array(X)
        check_is_fitted(self.model,
                        msg="This ACPI instance is not fitted yet. Call 'fit' with appropriate arguments"
                            " before using this estimator")
        return self.model.predict(X)

    def check_is_acpi_fitted(self):
        """Check if ACPI estimator is fitted.

        """
        check_is_fitted(self.model,
                        msg="ACPI estimator instance is not fitted yet. Call 'fit_calibration' with appropriate arguments"
                            " before using this estimator")
        if not self.check_is_acpi_fit:
            self.ACPI = BaseAgnosTree(self.model, self.d)
            self.check_is_acpi_fit = True