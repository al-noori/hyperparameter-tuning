import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import check_array, column_or_1d, check_consistent_length, check_scalar

from sklearn.metrics.pairwise import pairwise_kernels


class GaussianProcessRegression(BaseEstimator, RegressorMixin):
    """GaussianProcessRegression

    Parameters
    ----------
    metric_dict : dict or None, default=None
        Keyword arguments forwarded to `pairwise_kernels`.
        If None, an RBF kernel is used with `gamma = 1.0 / n_features`.
    beta : float
        Noise precision (i.e. observation noise variance = 1/beta).

    Attributes
    ----------
    X_: numpy.ndarray, shape (n_samples, n_features)
        The sample matrix `X_` is the feature matrix representing the training samples.
    y_: array-like, shape (n_samples) or (n_samples, n_outputs)
        The array `y_` contains the class labels of the training samples.

    References
    ----------
    [1] O. Chapelle, "Active Learning for Parzen Window Classifier",
        Proceedings of the Tenth International Workshop Artificial Intelligence and Statistics, 2005.
    """

    def __init__(self, beta=1.e-3, metrics_dict=None):
        self.beta = beta
        self.metrics_dict = metrics_dict

    def fit(self, X, y):
        """
        Fit the model using `X` as training data and `y` as class labels.

        Parameters
        ----------
        X: matrix-like, shape (n_samples, n_features)
            The sample matrix `X` is the feature matrix representing the samples for training.
        y: array-like, shape (n_samples) or (n_samples, n_outputs)
            The array `y` contains the class labels of the training samples.

        Returns
        -------
        self: GaussianProcessRegression,
            The `GaussianProcessRegression` is fitted on the training data.
        """
        # Check attributes and parameters.
        check_scalar(self.beta, min_val=0, target_type=float, name='beta')
        self.metrics_dict_ = {metric: "rbf"} if self.metrics_dict is None else dict(self.metrics_dict)
        self.X_ = check_array(X)
        self.y_ = column_or_1d(y)
        check_consistent_length(self.X_, self.y_)
        self._check_n_features(self.X_, reset=True)

        # Compute matrix `C_N` using the function `pairwise_kernels` with
        # `self.metric_dict_` as its parameters.
        K = pairwise_kernels(self.X_, **self.metrics_dict_) # <-- SOLUTION
        C_N = K + self.beta**(-1) * np.eye(len(K)) # <-- SOLUTION

        # Compute inverse `self.C_N_inv_` of matrix `C_N`.
        self.C_N_inv_ = np.linalg.pinv(C_N) # <-- SOLUTION

        return self

    def predict(self, X, return_std=False):
        """
        Return class label predictions for the test data X.

        Parameters
        ----------
        X:  array-like, shape (n_samples, n_features) or shape (n_samples, m_samples) if metric == 'precomputed'
            Test samples.

        Returns
        -------
        y:  numpy.ndarray, shape = [n_samples]
            Predicted class labels class.
        """
        # Check parameters.
        X = check_array(X)
        self._check_n_features(X, reset=False)
        return_std = bool(return_std)

        # Compute Gram matrix `K` between `X` and `self.X_` using the function
        # `pairwise_kernels` with `self.metric_dict_` as its parameters.
        K = pairwise_kernels(X, self.X_, **self.metrics_dict_) # <-- SOLUTION

        # Compute mean predictions `means` for samples `X`.
        means = K @ self.C_N_inv_ @ self.y_ # <-- SOLUTION

        if return_std:
            # Compute standard deviations `stds` for predicted means.
            c = np.diag(pairwise_kernels(X, **self.metrics_dict_)) + self.beta**(-1) # <-- SOLUTION
            stds = np.sqrt(c - np.diag(K @ self.C_N_inv_ @ K.T)) # <-- SOLUTION
            return means, stds
        else:
            return means
