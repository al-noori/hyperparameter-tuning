import numpy as np

from sklearn.utils import column_or_1d, check_consistent_length
from scipy.special import xlogy


def zero_one_loss(y_true, y_pred):
    """
    Computes the empirical risk for the zero-one loss function.

    Parameters
    ----------
    y_true : array-like of shape (n_labels,)
        True class labels as array-like object.
    y_pred : array-like of shape (n_labels,)
        Predicted class labels as array-like object.

    Returns
    -------
    risk : float in [0, 1]
        Empirical risk computed via the zero-one loss function.
    """
    y_true = column_or_1d(y_true)
    y_pred = column_or_1d(y_pred)
    check_consistent_length(y_true, y_pred)

    # Compute and return the empirical risk.
    risk = np.mean(y_true != y_pred) # <-- SOLUTION
    return risk # <-- SOLUTION


def binary_cross_entropy_loss(y_true, y_pred):
    """
    Computes the empirical risk for the binary cross entropy (BCE) loss function.

    Parameters
    ----------
    y_true : array-like of shape (n_labels,)
        True conditional class probabilities as array-like object.
    y_pred : array-like of shape (n_labels,)
        Predicted conditional class probabilities as array-like object.

    Returns
    -------
    risk : float in [0, +infinity]
        Empirical risk computed via the BCE loss function.
    """
    y_true = column_or_1d(y_true, dtype=float)
    y_pred = column_or_1d(y_pred, dtype=float)

    # Check value ranges of probabilities and raise ValueError if the ranges are invalid. In this case, it should be
    # allowed to have estimated probabilities in the interval [0, 1] instead of only (0, 1).
    # BEGIN SOLUTION
    for name, y in [("y_true", y_true), ("y_pred", y_pred)]:
        if np.min(y) < 0 or np.max(y) > 1:
            raise ValueError(f"`{name}` must have values in the interval [0, 1].")
    # END SOLUTION

    # Compute and return the empirical risk.
    risk = (- xlogy(y_true, y_pred) - xlogy(1 - y_true, 1 - y_pred)).mean() # <-- SOLUTION
    return risk # <-- SOLUTION