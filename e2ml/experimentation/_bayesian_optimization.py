import numpy as np

from copy import deepcopy
from scipy.stats import norm
from sklearn.utils import check_consistent_length, column_or_1d, check_scalar, check_array

from ..models import GaussianProcessRegression


def acquisition_pi(mu, sigma, tau):
    """
    Computes probability improvement scores.

    Parameters
    ----------
    mu : array-like of shape (n_samples,)
        Mean predictions.
    sigma : array-like of shape (n_samples,)
        Standard deviations of mean predictions.
    tau : float
        Reference value for improvement computation.

    Returns
    -------
    pi_scores : numpy.ndarray of shape (n_samples,)
        Computed probability improvement scores.
    """
    # Check parameters.
    check_scalar(tau, name='tau', target_type=float)
    mu = column_or_1d(mu)
    sigma = column_or_1d(sigma)
    check_consistent_length(mu, sigma)

    # Compute and return probability improvement as `pi_scores`.
    # BEGIN SOLUTION
    z = (mu - tau) / sigma
    pi_scores = norm.cdf(z)
    return pi_scores
    # END SOLUTION


def acquisition_ei(mu, sigma, tau):
    """
    Computes expected improvement scores.

    Parameters
    ----------
    mu : array-like of shape (n_samples,)
       Mean predictions.
    sigma : array-like of shape (n_samples,)
       Standard deviations of mean predictions.
    tau : float
       Reference value for improvement computation.

    Returns
    -------
    ei_scores : numpy.ndarray of shape (n_samples,)
       Computed expected improvement scores.
    """
    # Check parameters.
    check_scalar(tau, name='tau', target_type=float)
    mu = column_or_1d(mu)
    sigma = column_or_1d(sigma)
    check_consistent_length(mu, sigma)

    # Compute and return probability improvement as `ei_scores`.
    # BEGIN SOLUTION
    z = (mu - tau) / sigma
    term_1 = (mu - tau) * norm.cdf(z)
    term_2 = sigma * norm.pdf(z)
    ei_scores = term_1 + term_2
    return ei_scores
    # END SOLUTION


def acquisition_ucb(mu, sigma, kappa):
    """
    Computes upper confidence bound scores.

    Parameters
    ----------
    mu : array-like of shape (n_samples,)
       Mean predictions.
    sigma : array-like of shape (n_samples,)
       Standard deviations of mean predictions.
    kappa : float
       Factor for sigma.

    Returns
    -------
    ucb_scores : numpy.ndarray of shape (n_samples,)
       Computed upper confidence bound scores.
    """
    # Check parameters.
    check_scalar(kappa, name='kappa', min_val=0, target_type=float)
    mu = column_or_1d(mu)
    sigma = column_or_1d(sigma)
    check_consistent_length(mu, sigma)

    # Compute and return probability improvement as `ucb_scores`.
    # BEGIN SOLUTION
    ucb_scores = mu + kappa * sigma
    return ucb_scores
    # END SOLUTION


def perform_bayesian_optimization(X_cand, gpr, acquisition_func, obj_func, n_evals, n_random_init, seed=None):
    """
    Perform Bayesian optimization according to a specified acquisition function for given Gaussian
    process model, objective function, and maximum number of function evaluations.

    Parameters
    ----------
    X_cand : array-like of shape (n_samples, n_features)
        Candidate samples that can be selected for function evaluation.
    gpr : e2ml.models.GaussianProcessRegression
        Gaussian process as surrogate probabilistic model.
    acquisition_func : 'pi' or 'ei' or 'ucb'
        Specifies one of the three available acquisition functions for selecting samples.
    obj_func : callable
        Takes samples of `X_cand` as input to evaluate objective values.
    n_evals : int
        Number of samples to be acquired, i.e., selected for evaluation.
    n_random_init : int
        Number of samples to be randomly acquired for initialization. Subsequently, the acquisition
        function will be used to select samples.

    Returns
    -------
    X_acquired : numpy.ndarray (n_evals, n_features)
        Acquired, i.e., selected for evaluation, samples.
    y_acquired : numpy.ndarray (n_evals,)
        Obtained objective function values for acquired samples.
    
    """
    # Check parameters.
    if not isinstance(gpr, GaussianProcessRegression):
        raise TypeError('`gpr` must be a `e2ml.models.GaussianProcessRegression` instance.')
    gpr = deepcopy(gpr)
    if not callable(obj_func):
        raise TypeError('`obj_func` must be a callable.')
    if not acquisition_func in ['pi', 'ei', 'ucb']:
        raise ValueError("`acquisition_func` must be in `['pi', 'ei', 'ucb']`.")
    X_cand = check_array(X_cand)
    check_scalar(
        n_evals, name='n_evals', target_type=int, min_val=1, max_val=len(X_cand)-1
    )
    check_scalar(
        n_evals, name='n_random_init', target_type=int, min_val=1, max_val=len(X_cand)-1
    )

    # Perform Bayesian optimization until `n_evals` have been performed.
    # BEGIN SOLUTION
    is_acquired = np.zeros(len(X_cand), dtype=bool)
    X_acquired = []
    y_acquired = []
    for i in range(n_evals):
        if i < n_random_init:
            if seed is not None:
                np.random.seed(seed + i )
                selected_idx = np.random.choice(np.arange(len(X_cand)), size=1)[0]
            else:
                selected_idx = np.random.choice(np.arange(len(X_cand)), size=1)[0]
        else:
            gpr.fit(X_acquired, y_acquired)
            mu, sigma = gpr.predict(X_cand[~is_acquired], return_std=True)
            tau = np.max(y_acquired)
            if acquisition_func == 'pi':
                scores = acquisition_pi(mu, sigma, tau)
            elif acquisition_func == 'ei':
                scores = acquisition_ei(mu, sigma, tau)
            elif acquisition_func == 'ucb':
                scores = acquisition_ucb(mu, sigma, kappa=1.)
            selected_idx = np.argmax(scores)
            not_acquired_indices = np.argwhere(is_acquired == 0).ravel()
            selected_idx = not_acquired_indices[selected_idx]
        is_acquired[selected_idx] = True
        X_acquired.append(X_cand[selected_idx])
        y_acquired.append(obj_func(X_cand[selected_idx])[0])
    return np.array(X_acquired), np.array(y_acquired)
    # END SOLUTION
