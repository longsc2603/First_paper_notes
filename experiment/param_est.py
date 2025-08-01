"""
Utility functions to estimate parameters of the Ornstein–Uhlenbeck (OU)
process using various methods.  In the OU model

    dX_t = alpha * (mu - X_t) dt + sigma dW_t,

the parameters ``alpha`` (mean‑reversion rate), ``mu`` (long‑term mean) and
``sigma`` (volatility) can be estimated from discretely sampled time series
using ordinary least squares (OLS), maximum likelihood via a Kalman filter,
or an expectation–maximization (EM) algorithm that accounts for additive
measurement noise.  These routines accept a three‑dimensional input array
``X`` of shape ``(num_trajectories, num_steps, d)`` and return estimates
for each dimension in ``d``.

The implementations below favour clarity over efficiency; they are intended
for educational purposes and modest data sizes.
"""

import numpy as np
from typing import Tuple

try:
    from scipy.optimize import minimize
except ImportError:
    minimize = None  # type: ignore

from scipy.linalg import logm, solve, cholesky, expm, eigh, LinAlgError


def estimate_ou_ols(X: np.ndarray, delta_t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate parameters of a multivariate OU process using OLS.
    Works for arbitrary drift matrices by projecting complex values back to reals.
    
    Parameters
    ----------
    X : np.ndarray
        Array of shape (num_trajectories, num_steps, d)
    delta_t : float
        Time between observations

    Returns
    -------
    A : np.ndarray of shape (d, d)
        Real-valued drift matrix
    mu : np.ndarray of shape (d,)
        Long-term mean
    H : np.ndarray of shape (d, d)
        Real-valued diffusion matrix (lower triangular)
    """
    if X.ndim != 3:
        raise ValueError("X must be of shape (num_trajectories, num_steps, d)")
    
    num_traj, num_steps, d = X.shape
    if num_steps < 2:
        raise ValueError("Need at least two time steps.")

    # Stack time pairs
    X_t = X[:, :-1, :].reshape(-1, d)
    X_tp1 = X[:, 1:, :].reshape(-1, d)

    # Center the data
    X_t_mean = X_t.mean(axis=0, keepdims=True)
    X_tp1_mean = X_tp1.mean(axis=0, keepdims=True)
    X_t_centered = X_t - X_t_mean
    X_tp1_centered = X_tp1 - X_tp1_mean

    # Estimate discrete-time AR(1) transition matrix
    XtX = X_t_centered.T @ X_t_centered
    XtY = X_t_centered.T @ X_tp1_centered
    a = XtY @ np.linalg.pinv(XtX)  # shape (d, d)

    # Use full matrix logarithm
    log_a = logm(a)
    A = -log_a / delta_t

    # # If imaginary part is small, drop it
    # if np.max(np.abs(A.imag)) > 1e-6:
    #     raise RuntimeError("Matrix logarithm resulted in large imaginary values.")
    A = A.real

    # Estimate long-term mean mu: b = mu (I - a)
    b = X_tp1_mean.T - a @ X_t_mean.T
    mu = solve(np.eye(d) - a, b).flatten()

    # Residuals
    residuals = X_tp1 - (X_t @ a.T + mu)
    cov_eps = (residuals.T @ residuals) / residuals.shape[0]
    cov_eps = (cov_eps + cov_eps.T) / 2  # force symmetry

    # Ensure positive definite covariance for Cholesky
    try:
        H = cholesky(cov_eps, lower=True)
    except np.linalg.LinAlgError:
        jitter = 1e-5 * np.eye(d)
        H = cholesky(cov_eps + jitter, lower=True)

    return A, mu, H


def _kalman_loglik(params: np.ndarray, X_dim: np.ndarray, delta_t: float, measurement_var: float = 1e-8) -> float:
    """Compute the negative log‑likelihood of an OU model via a Kalman filter.

    This helper routine is used internally by :func:`estimate_ou_kalman`.  It
    treats the OU process as a linear Gaussian state‑space model with the
    state equation

        x_{t+1} = mu + (x_t − mu) e^{−alpha * delta_t} + w_t,

    where ``w_t`` ~ N(0, process_var) with

        process_var = sigma^2 (1 − e^{−2 alpha delta_t}) / (2 alpha),

    and the measurement equation y_t = x_t + v_t with v_t ~ N(0, measurement_var).
    The likelihood is summed across all trajectories passed in ``X_dim``.

    Parameters
    ----------
    params : np.ndarray
        Array [alpha, mu, sigma] containing parameters to evaluate.
    X_dim : np.ndarray
        A two‑dimensional array of shape (num_trajectories, num_steps)
        containing the data for a single spatial dimension.
    delta_t : float
        Sampling interval.
    measurement_var : float, optional
        Measurement noise variance.  A small positive value avoids
        numerical singularities when the observations are taken to be the
        latent process directly.

    Returns
    -------
    float
        The negative log‑likelihood of the data given the parameters.
    """
    alpha, mu, sigma = params
    # Enforce positivity of alpha and sigma
    if alpha <= 0 or sigma <= 0:
        return np.inf

    # Precompute constants
    exp_neg = np.exp(-alpha * delta_t)
    process_var = sigma ** 2 * (1.0 - np.exp(-2.0 * alpha * delta_t)) / (2.0 * alpha)

    total_loglik = 0.0
    # Iterate over trajectories
    for traj in range(X_dim.shape[0]):
        observations = X_dim[traj]
        # Initial state estimate and variance; assume stationary distribution
        x_est = observations[0]
        P = sigma ** 2 / (2.0 * alpha)
        # Accumulate log‑likelihood of the first observation
        S0 = P + measurement_var
        resid0 = observations[0] - x_est
        total_loglik += -0.5 * (np.log(2.0 * np.pi * S0) + resid0 ** 2 / S0)
        for t in range(1, observations.size):
            # Predict state and variance
            x_pred = mu + (x_est - mu) * exp_neg
            P_pred = exp_neg ** 2 * P + process_var
            # Update given observation
            S = P_pred + measurement_var
            resid = observations[t] - x_pred
            total_loglik += -0.5 * (np.log(2.0 * np.pi * S) + resid ** 2 / S)
            K = P_pred / S  # Kalman gain
            x_est = x_pred + K * resid
            P = (1.0 - K) * P_pred
    return -total_loglik


def safe_inv_sym(M, eps=1e-8):
    eigvals, eigvecs = eigh(M)
    eigvals = np.clip(eigvals, eps, None)
    return eigvecs @ np.diag(1.0 / eigvals) @ eigvecs.T

def safe_logdet_sym(M, eps=1e-8):
    eigvals = np.linalg.eigvalsh(M)
    eigvals = np.clip(eigvals, eps, None)
    return np.sum(np.log(eigvals))

def estimate_ou_kalman(X: np.ndarray, delta_t: float):
    """
    Estimate parameters of a multivariate OU process using Kalman filter likelihood.

    Parameters
    ----------
    X : np.ndarray
        Array of shape (num_trajectories, num_steps, d).
    delta_t : float
        Time step between samples.

    Returns
    -------
    A : np.ndarray (d x d) -- drift matrix
    mu : np.ndarray (d,)   -- long-term mean
    H : np.ndarray (d x d) -- diffusion matrix
    """
    n_traj, n_steps, d = X.shape

    # OLS initialization for F, mu, Q
    X_t = X[:, :-1, :].reshape(-1, d)
    X_tp1 = X[:, 1:, :].reshape(-1, d)
    X_mean = X_t.mean(axis=0)
    Y_mean = X_tp1.mean(axis=0)
    Xt_centered = X_t - X_mean
    Yt_centered = X_tp1 - Y_mean
    F0 = np.linalg.lstsq(Xt_centered, Yt_centered, rcond=None)[0].T
    mu0 = Y_mean - F0 @ X_mean
    resid = X_tp1 - (X_t @ F0.T + mu0)
    Q0 = np.cov(resid.T)
    chol_Q0 = np.linalg.cholesky(Q0 + 1e-8 * np.eye(d))
    x0 = np.concatenate([F0.ravel(), mu0, chol_Q0[np.tril_indices(d)]])

    def unpack_params(params):
        idx = 0
        F = params[idx:idx + d * d].reshape(d, d)
        idx += d * d
        mu = params[idx:idx + d]
        idx += d
        L = np.zeros((d, d))
        L[np.tril_indices(d)] = params[idx:]
        Q = L @ L.T
        return F, mu, Q

    def kalman_nll(params):
        F, mu, Q = unpack_params(params)
        R = 1e-8 * np.eye(d)  # Small observation noise for stability
        nll = 0.0
        for traj in X:
            x_pred = traj[0]
            P_pred = np.eye(d) * 1.0
            for t in range(1, n_steps):
                # Predict
                x_pred = mu + F @ (x_pred - mu)
                P_pred = F @ P_pred @ F.T + Q
                P_pred = (P_pred + P_pred.T) / 2

                y = traj[t]
                innov = y - x_pred
                S = P_pred + R
                S = (S + S.T) / 2
                S_inv = safe_inv_sym(S)
                logdet = safe_logdet_sym(S)
                nll += 0.5 * (logdet + innov.T @ S_inv @ innov + d * np.log(2 * np.pi))
                # Update
                K = P_pred @ S_inv
                x_pred = x_pred + K @ innov
                P_pred = (np.eye(d) - K) @ P_pred
        return nll

    res = minimize(
        kalman_nll,
        x0,
        method='L-BFGS-B',
        options={'maxiter': 300}
    )
    if not res.success:
        raise RuntimeError("Kalman MLE failed: " + res.message)
    F_opt, mu_opt, Q_opt = unpack_params(res.x)

    # Recover A from F = exp(-A Δt)
    A_est = -logm(F_opt) / delta_t
    A_est = A_est.real  # Discard tiny imaginary part
    # Diffusion matrix H from Q
    eigvals, eigvecs = eigh(Q_opt)
    eigvals = np.clip(eigvals, 0, None)
    H_est = eigvecs @ np.diag(np.sqrt(eigvals))

    return A_est, mu_opt, H_est


def _kalman_filter_smoother(
    y: np.ndarray,
    phi: float,
    c: float,
    Q: float,
    R: float,
    x0: float,
    P0: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run Kalman filter and RTS smoother for an AR(1) state‑space model.

    Parameters
    ----------
    y : np.ndarray
        1D array of observations.
    phi : float
        State transition coefficient (phi = e^{−alpha * delta_t}).
    c : float
        Drift term in the state equation.
    Q : float
        Process noise variance.
    R : float
        Measurement noise variance.
    x0 : float
        Initial state mean.
    P0 : float
        Initial state variance.

    Returns
    -------
    m_smooth : np.ndarray
        Smoothed state means (E[x_t | y_{1:T}]).
    P_smooth : np.ndarray
        Smoothed state variances (Var[x_t | y_{1:T}]).
    P_lag : np.ndarray
        Smoothed lag‑one covariances Cov(x_t, x_{t-1} | y_{1:T}) for t=1..T-1.

    Notes
    -----
    This routine follows the standard Rauch–Tung–Striebel (RTS) smoother
    algorithm.  It is used internally by the EM estimator.
    """
    T = y.size
    # Preallocate arrays
    m_pred = np.zeros(T)
    P_pred = np.zeros(T)
    m_filt = np.zeros(T)
    P_filt = np.zeros(T)
    # Initial prediction
    m_pred[0] = x0
    P_pred[0] = P0
    # Filter step
    for t in range(T):
        if t == 0:
            # Update at initial time
            S = P_pred[t] + R
            K = P_pred[t] / S
            m_filt[t] = m_pred[t] + K * (y[t] - m_pred[t])
            P_filt[t] = (1.0 - K) * P_pred[t]
        else:
            # Predict
            m_pred[t] = phi * m_filt[t - 1] + c
            P_pred[t] = phi * phi * P_filt[t - 1] + Q
            # Update
            S = P_pred[t] + R
            K = P_pred[t] / S
            m_filt[t] = m_pred[t] + K * (y[t] - m_pred[t])
            P_filt[t] = (1.0 - K) * P_pred[t]
    # RTS smoother
    m_smooth = np.zeros(T)
    P_smooth = np.zeros(T)
    P_lag = np.zeros(T - 1)
    m_smooth[-1] = m_filt[-1]
    P_smooth[-1] = P_filt[-1]
    for t in reversed(range(T - 1)):
        # Smoothing gain
        A = P_filt[t] * phi / P_pred[t + 1]
        m_smooth[t] = m_filt[t] + A * (m_smooth[t + 1] - m_pred[t + 1])
        P_smooth[t] = P_filt[t] + A * A * (P_smooth[t + 1] - P_pred[t + 1])
        P_lag[t] = A * P_smooth[t + 1]
    return m_smooth, P_smooth, P_lag


def estimate_ou_em(X, dt, n_iter=50, tol=1e-6):
    """
    EM algorithm for parameter estimation in a multivariate OU process
    for multiple trajectories.

    X: array, shape (num_trajectories, num_steps, d)
    dt: time step
    n_iter: max number of EM iterations
    tol: convergence tolerance

    Returns:
        mu: estimated mean vector (d,)
        A: estimated drift matrix (d, d)
        Sigma: estimated diffusion matrix (d, d)
    """
    num_trajectories, num_steps, d = X.shape

    # Stack all trajectories into one big sequence (ignore dependence for estimation)
    X_all = X.reshape(-1, d)         # (num_trajectories*num_steps, d)
    X0 = X[:, :-1, :].reshape(-1, d) # (num_trajectories*(num_steps-1), d)
    X1 = X[:, 1:, :].reshape(-1, d)
    dX = (X1 - X0) / dt              # increments

    # Initial guesses
    mu = X_all.mean(axis=0)
    A = np.eye(d)
    Sigma = np.cov(dX.T)

    for it in range(n_iter):
        # E-step (for full obs, this is just sufficient stats from data)
        X0c = X0 - mu   # centered data
        dXc = dX

        # M-step (least squares)
        A_new = np.linalg.lstsq(X0c, dXc, rcond=None)[0].T  # shape (d, d)
        mu_new = mu  # Stationary assumption

        residuals = dXc - X0c @ A_new.T
        Sigma_new = (residuals.T @ residuals) / (len(X0) - 1)

        # Convergence
        if (np.linalg.norm(A_new - A) < tol and
            np.linalg.norm(Sigma_new - Sigma) < tol and
            np.linalg.norm(mu_new - mu) < tol):
            break

        A, Sigma, mu = A_new, Sigma_new, mu_new

    return A, mu, Sigma


if __name__ == "__main__":
    # Example usage and simple sanity check
    # Generate synthetic OU data and verify that estimators recover parameters
    np.random.seed(42)
    num_traj, num_steps, d = 3, 200, 2
    delta_t = 0.1
    true_alphas = np.array([1.5, 0.8])
    true_mus = np.array([0.5, -1.0])
    true_sigmas = np.array([0.7, 0.4])
    # Simulate trajectories
    X = np.zeros((num_traj, num_steps, d))
    for dim in range(d):
        alpha = true_alphas[dim]
        mu = true_mus[dim]
        sigma = true_sigmas[dim]
        phi = np.exp(-alpha * delta_t)
        process_var = sigma ** 2 * (1.0 - np.exp(-2.0 * alpha * delta_t)) / (2.0 * alpha)
        for traj in range(num_traj):
            x = np.zeros(num_steps)
            x[0] = mu
            for t in range(1, num_steps):
                x[t] = mu + (x[t - 1] - mu) * phi + np.sqrt(process_var) * np.random.randn()
            X[traj, :, dim] = x
    # Apply estimators
    ols_alphas, ols_mus, ols_sigmas = estimate_ou_ols(X, delta_t)
    kal_alphas, kal_mus, kal_sigmas = estimate_ou_kalman(X, delta_t)
    em_alphas, em_mus, em_sigmas = estimate_ou_em(X, delta_t, max_iter=50, tol=1e-5)
    print("True alphas:", true_alphas)
    print("OLS alphas:", ols_alphas)
    print("Kalman alphas:", kal_alphas)
    print("EM alphas:", em_alphas)
    print("True mus:", true_mus)
    print("OLS mus:", ols_mus)
    print("Kalman mus:", kal_mus)
    print("EM mus:", em_mus)
    print("True sigmas:", true_sigmas)
    print("OLS sigmas:", ols_sigmas)
    print("Kalman sigmas:", kal_sigmas)
    print("EM sigmas:", em_sigmas)