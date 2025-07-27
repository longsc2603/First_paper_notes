import numpy as np
import matplotlib.pyplot as plt
import argparse
import scipy
import scipy.linalg
import time
import random
from sklearn.impute import KNNImputer
from numpy.lib.stride_tricks import sliding_window_view

from experiments import generate_independent_points, linear_additive_noise_data


def data_masking(
    data: np.ndarray,
    p: float = 0.2,
    known_initial_value: bool = False,
    random_seed: int = 167
    ) -> np.ndarray:
    """Randomly mask the input data, such that for each period of time there is
    at least one trajectory with data present there.

    Args:
        data (np.ndarray): Input data with shape
            (num_trajectories, num_steps, d).
        p (float, optional): Number of trajectories masked at each segment.
            Defaults to 0.2.
        known_initial_value (bool, optional): If True, the initial steps of trajectories are fixed.
            Defaults to False.
        random_seed (int, optional): Random seed for reproducibility. Defaults to 167.

    Returns:
        np.ndarray: Randomly-masked data with default masking value np.nan.
    """
    random.seed(random_seed)
    masked_data = np.copy(data)
    if known_initial_value:
        start = 1
    else:
        start = 0
    for segment in range(start, data.shape[1]):
        # number of trajectories being masked at each segment
        num_rows_to_mask = int(data.shape[0] * p)
        rows_to_be_masked = random.sample(range(data.shape[0]), num_rows_to_mask)
        for row in rows_to_be_masked:
            masked_data[row, segment, :] = np.nan
        
    # Store the missing indices
    missing_indices = np.isnan(masked_data)

    return masked_data, missing_indices


# Temporal KNN imputation with sliding windows
def knn_impute_with_temporal_context(data_missing, window_size=3, n_neighbors=3):
    """Impute missing values in the data using KNN with temporal context.

    Args:
        data_missing (np.ndarray): Input data with shape (num_trajectories, num_segments, step_per_segment, d)
        window_size (int, optional): Sliding window size. Defaults to 5.
        n_neighbors (int, optional): Number of neighbors for KNN algorithm. Defaults to 5.

    Returns:
        nd.ndarray: Imputed data with the same shape as input.
    """
    # Ensure window size is odd to have symmetric context
    assert window_size % 2 == 1, "Window size must be odd."

    if data_missing.ndim == 3:
        num_traj, num_step, d = data_missing.shape
    else:
        data_missing = np.reshape(data_missing,
                        (data_missing.shape[0], data_missing.shape[1]*data_missing.shape[2], data_missing.shape[3]))
        num_traj, num_step, d = data_missing.shape
    imputed_data = np.empty_like(data_missing)

    for i in range(num_traj):
        traj = data_missing[i]  # shape (num_step, d)
        pad = window_size // 2
        padded = np.pad(traj, ((pad, pad), (0, 0)), constant_values=np.nan)
        windows = sliding_window_view(padded, (window_size, d))[:, 0, :, :]  # shape (num_step, window_size, d)
        flat_windows = windows.reshape(num_step, window_size * d)

        imputer = KNNImputer(n_neighbors=n_neighbors)
        imputed_windows = imputer.fit_transform(flat_windows)

        # Extract center of each window (the timestep we're imputing)
        center_vals = imputed_windows[:, (window_size // 2) * d : (window_size // 2 + 1) * d]
        imputed_data[i] = center_vals

    return imputed_data


def dag_penalty(A: np.ndarray, alpha: float = 0.1) -> float:
    """NOTEARS penalty for adjacency matrix A:
    h(A) = trace(e^(A◦A)) - d, where d is the dimension (number of variables).

    If h(A) = 0, the graph is acyclic. Otherwise, h(A) > 0.

    Args:
        A (np.ndarray): Drift matrix (adjacency matrix).
        alpha (float): Regularization hyper-parameter. Defaults to 0.1.

    Returns:
        float: Penalty to add to the loss.
    """
    d = A.shape[0]
    # A◦A means elementwise multiplication
    # e^(A◦A) is matrix exponential of the elementwise product
    # but in the standard NOTEARS approach, it's actually the matrix exponential
    # of the adjacency matrix. A direct approach for the penalty often uses
    # repeated series expansions or a specialized algorithm.
    #
    # For simplicity, we do a rough approximation: compute exp of each element
    # then sum the diagonal. This is not the full matrix exponential approach
    # from the original NOTEARS paper, but it’s a practical placeholder.

    # "A ◦ A" is just A^2 elementwise
    A_sq = np.multiply(A, A)
    # Exponentiate elementwise
    exp_A_sq = scipy.linalg.expm(A_sq)
    trace_val = np.trace(exp_A_sq)
    penalty = alpha*(trace_val - d)

    return penalty


# TODO: Try some other convergence scores besides NLL
def compute_nll(X: np.ndarray, A: np.ndarray, H: np.ndarray, dt: float) -> float:
    """Compute the negative log-likelihood of the current segment under the
    current inferred model parameters.

    Args:
        X (np.ndarray): Segments, shape (num_time_steps, d)
        A (np.ndarray): Estimated drift matrix, shape (d, d)
        H (np.ndarray): Estimated (observational) diffusion matrix, shape (d, m), H = G@G.T
        dt (float): Time step size

    Returns:
        float: Total NLL of the segment
    """
    nll = 0.0
    if len(X.shape) == 3:
        # this is the case of shape (num_segments, steps_per_segment, d)
        X = np.reshape(X, (X.shape[0]*X.shape[1], X.shape[2]))
    num_steps, d = X.shape
    H_dt = H * dt
    # quick fix, instead of getting logdet of H_dt, which
    # is -inf due to det(H_dt) = 0, we get logdet of H_dt + eps*I
    # This ensures that H_dt_reg = H + eps*I is full rank and invertible.
    eps = 1e-2
    H_dt_reg = H_dt + eps*np.eye(H_dt.shape[0])
    # Precompute inverse and determinant of H_dt
    H_dt_inv = np.linalg.pinv(H_dt)
    sign, logdet_H_dt = np.linalg.slogdet(H_dt_reg)
    const_term = 0.5 * (d * np.log(2 * np.pi) + logdet_H_dt)
    for t in range(num_steps - 1):
        X_t = X[t]
        X_tp1 = X[t + 1]
        # Compute mean: mu_t = e^{A dt}  X_t (approximate if needed)
        # mu_t = np.exp(A*dt)@X_t
        mu_t = X_t + A @ X_t * dt  # Using Euler-Maruyama approximation
        diff = X_tp1 - mu_t
        exponent = 0.5 * diff.T @ H_dt_inv @ diff
        nll += exponent + const_term

    return nll


def compute_nlp(A: np.ndarray, H: np.ndarray, nll: float) -> float:
    """Compute the negative log-posterior (NLP) of the current segment under the
    current inferred model parameters. Based on Bayes's theorem, posterior is proportional
    to likelihood*prior. Thus, minimizing NLP would be the same as minimizing -(NLL + log_prior).
    We assume that the prior on A is Gaussian and the prior on H is improper p(H) ~ 1/H ->
    log p(H) = -log p(H)

    Args:
        A (np.ndarray): Estimated drift matrix, shape (d, d)
        H (np.ndarray): Estimated (observational) diffusion matrix, shape (d, m), H = G@G.T
        nll (float): Negative Log-likelihood.

    Returns:
        float: Total NLP of the segment
    """
    # Enforce positivity constraint on diagonal elements of H
    if any(H[i, i] <= 0 for i in range(H.shape[0])):
        return np.inf

    # --------- Log-Prior on A -----------
    # A ~ N(0, prior_std_a^2)
    # => log p(a) = -0.5 [ a^2 / (prior_std_a^2) + ln(2π (prior_std_a^2)) ]
    prior_var = 10
    log_prior_A = -0.5 * (
        (A**2) / (prior_var)
        + np.log(2 * np.pi * (prior_var)))

    # --------- Log-Prior on H ---------
    # H ~ N(0, prior_std_a^2)
    # => log p(H) = -0.5 [ H^2 / (prior_std_H^2) + ln(2π (prior_std_H^2)) ]
    log_prior_H = -0.5 * (
        (H**2) / (prior_var)
        + np.log(2 * np.pi * (prior_var)))

    # Combine them
    log_prior = np.sum(log_prior_A) + np.sum(log_prior_H)

    return nll - log_prior


def cal_error(drift, H, score, n, d):
    drift = np.full((n, d), drift)
    error = np.mean(np.square(drift - score @ H))

    return error


def estimate_mean_cov(X, noisy_measurements_sigma=0):
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError("X must be a 2-D array with shape (N, d)")

    # ---- mean ------------
    m_hat = X.mean(axis=0)

    # ---- covariance ------
    # Unbiased sample covariance (divides by N-1)
    C_hat = np.cov(X, rowvar=False, bias=False) - np.eye(X.shape[1]) * noisy_measurements_sigma**2

    return m_hat, C_hat


def cal_score(A, H, x_t, t, x0, noisy_measurements_sigma=0):
    # d = A.shape[0]
    # # 1) mean m_t
    # m_t = x0 @ scipy.linalg.expm(A * t)

    # # 2) build the 2d×2d block matrix
    # #      G = [ A      H
    # #            0   -A^T ]
    # G = np.block([
    #     [A,        H       ],
    #     [np.zeros((d,d)), -A.T]
    # ])

    # # 3) compute expm(G t)
    # M = scipy.linalg.expm(G * t)

    # # 4) extract blocks
    # M12 = M[:d,    d:]   # top-right block = C(t)
    # M22 = M[d:,    d:]   # bottom-right block = D(t)

    # # 5) P_t = C(t) @ D(t)^{-1}
    # P_t = M12.dot(scipy.linalg.inv(M22))

    m_t, P_t = estimate_mean_cov(x_t, noisy_measurements_sigma=noisy_measurements_sigma)
    try:
        inv_P = np.linalg.inv(P_t)
    except np.linalg.LinAlgError:
        # If P_t is singular, use pseudo-inverse
        inv_P = np.linalg.pinv(P_t)

    score = (x_t - m_t) @ (-inv_P)

    return score


def reorder_step_by_step(
    data: np.ndarray,
    order: np.ndarray,
    A: np.ndarray,
    H: np.ndarray,
    time_step_size: float,
    nll: float,
    known_initial_value: bool = False,
    use_quick_sort: bool = False,
    noisy_measurements_sigma: float = 0,
    missing_indices: np.ndarray = None
    ) -> np.ndarray:
    """Reordering trajectory data to maximize our objective (log-likelihood minus DAG penalty),
    which is minimizing (negative log-likelihood plus DAG penalty)

    Args:
        data (np.ndarray): Arrays of trajectory to be re-ordered,
            shape (num_trajectories, num_segments, points_per_segment, d).
        order (np.ndarray): Indices of data order, shape (num_steps,).
        A (np.ndarray): Current estimated drift matrix, shape (d, d).
        H (np.ndarray): Current estimated (observational) diffusion matrix, shape (d, d).
        time_step_size (float): Step size with respect to time between points.
        alpha (float, optional): Regularization hyper-parameter for DAG penalty. Defaults to 0.1.
        beta (float, optional): Penalization hyper-parameter used in calculating loss of ordering.
            Loss = NLL + DAG_penalty + beta*Distance_cost
        known_initial_value (bool, optional): If True, the initial values of trajectories
            are known beforehand, and should stay fixed. Defaults to False.
        noisy_measurements_sigma (float, optional): Standard deviation of the Gaussian noise.
            Defaults to 0.
        missing_indices (np.ndarray, optional): Indices of the missing data in the input.
            Defaults to None.

    Returns:
        np.ndarray: A re-ordered array of segments.
    """
    num_trajectories, num_steps, d = data.shape
    
    if known_initial_value:
        start = 1
    else:
        start = 0
    
    x0 = data[:, 0, :].reshape(num_trajectories, d)
    end = num_steps - 1
    temp_data = np.copy(data)
    A_temp, H_temp = A, H
    if use_quick_sort:
        pass
        # s = time.time()
        # quick_sort(temp_data[:, 1:, :], A_temp, H_temp, time_step_size, x0)
        # print("Time taken for quick sort:", time.time() - s)
    else:
        s = time.time()
        while end > start:
            for i in range(start, end):
                j = i + 1
                x_i = temp_data[:, i, :].reshape(num_trajectories, d)
                x_j = temp_data[:, j, :].reshape(num_trajectories, d)
                score_i = cal_score(A=A_temp, H=H_temp, x_t=x_i, t=time_step_size*(i+1),
                                    x0=x0, noisy_measurements_sigma=noisy_measurements_sigma)
                score_j = cal_score(A=A_temp, H=H_temp, x_t=x_j, t=time_step_size*(j+1),
                                    x0=x0, noisy_measurements_sigma=noisy_measurements_sigma)
                drift = np.mean((x_j - x_i)/time_step_size, axis=0)
                err_i = cal_error(drift=drift, H=H_temp, score=score_i, n=num_trajectories, d=d)
                err_j = cal_error(drift=drift, H=H_temp, score=score_j, n=num_trajectories, d=d)

                if err_i < err_j:
                    temp = np.copy(temp_data[:, i, :])
                    temp_data[:, i, :] = np.copy(temp_data[:, j, :])
                    temp_data[:, j, :] = np.copy(temp)
                    # change indices of order
                    temp = np.copy(order[i])
                    order[i] = np.copy(order[j])
                    order[j] = np.copy(temp)

                    if missing_indices is not None:
                        # If there are missing indices, we need to update them
                        temp = np.copy(missing_indices[:, i, :])
                        missing_indices[:, i, :] = np.copy(missing_indices[:, j, :])
                        missing_indices[:, j, :] = np.copy(temp)

                if i == end - 1:
                    end = i
        print("Time taken for bubble sort:", time.time() - s)

    return temp_data, order, missing_indices


def left_Var_Equation(A1, B1):
    """Stable solver for np.matmul(sum_Edxt_Ext, np.linalg.pinv(sum_Ext_ExtT)) * (1 / dt)
    via least squares formulation of XA = B  <=> A^T X^T = B^T.
    This function is from the APPEX paper.
    """
    m = B1.shape[0]
    n = A1.shape[0]
    X = np.zeros((m, n))
    for i in range(m):
        X[i, :] = np.linalg.lstsq(np.transpose(A1), B1[i, :], rcond=None)[0]
    return X


def update_sde_parameters(X, dt, T, pinv=False):
    """Calculate the approximate closed form estimator A_hat for
    time homogeneous linear drift from multiple trajectories.
    This function is from the APPEX paper.

    Args:
        X (numpy.ndarray): 3D array (num_trajectories, num_steps, d),
            where each slice corresponds to a single trajectory.
        dt (float): Discretization time step.
        T (float): Total time period.
        pinv: whether to use pseudo-inverse. Otherwise, we use left_Var_Equation.
            Defaults to False.

    Returns:
        numpy.ndarray: Estimated drift-diffusion matrices A, H given the set of trajectories
    """
    num_trajectories, num_steps, d = X.shape
    sum_Edxt_Ext = np.zeros((d, d))
    sum_Ext_ExtT = np.zeros((d, d))
    for t in range(num_steps - 1):
        sum_dxt_xt = np.zeros((d, d))
        sum_xt_xt = np.zeros((d, d))
        for n in range(num_trajectories):
            xt = X[n, t, :]
            dxt = X[n, t + 1, :] - X[n, t, :]
            sum_dxt_xt += np.outer(dxt, xt)
            sum_xt_xt += np.outer(xt, xt)
        sum_Edxt_Ext += sum_dxt_xt / num_trajectories
        sum_Ext_ExtT += sum_xt_xt / num_trajectories

    # Add a small regularization term I*epsilon to ensure full-rank
    eps = 1e-4
    sum_Ext_ExtT_reg = sum_Ext_ExtT + eps*np.eye(d)

    if pinv:
        A_est = np.matmul(sum_Edxt_Ext, np.linalg.pinv(sum_Ext_ExtT_reg)) * (1 / dt)
    else:
        A_est = left_Var_Equation(sum_Ext_ExtT_reg, sum_Edxt_Ext * (1 / dt))

    # Initialize the GG^T matrix
    GGT = np.zeros((d, d))

    if A_est is None:
        # Compute increments ΔX for each trajectory (no drift adjustment)
        increments = np.diff(X, axis=1)
    else:
        # Adjust increments by subtracting the deterministic drift: ΔX - A * X_t * dt
        increments = np.diff(X, axis=1) - dt * np.einsum('ij,nkj->nki', A_est, X[:, :-1, :])

    # Sum up the products of increments for each dimension pair across all trajectories and steps
    for i in range(d):
        for j in range(d):
            GGT[i, j] = np.sum(increments[:, :, i] * increments[:, :, j])

    # Divide by total time T*num_trajectories to normalize
    GGT /= T * num_trajectories

    return A_est, GGT


def check_convergence(score_list: list, score_type: str='NLL') -> bool:
    """Check if convergence happened, stop the algorithm if True.

    Args:
        score_list (list): List of all scores through epochs
        score_type (str, optional): Type of score, currently only 'NLL'. Defaults to 'NLL'.

    Returns:
        bool: True if converged, False otherwise.
    """
    if len(score_list) == 0:
        return False
    if score_type == 'NLL':
        if len(score_list) < 5:
            return False
        if len(score_list) >= 5 and score_list[-1] == score_list[-3] and score_list[-2] == score_list[-4] \
            and score_list[-1] < score_list[-2]:
            # for cases with fluctuating NLL scores (good - bad - good - bad - good)
            return True
        elif score_list[-1] < 0:
            good_iterations = sum(1 for i in score_list if i < 0)
            if good_iterations >= 5:
                return True
        diff = [score_list[id].item() - score_list[id+1].item() for id in range(len(score_list)-1)]
        if np.sum(diff[-5:]) < 1:
            return True
    else:
        pass

    return False


def check_sorting_accuracy(
    original_data: np.ndarray,
    reordered_data: np.ndarray,
    check_by_indices_order: bool = True,
    reordered_order: np.ndarray = None
    ) -> float:
    """Check the sorting accuracy of the reordered data by comparing it to the original data.
    Args:
        original_data (np.ndarray): Original data, shape (num_trajectories, num_steps, d).
        reordered_data (np.ndarray): Reordered data, shape (num_trajectories, num_steps, d).
        check_by_indices_order (bool, optional): If True, check the order of indices for calculating accuracy.
            If False, check the values of the data instead, this option should not be used in case of
            missing data or noisy measurements. Defaults to True.
        reordered_order (np.ndarray, optional): Reordered order of indices, shape (num_steps,).
    Raises:
        AssertionError: If check_by_indices_order is True and reordered_order is None.

    Returns:
        float: Percentage of correctly sorted points per trajectory.
    """
    if reordered_data.ndim == 4:
        reordered_data = np.reshape(reordered_data, (reordered_data.shape[0],
                                    reordered_data.shape[1]*reordered_data.shape[2], reordered_data.shape[3]))
    num_trajectories, num_steps, d = reordered_data.shape
    if check_by_indices_order:
        assert reordered_order is not None, "reordered_order must be provided if check_by_indices_order is True."
        # this is for data that is randomized by time steps through all trajectories
        original_order = np.arange(num_steps)
        right_count = 0
        for i in range(len(reordered_order)):
            if original_order[i] == reordered_order[i]:
                right_count += 1
        right_percent = right_count / num_steps
    else:
        right_percent_through_traj = []
        for traj in range(num_trajectories):
            right_value = ((original_data[traj] - reordered_data[traj]) == 0.0).astype(int)
            right_value_count = (right_value == 1).sum()
            right_value_count /= num_steps*d
            right_percent_through_traj.append(right_value_count)
        right_percent = np.mean(right_percent_through_traj)
    
    return right_percent


def estimate_sde_parameters(
    data: np.ndarray,
    original_data: np.ndarray,
    reordered_order: np.ndarray,
    time_step_size: float,
    T: float,
    true_A: np.ndarray,
    true_H: np.ndarray,
    max_iter: int = 10,
    alpha: float = 0.1,
    known_initial_value: bool = False,
    noisy_measurements_sigma: float = 0,
    data_missing_percent: float = 0.0
    ) -> tuple[np.ndarray, np.ndarray]:
    """Main iterative scheme:
    1) Reconstruct / reorder segments using the current (A, G).
    2) Re-estimate A, G from the newly-reconstructed data.

    Args:
        data (np.ndarray): Input data, shape (num_trajectories, num_steps, d)
        original_data (np.ndarray): Original data, shape (num_trajectories, num_steps, d).
        reordered_order (np.ndarray): Reordered order of indices, shape (num_steps,).
        time_step_size (float): Step size with respect to time between points.
        T (float): Time period.
        true_A (nd.ndarray): True drift matrix A. Used for computing MAE.
        true_H (nd.ndarray): True (observational) diffusion matrix H (= G@G.T). Used for computing MAE.
        max_iter (int, optional): Maximum number of iterations. Defaults to 10.
        alpha (float, optional): Regularization hyper-parameter. Defaults to 0.1.
        known_initial_value (bool, optional): If True, the initial values of trajectories
            are known beforehand, and should stay fixed. Defaults to False.
        noisy_measurements_sigma (float, optional): Standard deviation of the Gaussian noise.
            Defaults to 0.
        data_missing_percent (float, optional): Percentage of missing data in the input.
            Defaults to 0.0.

    Returns:
        tuple[np.ndarray, np.ndarray]: The estimated drift-diffusion matrices (A, H = G@G.T),
            the reordered data, the best ordered data, and lists of scores
            (NLL, NLP, MAE_A, MAE_H, right_percent).
    """
    num_trajectories, num_steps, d = data.shape

    if data_missing_percent > 0:
        # If data is missing, we need to impute it first
        data, missing_indices = data_masking(data, p=data_missing_percent,
                                             known_initial_value=known_initial_value)
        start = time.time()
        data = knn_impute_with_temporal_context(data)
        print("Time taken for KNN imputation:", time.time() - start)
    else:
        missing_indices = None
        
    reordered_data = np.copy(data)

    # 2) Iterative scheme for estimating SDE parameters
    all_nlls = []
    all_nlps = []
    all_mae_a = []
    all_mae_h = []
    all_right_percent = []
    best_ordered_data = np.copy(reordered_data)
    for iteration in range(max_iter):
        if iteration > 0:
            # If not first iteration, re-impute the data
            if data_missing_percent > 0:
                # Reverse the masking to re-impute the data
                reordered_data[missing_indices] = np.nan
                # Impute the data again with the new ordering
                start = time.time()
                reordered_data = knn_impute_with_temporal_context(reordered_data)     
                print("Time taken for KNN imputation:", time.time() - start)

        # Update SDE parameters A, G using MLE with the newly completed data
        A, H = update_sde_parameters(reordered_data, dt=time_step_size, T=T)

        average_nll = 0.0
        # average_nlp = 0.0
        for traj in range(num_trajectories):
            nll = compute_nll(reordered_data[traj], A, H, dt=time_step_size)
            average_nll += nll
            # average_nlp += compute_nlp(nll=nll, A=A, H=H)

        average_nll = average_nll/num_trajectories
        MAE_A = np.mean(np.abs(A - true_A))
        MAE_H = np.mean(np.abs(H - true_H))
        print(f"Iteration {iteration+1}:\nNLL = {average_nll:.3f}\nMAE to true A = {MAE_A:.3f}\nMAE to true H = {MAE_H:.3f}")
        
        all_nlls.append(average_nll)
        all_mae_a.append(MAE_A)
        all_mae_h.append(MAE_H)

        if check_convergence(score_list=all_nlls):
            # converged, should stop algorithm
            break

        reordered_data, reordered_order, missing_indices = reorder_step_by_step(reordered_data, reordered_order, A, H, time_step_size, float(average_nll.item()),
                                              known_initial_value=known_initial_value,
                                              noisy_measurements_sigma=noisy_measurements_sigma,
                                              missing_indices=missing_indices)

        right_percent = check_sorting_accuracy(original_data, reordered_data, reordered_order=reordered_order)
        print(f"Right sorting percent: {right_percent:.3f}")
        all_right_percent.append(right_percent)

    return A, H, reordered_data, best_ordered_data, all_nlls, all_nlps, all_mae_a, all_mae_h, all_right_percent


def check_rank(data, A, H):
    d = data.shape[2]   # data shape (num_trajectories, num_steps, d)
    x0 = np.copy(data[0, 0, :]).reshape((d, 1))
    M1 = np.copy(x0)
    M2 = np.copy(H)
    for i in range(1, d):
        M1 = np.concatenate((M1, A @ x0), axis=1)
        M2 = np.concatenate((M2, A @ H), axis=1)
        A = A @ A
    M1 = np.concatenate((M1, A @ x0), axis=1)
    M2 = np.concatenate((M2, A @ H), axis=1)
    M = np.concatenate((M1, M2), axis=1)
    rank = np.linalg.matrix_rank(M)
    
    return (rank == d, rank)


def noisy_measurement(data: np.ndarray, sigma: float = 0.1, random_seed: int = 167) -> np.ndarray:
    """Adding noise ~ N(0, I*sigma^2) to the data to mimic noisy measurements.

    Args:
        data (np.ndarray): Data to be added noise.
        sigma (float): Standard deviation of the Gaussian noise.
        random_seed (int, optional): Random seed for reproducibility. Defaults to 167.

    Returns:
        np.ndarray: Noisy data.
    """
    np.random.seed(random_seed)
    noise = np.random.normal(0, sigma, data.shape)
    return data + noise


def randomize_data(
    data: np.ndarray,
    known_initial_value: bool = False,
    random_percent: float = 0.5,
    random_seed: int = 167
    ) -> np.ndarray:
    """Randomize the data by shuffling time steps for all trajectories.
    Args:
        data (np.ndarray): Data to be randomized, shape (num_trajectories, num_steps, d).
        known_initial_value (bool, optional): If True, the initial values of trajectories
            are known beforehand, and should stay fixed. Defaults to False.
        random_percent (float, optional): Percentage of data to be randomized. Defaults to 0.5.
        random_seed (int, optional): Random seed for reproducibility. Defaults to 167.
    Returns:
        np.ndarray: Randomized data.
        np.ndarray: Permutation indices used for randomization.
    """
    np.random.seed(random_seed)
    num_trajectories, num_steps, d = data.shape
    permutation_id = np.arange(num_steps)

    num_fixed_steps = int(num_steps * (1 - random_percent))
    fixed_indices = np.random.choice(permutation_id, size=num_fixed_steps, replace=False)
    random_indices = [i for i in permutation_id if i not in fixed_indices]
    random_indices = np.random.permutation(random_indices)
    for i in range(len(permutation_id)):
        if i in fixed_indices:
            continue
        else:
            permutation_id[i] = random_indices[0]
            random_indices = random_indices[1:]

    if known_initial_value:
        id = np.argwhere(permutation_id == 0)
        permutation_id = np.delete(permutation_id, id)
        permutation_id = np.concatenate(([0], permutation_id))

    randomized_data = data[:, permutation_id, :]

    return randomized_data, permutation_id


def run_experiment(
    T: float, d: int, dt: float,
    num_trajectories: int,
    version: int = 3, max_iter: int = 20,
    known_initial_value: bool = False,
    noisy_measurements_sigma: float = 0,
    data_missing_percent: float = 0.0,
    random_percent: float = 1,
    random_seed: int = 167
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list, list, list, list, list]:
    """Running experiments and plotting the results

    Args:
        T (float): Time period.
        d (int): Variables dimension.
        dt (float): Time step size.
        num_trajectories (int): Number of trajectories.
        version (int, optional): Versions of experiment, as follow:
            Version 1: only noise, drift term is zero.
            Version 2: missing data, to-be-implemented.
            Version 3: time-homogeneous linear additive noise SDE.
            Defaults to 3.
        max_iter (int, optional): Number of iterations. Defaults to 20.        
        known_initial_value (bool, optional): If True, the initial values of trajectories
            are known beforehand, and should stay fixed. Defaults to False.
        noisy_measurement_sigma (float, optional): If larger than 0, add Gaussian noise to the data.
            Defaults to 0.
        data_missing_percent (float, optional): If larger than 0, some data will be missing.
            Defaults to 0.0.
        random_percent (float, optional): Percentage of data to be randomized. Defaults to 0.5.
        random_seed (int, optional): Random seed for reproducibility. Defaults to 167.
    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list, list, list, list, list]:
            Estimated drift-diffusion matrices (A, G), reordered data,
            best ordered data, all NLLs, all NLPs, all MAE_A, all MAE_H,
            and all right_sorting_percent results through trajectories.
    """
    np.random.seed(random_seed)
    if version == 1:
        A = np.zeros((d, d))
        G = np.ones((d, d)) * 5
    elif version == 2:
        # our_data = data_masking(our_data)
        pass
    elif version == 3:
        A = np.array([[0, 1], [-1, 0]])
        G = np.ones((d, d))
    elif version == 4:
        A = np.array([[1, 2], [1, 0]])
        G = np.array([[1, 2], [-1, -2]])
    elif version == 5:
        A = np.random.randn(d, d)
        G = np.random.randn(d, d)
        b = np.random.randn(d)
    elif version == 6:
        one_rand_A = np.random.randn(d)
        one_rand_G = np.random.randn(d)
        A = np.stack([one_rand_A for _ in range(d)], axis=0) * 5
        G = np.stack([one_rand_G for _ in range(d)], axis=0) * 2.5
    
    # Generating data
    points = generate_independent_points(d, d)
    X0_dist = [(point, 1 / len(points)) for point in points]
    X_appex = linear_additive_noise_data(
        num_trajectories=num_trajectories, d=d, T=T, dt_EM=dt, dt=dt,
        A=A, G=G, b=b, sample_X0_once=True, X0_dist=X0_dist)
    print("Generated data shape:", X_appex.shape)

    if noisy_measurements_sigma > 0:
        # Adding noise to the data
        X_appex = noisy_measurement(X_appex, sigma=noisy_measurements_sigma, random_seed=random_seed)

    # Randomize segments between each trajectory (to get rid of the temporal order between segments)
    random_X, permutation_id = randomize_data(X_appex, known_initial_value=known_initial_value,
                              random_percent=random_percent, random_seed=random_seed)

    right_percent = check_sorting_accuracy(X_appex, random_X, check_by_indices_order=True,
                                            reordered_order=permutation_id)
    print(f"Right-sorting percent before reordering: {right_percent:.3f}")
    print(f"Noise sigma: {noisy_measurements_sigma}")

    # Estimating SDE's parameters A, G
    estimated_A, estimated_H, reordered_X, best_ordered_data, all_nlls, all_nlps, all_mae_a, all_mae_h, all_right_percent = \
        estimate_sde_parameters(random_X, X_appex, reordered_order=permutation_id, time_step_size=dt, T=T, max_iter=max_iter, true_A=A,
                                true_H=G@G.T, known_initial_value=known_initial_value,
                                noisy_measurements_sigma=noisy_measurements_sigma,
                                data_missing_percent=data_missing_percent)
    all_nlls = [float(item) for item in all_nlls]
    all_right_percent.insert(0, right_percent)
    all_right_percent = [float(item) for item in all_right_percent]

    # Plot results
    num_points = X_appex.shape[1]

    # ---------------------------------------------------------
    # Plot each trajectory (each sub-array along axis=0)
    # ---------------------------------------------------------
    fig = plt.figure(figsize=(12, 12))
    # Create a 2x2 gridspec
    gs = fig.add_gridspec(2, 2)

    # Subplot for top-left
    ax1 = fig.add_subplot(gs[0, 0])

    # Now, subdivide the top-right section into two smaller subplots.
    gs_top_right = gs[0, 1].subgridspec(2, 1)

    ax2_top = fig.add_subplot(gs_top_right[0, 0])
    ax2_bottom = fig.add_subplot(gs_top_right[1, 0])

    # Subplot for bottom-left
    ax3 = fig.add_subplot(gs[1, 0])

    # Subplot for bottom-right
    ax4 = fig.add_subplot(gs[1, 1])

    time = np.arange(num_points)
    num_iterations = np.arange(1, len(all_nlls)+1)
    for i in range(X_appex.shape[0]):
        # data[i] is shape (num_points, 2)
        # Plot one line for each trajectory
        if i <= 5:
            ax1.plot(time, X_appex[i, :, 0], label=f"Trajectory {i}")
        # ax2.plot(time, best_ordered_data[i, :, 0], label=f"Trajectory {i}")
        if i <= 5:
            ax3.plot(time, reordered_X[i, :, 0], label=f"Trajectory {i}")

    ax1.set_xlabel('Time')
    ax1.set_ylabel('Variable 0')
    ax1.set_title("Original data")
    # ax1.legend()

    ax2_top.plot(num_iterations, all_mae_a, label=f"MAE to True A")
    ax2_bottom.plot(num_iterations, all_mae_h, label=f"MAE to True H")
    ax2_top.set_xlabel('Epoch')
    ax2_top.set_ylabel('Mean Absolute Error')
    ax2_bottom.set_xlabel('Epoch')
    ax2_bottom.set_ylabel('Mean Absolute Error')
    ax2_top.set_title("MAE between the estimated and true params")
    ax2_top.legend()
    ax2_bottom.legend()

    ax3.set_xlabel('Time')
    ax3.set_ylabel('Variable 0')
    ax3.set_title("Reconstructed data")
    ax3.legend()

    # ax4.plot(num_iterations, all_nlps)
    # ax4.set_xlabel('Epoch')
    # ax4.set_ylabel('Negative Log-Posterior')
    # ax4.set_title("Negative Log-Posterior through epochs")

    ax4.plot(num_iterations, all_nlls)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Negative Log-likelihood')
    ax4.set_title("Negative Log-likelihood through epochs")

    plt.savefig('figures/progress.png')

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot()
    ax.plot(np.arange(1, len(all_right_percent)+1), all_right_percent)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Right value percentage')
    ax.set_title("Right value percentage - averaging over each trajectory")

    plt.savefig('figures/reordering_accuracy.png')

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run experiments with custom parameters.')

    parser.add_argument('--T', type=float, default=1, help='Total time period T (seconds)')
    parser.add_argument('--d', type=int, default=10, help='Dimension d')
    parser.add_argument('--dt', type=float, default=0.01, help='Time step size dt (seconds)')
    parser.add_argument('--num_trajectories', type=int, default=2000, help='Number of Euler-Maruyama-simulated trajectories')
    parser.add_argument('--version', type=int, default=5, help='Experiment version')
    parser.add_argument('--max_iter', type=int, default=10, help='Maximum number of iterations')
    parser.add_argument('--known_initial_value', type=bool, default=False, help='Flag if initial step is known and fixed (not randomized)')
    parser.add_argument('--noisy_measurements_sigma', type=float, default=0.2, help='Noise Sigma in noisy measurements scenario')
    parser.add_argument('--data_missing_percent', type=float, default=0, help='Percentage of data missing')
    parser.add_argument('--random_percent', type=float, default=1, help='Randomized percentage of data (0-1), 1 being fully randomized')
    parser.add_argument('--random_seed', type=int, default=167, help='Random seed')

    args = parser.parse_args()

    np.random.seed(args.random_seed)

    # Run different experiments
    run_experiment(T=args.T, d=args.d, dt=args.dt, num_trajectories=args.num_trajectories,
                   version=args.version, max_iter=args.max_iter,
                   known_initial_value=args.known_initial_value,
                   noisy_measurements_sigma=args.noisy_measurements_sigma,
                   data_missing_percent=args.data_missing_percent,
                   random_percent=args.random_percent, random_seed=args.random_seed)

    """
    Ver 2: Data without Temporal Order
    Solution: Follow an iterative scheme with 2 steps, updating SDE's
    parameters and re-sorting data using empirical score estimation to sort step-by-step.
    """