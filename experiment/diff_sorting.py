import numpy as np
import matplotlib.pyplot as plt
import itertools
import scipy
import scipy.linalg
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.impute import KNNImputer
from numpy.lib.stride_tricks import sliding_window_view

from experiments import generate_independent_points, linear_additive_noise_data


def data_masking(data: np.ndarray, p: float = 0.2, known_initial_value: bool = False) -> np.ndarray:
    """Randomly mask the input data, such that for each period of time there is
    at least one trajectory with data present there.

    Args:
        data (np.ndarray): Input data with shape
            (num_trajectories, num_steps, d).
        p (float, optional): Number of trajectories masked at each segment.
            Defaults to 0.2.
        known_initial_value (bool, optional): If True, the initial steps of trajectories are fixed.
            Defaults to False.

    Returns:
        np.ndarray: Randomly-masked data with default masking value np.nan.
    """
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
        X = torch.reshape(X, (X.shape[0]*X.shape[1], X.shape[2]))
    num_steps, d = X.shape
    H_dt = torch.Tensor(H * dt)
    # quick fix, instead of getting logdet of H_dt, which
    # is -inf due to det(H_dt) = 0, we get logdet of H_dt + eps*I
    # This ensures that H_dt_reg = H + eps*I is full rank and invertible.
    eps = 1e-2
    H_dt_reg = torch.Tensor(H_dt) + eps*torch.eye(H_dt.shape[0])
    # Precompute inverse and determinant of H_dt
    H_dt_inv = torch.linalg.pinv(H_dt)
    sign, logdet_H_dt = torch.linalg.slogdet(H_dt_reg)
    const_term = 0.5 * (d * np.log(2 * np.pi) + logdet_H_dt)
    for t in range(num_steps - 1):
        X_t = X[t]
        X_tp1 = X[t + 1]
        # Compute mean: mu_t = e^{A dt}  X_t (approximate if needed)
        # mu_t = np.exp(A*dt)@X_t
        mu_t = X_t + torch.Tensor(A) @ X_t * dt  # Using Euler-Maruyama approximation
        diff = X_tp1 - mu_t
        exponent = 0.5 * diff.T @ H_dt_inv @ diff
        nll += exponent + const_term

    return nll


def reorder_trajectories(
    data: np.ndarray,
    A: np.ndarray,
    H: np.ndarray,
    time_step_size: float,
    alpha: float = 0.5,
    beta: float = 10,
    known_initial_value: bool = False
    ) -> np.ndarray:
    """Reordering trajectory data to maximize our objective (log-likelihood minus DAG penalty),
    which is minimizing (negative log-likelihood plus DAG penalty)

    Args:
        data (np.ndarray): Arrays of trajectory to be re-ordered,
            shape (num_trajectories, num_segments, points_per_segment, d).
        A (np.ndarray): Current estimated drift matrix, shape (d, d).
        H (np.ndarray): Current estimated (observational) diffusion matrix, shape (d, d).
        time_step_size (float): Step size with respect to time between points.
        alpha (float, optional): Regularization hyper-parameter for DAG penalty. Defaults to 0.1.
        beta (float, optional): Penalization hyper-parameter used in calculating loss of ordering.
            Loss = NLL + DAG_penalty + beta*Distance_cost
        known_initial_value (bool, optional): If True, the initial values of trajectories
            are known beforehand, and should stay fixed. Defaults to False.

    Returns:
        np.ndarray: A re-ordered array of segments.
    """
    num_trajectories, num_segments, steps_per_segment, d = data.shape

    if known_initial_value:
        start_sorting_id = 1
    else:
        start_sorting_id = 0

    for traj in range(num_trajectories):
        best_orderings = {}
        if num_segments >= 5:
            num_orderings_kept = 5
        else:
            num_orderings_kept = num_segments

        # Find all permutations of the trajectory's not-missing segments
        if known_initial_value:
            all_indices_permutations = list(itertools.permutations(data[traj, start_sorting_id:]))
            all_indices_permutations = [[data[traj, 0, :, :]] + list(item) for item in all_indices_permutations]
        else:
            all_indices_permutations = list(itertools.permutations(data[traj]))
            all_indices_permutations = [list(item) for item in all_indices_permutations]

        count = 1
        for candidate_order in all_indices_permutations:
            # candidate_order is a list of 2D arrays, now we stack it
            # to get the trajectory to compute NLL
            reordered_trajectory = np.stack(candidate_order, axis=0)
            # Evaluate the negative log-likelihood
            nll = compute_nll(reordered_trajectory, A, H, time_step_size)
            # nlp = compute_nlp(nll=nll, A=A, H=H)
            # Also compute a DAG penalty on A
            # penalty = dag_penalty(A, alpha)
            loss = nll

            # quick, dirty fix
            if loss == -(np.inf):
                loss = -1e10
            elif loss == np.inf:
                loss = 1e10

            # loss = np.log(loss)

            # distance_cost = compute_distance(X=reordered_trajectory)
            # # print(loss, "loss")
            # # print(distance_cost, "distance")
            # # beta = int(loss/distance_cost)
            # # print(beta, "beta")
            # loss += beta*distance_cost

            # quick, dirty fix
            if loss in list(best_orderings.keys()):
                # to make the loss slightly different due to some
                # unknown reasons sometimes the loss are stuck at
                # a realy high value
                count += 1
                loss -= loss*count*1e-5

            best_orderings.update({loss: reordered_trajectory})
            best_orderings = dict(sorted(best_orderings.items(), key=lambda x:x[0])) # sort by key
            if len(best_orderings) > num_orderings_kept:
                best_orderings.popitem()

        # print([item.item() for item in list(best_orderings.keys())])
        # # Update the current trajectory randomly to one of the best trajectories just found
        # probabilities = list(best_orderings.keys())
        # # [-200, -4, 0, 52] --> [0, 196, 200, 252]
        # probabilities = [float(item) + abs(float(np.min(probabilities))) for item in probabilities]
        # # [0, 196, 200, 252] --> [0, 196/252, 200/252, 1]
        # probabilities = [item/max(probabilities) if max(probabilities) > 0 else item for item in probabilities]
        # probabilities = scipy.special.softmax(probabilities)
        # try:
        #     choice = np.random.choice(list(best_orderings.keys()), 1, p=probabilities)
        #     data[traj] = best_orderings[float(choice[0])]
        # except:
        data[traj] = list(best_orderings.values())[0]

    return data


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
    # if nll > 50:
    #     A_temp = np.random.randn(d, d)
    #     H_temp = np.random.randn(d, d)
    # else:
        # A_temp = A + np.random.randn(d, d)*0.5
        # H_temp = H + np.random.randn(d, d)*0.5
    A_temp, H_temp = A, H
    if use_quick_sort:
        s = time.time()
        # quick_sort(temp_data[:, 1:, :], A_temp, H_temp, time_step_size, x0)
        print("Time taken for quick sort:", time.time() - s)
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

    return temp_data, order


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
        if score_list[-1] < -150:
            good_iterations = sum(1 for i in score_list if i < -150)
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
        right_percent_through_traj = []
        for traj in range(num_trajectories):
            right_count = 0
            for i in range(len(reordered_order)):
                if original_order[i] == reordered_order[i]:
                    right_count += 1
            right_percent = right_count / num_steps
            right_percent_through_traj.append(right_percent)
        right_percent = np.mean(right_percent_through_traj)
    else:
        right_percent_through_traj = []
        for traj in range(num_trajectories):
            right_value = ((original_data[traj] - reordered_data[traj]) == 0.0).astype(int)
            right_value_count = (right_value == 1).sum()
            right_value_count /= num_steps*d
            right_percent_through_traj.append(right_value_count)
        right_percent = np.mean(right_percent_through_traj)
    
    return right_percent


def sinkhorn(log_alpha, n_iters=50):
    for _ in range(n_iters):
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=1, keepdim=True)
        log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=2, keepdim=True)
    return torch.exp(log_alpha)


class DifferentiableSorter(nn.Module):
    def __init__(self, num_steps: int, d: int):
        super().__init__()
        self.linear = nn.Linear(d, num_steps)

    def forward(self, X):
        # X has shape (num_trajectories, num_steps, d)
        log_alpha = self.linear(X)  # shape (num_trajectories, num_steps, num_steps)
        start = time.time()
        soft_perm = sinkhorn(log_alpha)[0]
        print("Time taken for Sinkhorn:", time.time() - start)

        return soft_perm


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
    # # 1) Sort all segments in each trajectory by variance
    # reordered_data = sort_data_by_var(data, decreasing_var=False,
    #                                   known_initial_value=known_initial_value) # assuming diverging SDEs

    if data_missing_percent > 0:
        # If data is missing, we need to impute it first
        data, missing_indices = data_masking(data, p=data_missing_percent,
                                             known_initial_value=known_initial_value)
        start = time.time()
        data = knn_impute_with_temporal_context(data)
        print("Time taken for KNN imputation:", time.time() - start)
        
    X = torch.Tensor(data)

    # 2) Iterative scheme for estimating SDE parameters
    all_nlls = []
    all_nlps = []
    all_mae_a = []
    all_mae_h = []
    all_right_percent = []
    best_nll = np.inf
    best_ordered_data = np.copy(X)

    sorter = DifferentiableSorter(num_steps=X.shape[1], d=X.shape[2])
    optimizer = optim.Adam(list(sorter.parameters()), lr=1e-2)

    for iteration in range(max_iter):
        optimizer.zero_grad()

        P_soft = sorter(X)
        X_sorted = P_soft @ X

        if iteration > 0:
            # If not first iteration, re-impute the data
            if data_missing_percent > 0:
                # Reverse the masking to re-impute the data
                X_sorted[missing_indices] = np.nan
                # Impute the data again with the new ordering
                start = time.time()
                X_sorted = knn_impute_with_temporal_context(X_sorted)     
                print("Time taken for KNN imputation:", time.time() - start)
            else:
                missing_indices = None

        # Update SDE parameters A, G using MLE with the newly completed data
        A, H = update_sde_parameters(X_sorted.detach().numpy(), dt=time_step_size, T=T)

        average_nll = 0.0
        start = time.time()
        for traj in range(num_trajectories):
            nll = compute_nll(X_sorted[traj], A, H, dt=time_step_size)
            average_nll += nll
        print("Time taken for NLL computation:", time.time() - start)

        average_nll = average_nll/num_trajectories

        average_nll.backward()
        optimizer.step()

        MAE_A = np.mean(np.abs(A - true_A))
        MAE_H = np.mean(np.abs(H - true_H))
        
        all_nlls.append(average_nll)
        all_mae_a.append(MAE_A)
        all_mae_h.append(MAE_H)

        print(f"Iteration {iteration+1}:\nNLL = {average_nll:.3f}\nMAE to true A = {MAE_A:.3f}\nMAE to true H = {MAE_H:.3f}")
        
        if check_convergence(score_list=all_nlls):
            # converged, should stop algorithm
            break

        # reordered_data, reordered_order = reorder_step_by_step(reordered_data, reordered_order, A, H, time_step_size, float(average_nll.item()),
        #                                       known_initial_value=known_initial_value,
        #                                       noisy_measurements_sigma=noisy_measurements_sigma,
        #                                       missing_indices=missing_indices)
        
        right_percent = check_sorting_accuracy(original_data, X_sorted, reordered_order=P_soft.argmax(dim=1).numpy())
        print(f"Right sorting percent: {right_percent:.3f}")
        all_right_percent.append(right_percent)

    return A, H, X_sorted, best_ordered_data, all_nlls, all_nlps, all_mae_a, all_mae_h, all_right_percent


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


def noisy_measurement(data: np.ndarray, sigma: float = 0.1):
    """Adding noise ~ N(0, I*sigma^2) to the data to mimic noisy measurements.

    Args:
        data (np.ndarray): Data to be added noise.
        sigma (float): Standard deviation of the Gaussian noise.

    Returns:
        np.ndarray: Noisy data.
    """
    noise = np.random.normal(0, sigma, data.shape)
    return data + noise


def randomize_data(
    data: np.ndarray,
    known_initial_value: bool = False,
    random_percent: float = 0.5,
    random_seed: int = 42
    ) -> np.ndarray:
    """Randomize the data by shuffling time steps for all trajectories.
    Args:
        data (np.ndarray): Data to be randomized, shape (num_trajectories, num_steps, d).
        known_initial_value (bool, optional): If True, the initial values of trajectories
            are known beforehand, and should stay fixed. Defaults to False.
        random_percent (float, optional): Percentage of data to be randomized. Defaults to 0.5.
        random_seed (int, optional): Random seed for reproducibility. Defaults to 42.
    Returns:
        np.ndarray: Randomized data.
        np.ndarray: Permutation indices used for randomization.
    """
    np.random.seed(random_seed)
    num_trajectories, num_steps, d = data.shape
    if known_initial_value:
        start = 1
    else:
        start = 0
    permutation_id = np.arange(start, num_steps)

    num_fixed_steps = int(num_steps * (1 - random_percent))
    fixed_indices = np.random.choice(permutation_id, size=num_fixed_steps, replace=False)
    random_indices = [i for i in permutation_id if i not in fixed_indices]
    random_indices = np.random.permutation(random_indices)
    for i in range(start, len(permutation_id)):
        if i in fixed_indices:
            continue
        else:
            permutation_id[i] = random_indices[0]
            random_indices = random_indices[1:]

    if known_initial_value:
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
    random_percent: float = 0.5
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
    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list, list, list, list, list]:
            Estimated drift-diffusion matrices (A, G), reordered data,
            best ordered data, all NLLs, all NLPs, all MAE_A, all MAE_H,
            and all right_sorting_percent results through trajectories.
    """
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
        A=A, G=G, sample_X0_once=True, X0_dist=X0_dist)
    print(X_appex.shape, "Generated data shape")

    if noisy_measurements_sigma > 0:
        # Adding noise to the data
        X_appex = noisy_measurement(X_appex, sigma=noisy_measurements_sigma)
        print(X_appex.shape, "Noisy data shape")

    # Randomize segments between each trajectory (to get rid of the temporal order between segments)
    random_X, permutation_id = randomize_data(X_appex, known_initial_value=known_initial_value,
                              random_percent=random_percent, random_seed=42)

    right_percent = check_sorting_accuracy(X_appex, random_X, check_by_indices_order=True,
                                            reordered_order=permutation_id)
    print(right_percent)

    # Estimating SDE's parameters A, G
    estimated_A, estimated_H, reordered_X, best_ordered_data, all_nlls, all_nlps, all_mae_a, all_mae_h, all_right_percent = \
        estimate_sde_parameters(random_X, X_appex, reordered_order=permutation_id, time_step_size=dt, T=T, max_iter=max_iter, true_A=A,
                                true_H=G@G.T, known_initial_value=known_initial_value,
                                noisy_measurements_sigma=noisy_measurements_sigma,
                                data_missing_percent=data_missing_percent)
    all_nlls = [float(item) for item in all_nlls]
    all_right_percent.insert(0, right_percent)
    all_right_percent = [float(item) for item in all_right_percent]
    
    # # We'll use Cholesky if H_est is positive definite
    # # else fallback to sqrtm or pseudo-chol
    # try:
    #     estimated_G = np.linalg.cholesky(estimated_H)
    # except np.linalg.LinAlgError:
    #     # if not SPD, do a symmetric sqrt
    #     from scipy.linalg import sqrtm
    #     estimated_G = sqrtm(estimated_H)
    #     # If it still fails, consider a small regularization

    # print(estimated_A, "A")
    # print(estimated_G, "G")

    # try:
    #     is_id, rank_val = check_rank(reordered_X, estimated_A, estimated_H)
    #     print(f"Rank = {rank_val}, Identifiable? {is_id}")
    # except Exception as e:
    #     print(f"Error in rank check: {e}")

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
            ax3.plot(time, reordered_X[i, :, 0].detach().numpy(), label=f"Trajectory {i}")

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
    # data generated by data_generation.py of APPEX code
    T = 0.20
    d = 5
    dt = 0.01
    num_trajectories = 1500
    version = 5
    max_iter = 50
    known_initial_value = True
    noisy_measurements_sigma = 0
    data_missing_percent = 0
    random_percent = 0.5

    # Run different experiments
    run_experiment(T=T, d=d, dt=dt, num_trajectories=num_trajectories,
                   version=version, max_iter=max_iter,
                   known_initial_value=known_initial_value,
                   noisy_measurements_sigma=noisy_measurements_sigma,
                   data_missing_percent=data_missing_percent,
                   random_percent=random_percent)

    """
    Ver 2: Data without Temporal Order
    Solution: Follow an iterative scheme with 2 steps, updating SDE's
    parameters and re-sorting data using empirical score estimation to sort step-by-step.
    """