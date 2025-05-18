import numpy as np
import matplotlib.pyplot as plt
import itertools
import scipy
import scipy.linalg
import time

from experiments import generate_independent_points, linear_additive_noise_data


def data_masking(data: np.ndarray) -> np.ndarray:
    """Randomly mask the input data, such that for each period of time there is
    at least one trajectory with data present there.

    Args:
        data (np.ndarray): Input data with shape
            (num_trajectories, num_segments, points_per_segment, d).

    Returns:
        np.ndarray: Randomly-masked data with default value -999.
    """
    masked_data = np.zeros((data.shape))
    for segment in range(data.shape[1]):
        # number of trajectories being masked at each segment
        p = np.random.choice(data.shape[0])
        rows_to_be_masked = np.random.choice(data.shape[0], p, replace=False)
        for row in rows_to_be_masked:
            masked_data[row, segment, :] = -999

    return masked_data


def sort_data_by_var(
    data: np.ndarray,
    known_initial_value: np.ndarray = None,
    decreasing_var: bool = False
    ) -> np.ndarray:
    """Sorting input data by the variance of the segments of each trajectory.

    Args:
        data (np.ndarray): Matrix with shape
            (num_trajectories, num_segments, points_per_segment, d).
        decreasing_var (bool, optional): If True, sorting by decreasing variance, this
            is for the case of converging SDEs, such as OU processes. Defaults to False.
        known_initial_value (bool, optional): If True, the initial values of trajectories
            are known beforehand, and should stay fixed. Defaults to False.

    Returns:
        np.ndarray: Sorted data with the same shape.
    """
    if known_initial_value:
        first_segments = np.zeros((data.shape[0], 1, data.shape[2], data.shape[3]))
        first_segments[:, 0] = data[:, 0]
        data = np.copy(data[:, 1:])
        sorted_data = np.zeros((data.shape))
    else:
        sorted_data = np.zeros((data.shape))
    for trajectory in range(data.shape[0]):
        all_variances = []
        for segment in range(data.shape[1]):
            if np.any(data[trajectory, segment, :, :] == -999):
                all_variances.append(-99)
                continue
            # Compute covariance matrix
            cov_matrix = np.cov(data[trajectory, segment, :, :],
                rowvar=False, ddof=1)  # rowvar=False means columns are variables
            # Compute total variance as the trace of the covariance matrix
            var = np.trace(cov_matrix)
            all_variances.append(var)
        # Sorting segments by their variances, missing segments stay in the same positions
        present_indices = {id:value for id, value in enumerate(all_variances) if value != -99}
        if decreasing_var:
            present_indices = dict(sorted(present_indices.items(), key=lambda x:x[1]))
            present_indices = list(present_indices.keys())
        else:
            present_indices = dict(sorted(present_indices.items(), key=lambda x:x[1], reverse=True))
            present_indices = list(present_indices.keys())

        temp = 0
        for id in range(data.shape[1]):
            if id in present_indices:
                sorted_data[trajectory, id] = data[trajectory, present_indices[temp], :, :]
                temp += 1

    if known_initial_value:
        sorted_data = np.concatenate((first_segments, sorted_data), axis=1)

    return sorted_data


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


def compute_distance(X: np.ndarray) -> float:
    """Calculating L2-distance between all pairs of consecutive points given in the input trajectory

    Args:
        X (np.ndarray): 2D arrays of points with shape (num_steps, d).

    Returns:
        distance_cost (float): Total distance cost.
    """
    num_steps = X.shape[0]
    distance_cost = 0.0

    for i in range(num_steps - 1):
        distance_cost += np.sqrt(np.sum(np.square(X[i, :] - X[i+1, :]))) # L2-distance

    return distance_cost


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


def estimate_mean_cov(X):
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError("X must be a 2-D array with shape (N, d)")

    # ---- mean ------------
    m_hat = X.mean(axis=0)

    # ---- covariance ------
    # Unbiased sample covariance (divides by N-1)
    C_hat = np.cov(X, rowvar=False, bias=False)

    return m_hat, C_hat


def cal_score(A, H, x_t, t, x0):
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

    m_t, P_t = estimate_mean_cov(x_t)
    try:
        inv_P = np.linalg.inv(P_t)
    except np.linalg.LinAlgError:
        # If P_t is singular, use pseudo-inverse
        inv_P = np.linalg.pinv(P_t)

    score = (x_t - m_t) @ (-inv_P)

    return score


def quick_sort(arr: np.ndarray, A, H, time_step_size, x0) -> np.ndarray:
    def partition(lo: int, hi: int) -> int:
        """Partition around arr[hi] (the pivot)."""
        pivot = arr[:, hi, :]
        score_pivot = cal_score(A=A, H=H, x_t=pivot, t=time_step_size*(hi+1), x0=x0)
        i = lo - 1
        for j in range(lo, hi):
            score_j = cal_score(A=A, H=H, x_t=arr[:, j, :], t=time_step_size*(j+1), x0=x0)
            drift = np.mean((pivot - arr[:, j, :])/time_step_size, axis=0)
            err_j = cal_error(drift=drift, H=H, score=score_j, n=arr.shape[0], d=A.shape[0])
            err_pivot = cal_error(drift=drift, H=H, score=score_pivot, n=arr.shape[0], d=A.shape[0])
            if err_j > err_pivot:
                i += 1
                temp = np.copy(arr[:, i, :])
                arr[:, i, :] = np.copy(arr[:, j, :])
                arr[:, j, :] = np.copy(temp)
        temp = np.copy(arr[:, i + 1, :])
        arr[:, i + 1, :] = np.copy(arr[:, hi, :])
        arr[:, hi, :] = np.copy(temp)
        return i + 1

    # ---- iterative quick‑sort (explicit stack) ----
    stack = [(0, arr.shape[1] - 1)]
    while stack:
        lo, hi = stack.pop()
        if lo < hi:
            p = partition(lo, hi)
            # push larger side first so the smaller side is processed next,
            # keeping the stack depth ≤ log₂(n) in the average case
            if p - lo < hi - p:
                stack.append((p + 1, hi))
                stack.append((lo, p - 1))
            else:
                stack.append((lo, p - 1))
                stack.append((p + 1, hi))


def reorder_step_by_step(
    data: np.ndarray,
    A: np.ndarray,
    H: np.ndarray,
    time_step_size: float,
    nll: float,
    known_initial_value: bool = False,
    use_quick_sort: bool = False
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
        quick_sort(temp_data[:, 1:, :], A_temp, H_temp, time_step_size, x0)
        print("Time taken for quick sort:", time.time() - s)
    else:
        s = time.time()
        while end > start:
            for i in range(start, end):
                j = i + 1
                x_i = temp_data[:, i, :].reshape(num_trajectories, d)
                x_j = temp_data[:, j, :].reshape(num_trajectories, d)
                score_i = cal_score(A=A_temp, H=H_temp, x_t=x_i, t=time_step_size*(i+1), x0=x0)
                score_j = cal_score(A=A_temp, H=H_temp, x_t=x_j, t=time_step_size*(j+1), x0=x0)
                drift = np.mean((x_j - x_i)/time_step_size, axis=0)
                err_i = cal_error(drift=drift, H=H_temp, score=score_i, n=num_trajectories, d=d)
                err_j = cal_error(drift=drift, H=H_temp, score=score_j, n=num_trajectories, d=d)

                if err_i < err_j:
                    temp = np.copy(temp_data[:, i, :])
                    temp_data[:, i, :] = np.copy(temp_data[:, j, :])
                    temp_data[:, j, :] = np.copy(temp)
                if i == end - 1:
                    end = i
        print("Time taken for bubble sort:", time.time() - s)

    return temp_data


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
        if score_list[-1] < 2000:
            good_iterations = sum(1 for i in score_list if i < 2000)
            if good_iterations >= 5:
                return True
            diff = [score_list[id].item() - score_list[id+1].item() for id in range(len(score_list)-1)]
            if np.sum(diff[-5:]) < 1:
                return True
    else:
        pass

    return False


def estimate_sde_parameters(
    data: np.ndarray,
    original_data,
    time_step_size: float,
    T: float,
    true_A: np.ndarray,
    true_H: np.ndarray,
    max_iter: int = 10,
    alpha: float = 0.1,
    known_initial_value: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
    """Main iterative scheme:
    1) Reconstruct / reorder segments using the current (A, G).
    2) Re-estimate A, G from the newly-reconstructed data.

    Args:
        data (np.ndarray): Input data, shape (num_trajectories, num_segments, points_per_segment, d)
        time_step_size (float): Step size with respect to time between points.
        T (float): Time period.
        true_A (nd.ndarray): True drift matrix A. Used for computing MAE.
        true_H (nd.ndarray): True (observational) diffusion matrix H (= G@G.T). Used for computing MAE.
        max_iter (int, optional): Maximum number of iterations. Defaults to 10.
        alpha (float, optional): Regularization hyper-parameter. Defaults to 0.1.
        known_initial_value (bool, optional): If True, the initial values of trajectories
            are known beforehand, and should stay fixed. Defaults to False.

    Returns:
        tuple[np.ndarray, np.ndarray]: The estimated drift-diffusion matrices (A, G)
    """
    # # 1) Sort all segments in each trajectory by variance
    # reordered_data = sort_data_by_var(data, decreasing_var=False,
    #                                   known_initial_value=known_initial_value) # assuming diverging SDEs
    # # reordered_data has shape (num_trajectories, num_segments, steps_per_segment, d)
    # # and is now transformed to (num_trajectories, num_steps, d) with
    # # num_steps = num_segments * steps_per_segment so that we can use the whole trajectory data
    # # to update parameters. After that, it is transformed back to the previous shape to be
    # # re-ordered again.
    num_segments = data.shape[1]
    steps_per_segment = data.shape[2]
    reordered_data = np.copy(data.reshape(num_trajectories,
                                                 num_segments*steps_per_segment, d))

    # 2) Iterative scheme for estimating SDE parameters
    all_nlls = []
    all_nlps = []
    all_mae_a = []
    all_mae_h = []
    all_right_percent = []
    best_nll = np.inf
    best_nlp = np.inf
    best_ordered_data = np.copy(reordered_data)
    for iteration in range(max_iter):
        # Update SDE parameters A, G using MLE with the newly completed data
        A, H = update_sde_parameters(reordered_data, dt=time_step_size, T=T)

        average_nll = 0.0
        # average_nlp = 0.0
        for traj in range(num_trajectories):
            nll = compute_nll(reordered_data[traj], A, H, dt=time_step_size)
            average_nll += nll
            # average_nlp += compute_nlp(nll=nll, A=A, H=H)

        average_nll = average_nll/num_trajectories
        # average_nlp = average_nlp/num_trajectories
        MAE_A = np.mean(np.abs(A - true_A))
        MAE_H = np.mean(np.abs(H - true_H))
        print(f"Iteration {iteration+1}:\nNLL = {average_nll:.3f}\nMAE to true A = {MAE_A:.3f}\nMAE to true H = {MAE_H:.3f}")
        
        all_nlls.append(average_nll)
        # all_nlps.append(average_nlp)
        all_mae_a.append(MAE_A)
        all_mae_h.append(MAE_H)
        # if average_nll < best_nll:
        #     best_nll = average_nll
        #     best_ordered_data = np.copy(reordered_data)
        # if average_nlp < best_nlp:
        #     best_nlp = average_nlp
        #     best_ordered_data = np.copy(reordered_data)
        
        # # Transform the data
        # reordered_data = np.reshape(reordered_data,
        #                             (num_trajectories, num_segments,
        #                             steps_per_segment, d))
        # # Reconstruct / reorder segments to maximize LL - DAG penalty
        # reordered_data = reorder_trajectories(reordered_data, A, H, time_step_size, alpha,
        #                                       known_initial_value=known_initial_value)

        if check_convergence(score_list=all_nlls):
            # converged, should stop algorithm
            right_percent_through_traj = []
            for traj in range(num_trajectories):
                right_value = ((original_data[traj] - reordered_data[traj]) == 0.0).astype(int)
                right_value_count = (right_value == 1).sum()
                right_value_count /= original_data[traj].shape[0]*original_data[traj].shape[1]
                right_percent_through_traj.append(right_value_count)
            right_percent = np.mean(right_percent_through_traj)
            all_right_percent.append(right_percent)
            break

        reordered_data = reorder_step_by_step(reordered_data, A, H, time_step_size, float(average_nll.item()),
                                              known_initial_value=known_initial_value)
        
        # # Transform the data back to its previous shape
        # reordered_data = np.reshape(reordered_data, (num_trajectories,
        #                                             num_segments*steps_per_segment, d))

        right_percent_through_traj = []
        for traj in range(num_trajectories):
            right_value = ((original_data[traj] - reordered_data[traj]) == 0.0).astype(int)
            right_value_count = (right_value == 1).sum()
            right_value_count /= original_data[traj].shape[0]*original_data[traj].shape[1]
            right_percent_through_traj.append(right_value_count)
        right_percent = np.mean(right_percent_through_traj)
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


def run_experiment(
    T: float, d: int, dt: float,
    num_trajectories: int,
    version: int = 3, max_iter: int = 20,
    known_initial_value: bool = False):
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
    """
    if version == 1:
        A = np.zeros((d, d))
        G = np.ones((d, d)) * 5
    elif version == 2:
        # our_data = data_masking(our_data)
        pass
    elif version == 3:
        A = np.ones((d, d)) * 5
        G = np.ones((d, d)) * 1.5
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
    print(X_appex.shape)
    
    # Reshape data from shape (num_trajectories, num_steps, d)
    # to shape (num_trajectories, num_segment, steps_per_segment, d)
    steps_per_segment = 1
    X_appex = X_appex.reshape(X_appex.shape[0], int(round(X_appex.shape[1]/steps_per_segment)),
                           steps_per_segment, X_appex.shape[2])
    print(X_appex.shape)

    # Randomize segments between each trajectory (to get rid of the temporal order between segments)
    random_X = np.zeros((X_appex.shape))
    # for i in range(X_appex.shape[0]):
    if known_initial_value:
        permutation_id = np.random.permutation(np.arange(1, X_appex.shape[1]))
        permutation_id = np.concatenate(([0], permutation_id))
        random_X = X_appex[:, permutation_id, :, :]
    else:
        permutation_id = np.random.permutation(X_appex.shape[1])
        random_X = X_appex[:, permutation_id, :, :]
    # random_X = np.copy(X_appex)

    right_percent_through_traj = []
    for traj in range(num_trajectories):
        right_value = ((X_appex[traj] - random_X[traj]) == 0.0).astype(int)
        right_value_count = (right_value == 1).sum()
        right_value_count /= X_appex[traj].shape[0]*X_appex[traj].shape[1]*X_appex[traj].shape[2]
        right_percent_through_traj.append(right_value_count)
    right_percent = np.mean(right_percent_through_traj)
    print(right_percent)

    # Estimating SDE's parameters A, G
    X_appex = X_appex.reshape(X_appex.shape[0], X_appex.shape[1]*X_appex.shape[2], X_appex.shape[3])
    estimated_A, estimated_H, reordered_X, best_ordered_data, all_nlls, all_nlps, all_mae_a, all_mae_h, all_right_percent = \
        estimate_sde_parameters(random_X, X_appex, time_step_size=dt, T=T, max_iter=max_iter, true_A=A,
                                true_H=G@G.T, known_initial_value=known_initial_value)
    all_nlls = [float(item) for item in all_nlls]
    
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

    try:
        is_id, rank_val = check_rank(reordered_X, estimated_A, estimated_H)
        print(f"Rank = {rank_val}, Identifiable? {is_id}")
    except Exception as e:
        print(f"Error in rank check: {e}")

    # Plot results
    random_X = random_X.reshape(random_X.shape[0], random_X.shape[1]*random_X.shape[2], random_X.shape[3])
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

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot()
    ax.plot(num_iterations, all_right_percent)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Right value percentage')
    ax.set_title("Right value percentage - averaging over each trajectory")

    plt.show()

    return


if __name__ == "__main__":
    # data generated by data_generation.py of APPEX code
    d = 10
    num_trajectories = 2000
    max_iter = 25
    known_initial_value = False

    # Run different experiments
    run_experiment(T=2.5, d=d, dt=0.01, num_trajectories=num_trajectories,
                   version=5, max_iter=max_iter, known_initial_value=known_initial_value)

    """
    Ver 2: Data without Temporal Order
    Solution: Follow an iterative scheme with 2 steps, updating SDE's
    parameters and re-sorting data using empirical score estimation to sort step-by-step.
    """