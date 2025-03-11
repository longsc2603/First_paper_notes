import numpy as np
import matplotlib.pyplot as plt
import itertools
import scipy
import scipy.special

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


def sort_data_by_var(data: np.ndarray, decreasing_var: bool = False) -> np.ndarray:
    """Sorting input data by the variance of the segments of each trajectory.

    Args:
        data (np.ndarray): Matrix with shape
            (num_trajectories, num_segments, points_per_segment, d).
        decreasing_var (bool, optional): If True, sorting by decreasing variance, this
            is for the case of converging SDEs, such as OU processes. Defaults to False.

    Returns:
        np.ndarray: Sorted data with the same shape.
    """
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
        H (np.ndarray): Estimated (observational) diffusion matrix, shape (d, m), H = G*G.T
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
    eps = 1e-3
    H_dt_reg = eps*np.eye(H_dt.shape[0])
    # Precompute inverse and determinant of H_dt
    H_dt_inv = np.linalg.pinv(H_dt)
    sign, logdet_H_dt = np.linalg.slogdet(H_dt_reg)
    const_term = 0.5 * (d * np.log(2 * np.pi) + logdet_H_dt)
    for t in range(num_steps - 1):
        X_t = X[t]
        X_tp1 = X[t + 1]
        # Compute mean: mu_t = e^{A dt} X_t (approximate if needed)
        mu_t = np.exp(A*dt)@X_t
        # mu_t = X_t + A @ X_t * dt  # Using Euler-Maruyama approximation
        diff = X_tp1 - mu_t
        exponent = 0.5 * diff.T @ H_dt_inv @ diff
        nll += exponent + const_term
    return nll


def reorder_trajectories(
    data: np.ndarray,
    A: np.ndarray,
    H: np.ndarray,
    time_step_size: float,
    alpha: float = 0.5
    ) -> np.ndarray:
    """Reordering trajectory data to maximize our objective (log-likelihood minus DAG penalty),
    which is minimizing (negative log-likelihood plus DAG penalty)

    Args:
        data (np.ndarray): Arrays of trajectory to be re-ordered,
            shape (num_trajectories, num_segments, points_per_segment, d).
        A (np.ndarray): Current estimated drift matrix, shape (d, d).
        H (np.ndarray): Current estimated (observational) diffusion matrix, shape (d, d).
        time_step_size (float): Step size with respect to time between points.
        alpha (float, optional): Regularization hyper-parameter. Defaults to 0.1.

    Returns:
        np.ndarray: A re-ordered array of segments.
    """
    num_trajectories, num_segments, steps_per_segment, d = data.shape
    for traj in range(num_trajectories):
        best_orderings = {}
        if num_segments >= 5:
            num_orderings_kept = 5
        else:
            num_orderings_kept = num_segments

        # Find all permutations of the trajectory's not-missing segments
        all_indices_permutations = list(itertools.permutations(data[traj]))
        all_indices_permutations = [list(item) for item in all_indices_permutations]

        for candidate_order in all_indices_permutations:
            # candidate_order is a list of 2D arrays, now we stack it 
            # to get the trajectory to compute NLL
            reordered_trajectory = np.stack(candidate_order, axis=0)
            # Evaluate the negative log-likelihood
            nll = compute_nll(reordered_trajectory, A, H, time_step_size)
            # Also compute a DAG penalty on A
            penalty = dag_penalty(A, alpha)
            loss = nll + penalty

            # quick, dirty fix
            if loss == -(np.inf):
                loss = -9999

            best_orderings.update({loss: reordered_trajectory})
            best_orderings = dict(sorted(best_orderings.items(), key=lambda x:x[0])) # sort by key
            if len(best_orderings) > num_orderings_kept:
                best_orderings.popitem()

        # Update the current trajectory randomly to one of the best trajectories just found
        probabilities = list(best_orderings.keys())
        # [-200, -4, 0, 52] --> [0, 196, 200, 252]
        probabilities = [float(item) + abs(float(np.min(probabilities))) for item in probabilities]
        # [0, 196, 200, 252] --> [0, 196/252, 200/252, 1]
        probabilities = [item/max(probabilities) if max(probabilities) > 0 else item for item in probabilities]
        probabilities = scipy.special.softmax(probabilities)
        try:
            choice = np.random.choice(list(best_orderings.keys()), 1, p=probabilities)
            data[traj] = best_orderings[float(choice[0])]
        except:
            data[traj] = list(best_orderings.values())[0]

    return data


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


def estimate_sde_parameters(
    data: np.ndarray,
    time_step_size: float,
    T: float,
    true_A: np.ndarray,
    true_H: np.ndarray,
    max_iter: int = 10,
    alpha: float = 0.1
    ) -> tuple[np.ndarray, np.ndarray]:
    """Main iterative scheme:
    1) Reconstruct / reorder segments using the current (A, G).
    2) Re-estimate A, G from the newly-reconstructed data.

    Args:
        data (np.ndarray): Input data, shape (num_trajectories, num_segments, points_per_segment, d)
        time_step_size (float): Step size with respect to time between points.
        T (float): Time period.
        true_A (nd.ndarray): True drift matrix A. Used for computing MAE.
        true_H (nd.ndarray): True (observational) diffusion matrix H (= G*G.T). Used for computing MAE.
        max_iter (int, optional): Maximum number of iterations. Defaults to 10.
        alpha (float, optional): Regularization hyper-parameter. Defaults to 0.1.

    Returns:
        tuple[np.ndarray, np.ndarray]: The estimated drift-diffusion matrices (A, G)
    """
    # 1) Sort all segments in each trajectory by variance
    reordered_data = sort_data_by_var(data, decreasing_var=False) # assuming diverging SDEs
    # reordered_data has shape (num_trajectories, num_segments, steps_per_segment, d)
    # and is now transformed to (num_trajectories, num_steps, d) with
    # num_steps = num_segments * steps_per_segment so that we can use the whole trajectory data
    # to update parameters. After that, it is transformed back to the previous shape to be
    # re-ordered again.
    num_segments = reordered_data.shape[1]
    steps_per_segment = reordered_data.shape[2]
    reordered_data = np.reshape(reordered_data, (num_trajectories,
                                                 num_segments*steps_per_segment, d))

    # 2) Iterative scheme for estimating SDE parameters
    all_nlls = []
    all_mae_a = []
    all_mae_h = []
    best_nll = np.inf
    best_ordered_data = np.copy(reordered_data)
    for iteration in range(max_iter):
        # Update SDE parameters A, G using MLE with the newly completed data
        A, H = update_sde_parameters(reordered_data, dt=time_step_size, T=T)

        # Transform the data
        reordered_data = np.reshape(reordered_data,
                                    (num_trajectories, num_segments,
                                    steps_per_segment, d))
        # Reconstruct / reorder segments to maximize LL - DAG penalty
        reordered_data = reorder_trajectories(reordered_data, A, H, time_step_size, alpha)

        # Transform the data back to its previous shape
        reordered_data = np.reshape(reordered_data, (num_trajectories,
                                                    num_segments*steps_per_segment, d))

        average_nll = 0.0
        for traj in range(num_trajectories):
            average_nll += compute_nll(reordered_data[traj], A, H, dt=time_step_size)

        average_nll = average_nll/num_trajectories
        MAE_A = np.mean(np.abs(A - true_A))
        MAE_H = np.mean(np.abs(H - true_H))
        print(f"Iteration {iteration+1}:\nNLL = {average_nll:.3f}\nMAE to true A = {MAE_A:.3f}\nMAE to true H = {MAE_H:.3f}")

        all_nlls.append(average_nll)
        all_mae_a.append(MAE_A)
        all_mae_h.append(MAE_H)
        if average_nll < best_nll:
            best_nll = average_nll
            best_ordered_data = np.copy(reordered_data)

    return A, H, reordered_data, best_ordered_data, all_nlls, all_mae_a, all_mae_h


def run_experiment(
    T: float, d: int, dt: float,
    num_trajectories: int,
    version: int = 3, max_iter: int = 20):
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
    
    # Generating data
    points = generate_independent_points(d, d)
    X0_dist = [(point, 1 / len(points)) for point in points]
    X_appex = linear_additive_noise_data(
        num_trajectories=num_trajectories, d=d, T=T, dt_EM=dt, dt=dt,
        A=A, G=G, sample_X0_once=True, X0_dist=X0_dist)
    print(X_appex.shape)
    
    # Reshape data from shape (num_trajectories, num_steps, d)
    # to shape (num_trajectories, num_segment, steps_per_segment, d)
    steps_per_segment = 2
    X_appex = X_appex.reshape(X_appex.shape[0], int(round(X_appex.shape[1]/steps_per_segment)),
                           steps_per_segment, X_appex.shape[2])
    print(X_appex.shape)

    # Randomize segments between each trajectory (to get rid of the temporal order between segments)
    random_X = np.zeros((X_appex.shape))
    for i in range(X_appex.shape[0]):
        permutation_id = np.random.permutation(X_appex.shape[1])
        random_X[i] = X_appex[i, permutation_id, :, :]

    # Estimating SDE's parameters A, G
    estimated_A, estimated_H, reordered_X, best_ordered_data, all_nlls, all_mae_a, all_mae_h = \
        estimate_sde_parameters(random_X, time_step_size=0.05, T=T, max_iter=max_iter, true_A=A, true_H=G*G.T)
    all_nlls = [float(item) for item in all_nlls]
    # We'll use Cholesky if H_est is positive definite
    # else fallback to sqrtm or pseudo-chol
    try:
        estimated_G = np.linalg.cholesky(estimated_H)
    except np.linalg.LinAlgError:
        # if not SPD, do a symmetric sqrt
        from scipy.linalg import sqrtm
        estimated_G = sqrtm(estimated_H)
        # If it still fails, consider a small regularization
    print(estimated_A, "A")
    print(estimated_G, "G")

    # Plot results
    X_appex = X_appex.reshape(X_appex.shape[0], X_appex.shape[1]*X_appex.shape[2], X_appex.shape[3])
    random_X = random_X.reshape(random_X.shape[0], random_X.shape[1]*random_X.shape[2], random_X.shape[3])
    num_points = X_appex.shape[1]

    # ---------------------------------------------------------
    # Plot each trajectory (each sub-array along axis=0)
    # ---------------------------------------------------------
    fig = plt.figure(figsize=(10, 10))
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)

    time = np.arange(num_points)
    num_iterations = np.arange(len(all_nlls))
    for i in range(X_appex.shape[0]):
        # data[i] is shape (num_points, 2)
        # Plot one line for each trajectory
        ax1.plot(time, X_appex[i, :, 0], label=f"Trajectory {i}")
        # ax2.plot(time, best_ordered_data[i, :, 0], label=f"Trajectory {i}")
        ax3.plot(time, reordered_X[i, :, 0], label=f"Trajectory {i}")

    ax1.set_xlabel('Time')
    ax1.set_ylabel('Variable 0')
    ax1.set_title("Original data")
    # ax1.legend()

    ax2.plot(num_iterations, all_mae_a, label=f"MAE to True A")
    ax2.plot(num_iterations, all_mae_h, label=f"MAE to True H")
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Mean Absolute Error')
    ax2.set_title("MAE between the estimated and true params")
    ax2.legend()

    ax3.set_xlabel('Time')
    ax3.set_ylabel('Variable 0')
    ax3.set_title("Reconstructed data")
    # ax3.legend()

    ax4.plot(num_iterations, all_nlls)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Negative Log-likelihood')
    ax4.set_title("Negative Log-likelihood through epochs")
    # ax4.legend()

    plt.show()

    return


if __name__ == "__main__":
    # data generated by data_generation.py of APPEX code
    d = 3
    num_trajectories = 5000
    max_iter = 10

    # Run different experiments
    run_experiment(T=0.10, d=d, dt=0.01, num_trajectories=num_trajectories, version=3, max_iter=max_iter)
