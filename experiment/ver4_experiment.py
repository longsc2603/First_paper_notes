import numpy as np
import matplotlib.pyplot as plt
import itertools
import scipy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math

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
    A_sq = np.multiply(torch.clone(A).detach().numpy(), torch.clone(A).detach().numpy())
    # Exponentiate elementwise
    exp_A_sq = scipy.linalg.expm(A_sq)
    trace_val = np.trace(exp_A_sq)
    penalty = alpha*(trace_val - d)

    return penalty


# TODO: Try some other convergence scores besides NLL
def compute_nll(X: np.ndarray, A: np.ndarray, G: np.ndarray, dt: float) -> float:
    """Compute the negative log-likelihood of the current segment under the
    current inferred model parameters.

    Args:
        X (np.ndarray): Segments, shape (num_time_steps, d)
        A (np.ndarray): Estimated drift matrix, shape (d, d)
        G (np.ndarray): Estimated diffusion matrix, shape (d, m)
        dt (float): Time step size

    Returns:
        float: Total NLL of the segment
    """
    nll = 0.0
    if len(X.shape) == 3:
        # this is the case of shape (num_segments, steps_per_segment, d)
        X = torch.reshape(X, (X.shape[0]*X.shape[1], X.shape[2]))
    num_steps, d = X.shape
    H_dt = G@G.T*dt
    H_dt_inv = G/dt # (H*dt)^-1 = dt^-1 * H^-1
    # quick fix, instead of getting logdet of H_dt, which
    # is -inf due to det(H_dt) = 0, we get logdet of H_dt + eps*I
    # This ensures that H_dt_reg = H + eps*I is full rank and invertible.
    eps = 1e-2
    H_dt_reg = H_dt + eps*torch.eye(H_dt.shape[0])
    # Precompute determinant of H_dt
    sign, logdet_H_dt = torch.linalg.slogdet(H_dt_reg)
    # logdet_G_dt = 2*torch.log(torch.linalg.det(G + eps*torch.eye(G.shape[0]))) + np.log(np.pow(dt, d))
    # logdet_G_dt = 2*torch.log(torch.linalg.det(G)) + np.log(np.pow(dt, d))
    const_term = 0.5 * (d * np.log(2 * np.pi) + logdet_H_dt)
    # const_term_G = 0.5 * (d * np.log(2 * np.pi) + logdet_G_dt)
    for t in range(num_steps - 1):
        X_t = X[t]
        X_tp1 = X[t + 1]
        # Compute mean: mu_t = e^{A dt}  X_t (approximate if needed)
        # mu_t = torch.exp(A*dt)@X_t
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


def reorder_trajectories(
    data: np.ndarray,
    A: np.ndarray,
    G: np.ndarray,
    time_step_size: float,
    alpha: float = 0.5,
    known_initial_value: bool = False
    ) -> np.ndarray:
    """Reordering trajectory data to maximize our objective (log-likelihood minus DAG penalty),
    which is minimizing (negative log-likelihood plus DAG penalty)

    Args:
        data (np.ndarray): Arrays of trajectory to be re-ordered,
            shape (num_trajectories, num_segments, points_per_segment, d).
        A (np.ndarray): Current estimated drift matrix, shape (d, d).
        G (np.ndarray): Current estimated diffusion matrix, shape (d, m).
        time_step_size (float): Step size with respect to time between points.
        alpha (float, optional): Regularization hyper-parameter. Defaults to 0.1.
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

        for candidate_order in all_indices_permutations:
            # candidate_order is a list of 2D arrays, now we stack it 
            # to get the trajectory to compute NLL
            reordered_trajectory = torch.stack(candidate_order, axis=0)
            # Evaluate the negative log-likelihood
            nll = compute_nll(reordered_trajectory, A, G, time_step_size)
            # nlp = compute_nlp(nll=nll, A=A, H=H)
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
        # print([item.item() for item in list(best_orderings.keys())])
        # Update the current trajectory randomly to one of the best trajectories just found
        probabilities = list(best_orderings.keys())
        probabilities = [torch.clone(item).detach().numpy() for item in probabilities]
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


class DriftNetwork(nn.Module):
    def __init__(self, d, init_scale=0.01):
        super().__init__()
        # A is a learnable parameter matrix of shape (d, d)
        A_init = init_scale * torch.randn(d, d)
        self.A = nn.Parameter(A_init)

    def forward(self):
        # Returns the drift matrix A of shape (d, d)

        return self.A


class DiffusionNetwork(nn.Module):
    def __init__(self, d, m=1, init_scale=0.01):
        super().__init__()
        # G is a learnable parameter matrix of shape (d, m)
        G_init = init_scale * torch.randn(d, m)
        self.G = nn.Parameter(G_init)

    def forward(self):
        # Returns the drift matrix G of shape (d, m)

        return self.G


def estimate_sde_parameters(
    data: np.ndarray,
    drift_net, diff_net, original_data,
    time_step_size: float,
    T: float,
    true_A: np.ndarray,
    true_G: np.ndarray,
    lr: float = 5e-2,
    max_iter: int = 10,
    alpha: float = 0.1,
    known_initial_value: bool = False
    ) -> tuple[DriftNetwork, DiffusionNetwork]:
    """Main iterative scheme:
    1) Reconstruct / reorder segments using the current (A, G).
    2) Re-estimate A, G from the newly-reconstructed data.

    Args:
        data (np.ndarray): Input data, shape (num_trajectories, num_segments, points_per_segment, d)
        time_step_size (float): Step size with respect to time between points.        
        drift_net: NN for A
        diff_net:  NN for G
        T (float): Time period.
        true_A (nd.ndarray): True drift matrix A. Used for computing MAE.
        true_G (nd.ndarray): True diffusion matrix G. Used for computing MAE.
        lr (float): Learning rate.
        max_iter (int, optional): Maximum number of iterations. Defaults to 10.
        alpha (float, optional): Regularization hyper-parameter. Defaults to 0.1.
        known_initial_value (bool, optional): If True, the initial values of trajectories
            are known beforehand, and should stay fixed. Defaults to False.

    Returns:
        tuple[DriftNetwork, DiffusionNetwork]: The estimated NN for the matrices (A, G)
    """
    optimizer = optim.Adam(
        list(drift_net.parameters()) + list(diff_net.parameters()),
        lr=lr,
        betas=(0.9, 0.99)
    )
    # 1) Sort all segments in each trajectory by variance
    reordered_data = sort_data_by_var(data, decreasing_var=False,
                                      known_initial_value=known_initial_value) # assuming diverging SDEs
    # reordered_data has shape (num_trajectories, num_segments, steps_per_segment, d)
    # and is now transformed to (num_trajectories, num_steps, d) with
    # num_steps = num_segments * steps_per_segment so that we can use the whole trajectory data
    # to update parameters. After that, it is transformed back to the previous shape to be
    # re-ordered again.
    num_segments = reordered_data.shape[1]
    steps_per_segment = reordered_data.shape[2]
    reordered_data = torch.tensor(reordered_data)
    reordered_data = torch.reshape(reordered_data, (num_trajectories,
                                                 num_segments*steps_per_segment, d))

    right_percent_through_traj = []
    for traj in range(num_trajectories):
        right_value = ((original_data[traj] - reordered_data[traj].detach().numpy()) == 0.0).astype(int)
        right_value_count = (right_value == 1).sum()
        right_value_count /= original_data[traj].shape[0]*original_data[traj].shape[1]
        right_percent_through_traj.append(right_value_count)
    right_percent = np.mean(right_percent_through_traj)
    print(right_percent)

    # 2) Iterative scheme for estimating SDE parameters
    all_nlls = []
    all_nlps = []
    all_mae_a = []
    all_mae_g = []
    all_right_percent = []
    best_nll = np.inf
    best_nlp = np.inf
    best_ordered_data = torch.clone(reordered_data)
    A = drift_net().double()
    G = diff_net().double()
    for iteration in range(max_iter):
        # Update SDE parameters A, G using MLE with the newly completed data
        optimizer.zero_grad()
        average_nll = 0.0
        average_nlp = 0.0
        for traj in range(num_trajectories):
            nll = compute_nll(reordered_data[traj], A, G, dt=time_step_size)
            average_nll += nll
            # average_nlp += compute_nlp(nll=nll, A=A, H=G@G.T)

        average_nll = average_nll/num_trajectories
        # average_nlp = average_nlp/num_trajectories

        average_nll.backward()
        torch.nn.utils.clip_grad_norm_(list(drift_net.parameters()) + list(diff_net.parameters()), 100.0)
        for name, p in drift_net.named_parameters():
            p_before = torch.clone(p.data)
            print(name, 'before: ', p.data)
            print(name, 'grad: ', p.grad)
        optimizer.step()
        for name, h in drift_net.named_parameters():
            print(name, 'after: ', h.data)
            print("Change:", (h.data - p_before))

        MAE_A = np.mean(np.abs(torch.clone(A).detach().numpy() - true_A))
        MAE_G = np.mean(np.abs(torch.clone(G).detach().numpy() - true_G))
        print(f"Iteration {iteration+1}:\nNLL = {average_nll:.3f}\nMAE to true A = {MAE_A:.3f}\nMAE to true G = {MAE_G:.3f}")
        
        all_nlls.append(average_nll)
        # all_nlps.append(average_nlp)
        all_mae_a.append(MAE_A)
        all_mae_g.append(MAE_G)
        if average_nll < best_nll:
            best_nll = average_nll
            best_ordered_data = torch.clone(reordered_data)
        # if average_nlp < best_nlp:
        #     best_nlp = average_nlp
        #     best_ordered_data = np.copy(reordered_data)
        
        # Transform the data
        reordered_data = torch.reshape(reordered_data,
                                    (num_trajectories, num_segments,
                                    steps_per_segment, d))
        # Reconstruct / reorder segments to maximize LL - DAG penalty
        reordered_data = reorder_trajectories(reordered_data, A, G, time_step_size, alpha,
                                              known_initial_value=known_initial_value)

        # Transform the data back to its previous shape
        reordered_data = torch.reshape(reordered_data, (num_trajectories,
                                                    num_segments*steps_per_segment, d))

        right_percent_through_traj = []
        for traj in range(num_trajectories):
            right_value = ((original_data[traj] - reordered_data[traj].detach().numpy()) == 0.0).astype(int)
            right_value_count = (right_value == 1).sum()
            right_value_count /= original_data[traj].shape[0]*original_data[traj].shape[1]
            right_percent_through_traj.append(right_value_count)
        right_percent = np.mean(right_percent_through_traj)
        all_right_percent.append(right_percent)

    return A, G, reordered_data, best_ordered_data, all_nlls, all_nlps, all_mae_a, all_mae_g, all_right_percent


def run_experiment(
    T: float, d: int, dt: float, lr: float,
    num_trajectories: int,
    version: int = 3, max_iter: int = 20,
    known_initial_value: bool = False):
    """Running experiments and plotting the results

    Args:
        T (float): Time period.
        d (int): Variables dimension.
        dt (float): Time step size.
        lr (float): Learning rate.
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
    drift_net = DriftNetwork(d=d)
    diff_net = DiffusionNetwork(d=d, m=d)
    if version == 1:
        A = np.zeros((d, d))
        G = np.ones((d, d)) * 5
    elif version == 2:
        # our_data = data_masking(our_data)
        pass
    elif version == 3:
        A = np.ones((d, d)) * 15
        G = np.ones((d, d)) * 7.5
    
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
        if known_initial_value:
            permutation_id = np.random.permutation(np.arange(1, X_appex.shape[1]))
            permutation_id = np.concatenate(([0], permutation_id))
            random_X[i] = X_appex[i, permutation_id, :, :]
        else:
            permutation_id = np.random.permutation(X_appex.shape[1])
            random_X[i] = X_appex[i, permutation_id, :, :]

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
    estimated_A, estimated_G, reordered_X, best_ordered_data, all_nlls, all_nlps, all_mae_a, all_mae_g, all_right_percent = \
        estimate_sde_parameters(random_X, drift_net, diff_net, X_appex, time_step_size=0.05, T=T,
                                max_iter=max_iter, true_A=A, true_G=G,
                                lr=lr, known_initial_value=known_initial_value)
    all_nlls = [float(item) for item in all_nlls]

    print(estimated_A, "A")
    print(estimated_G, "G")

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
    num_iterations = np.arange(len(all_nlls))
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
    ax2_bottom.plot(num_iterations, all_mae_g, label=f"MAE to True G")
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
    
    plt.show()

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot()
    ax.plot(num_iterations, all_right_percent)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MAE')
    ax.set_title("MAE between the original and reordered data")

    return


if __name__ == "__main__":
    # data generated by data_generation.py of APPEX code
    d = 2
    num_trajectories = 500
    max_iter = 5
    known_initial_value = True

    # Run different experiments
    run_experiment(T=0.1, d=d, dt=0.01, lr=5e-2, num_trajectories=num_trajectories,
                   version=3, max_iter=max_iter, known_initial_value=known_initial_value)

    """
    Ver 4: Data without Temporal Order, but A and G are approximated with Neural Networks
    Solution: First sort by variance, then do an iterative scheme with 2 steps, updating SDE's
    parameters and re-sorting data based on the current params.
    """