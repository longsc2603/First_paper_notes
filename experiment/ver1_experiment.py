import numpy as np
import time
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LinearRegression
from hyppo.independence import Hsic

from experiments import generate_independent_points, linear_additive_noise_data


# predicting the time dimension
def time_dimension_prediction(X: np.ndarray) -> np.ndarray:
    """Experimenting with predicting the time dimension of the whole dataset.
    For example, input data with specified time steps could be of the form:
        (num_trajectories, num_steps, d)    - d being the dimension of the SDE's drift
    But we may encounter data with no time dimension, which means that there is no
    temporal order in the dataset. Data points at different time steps are ordered
    randomly.
    Initial idea is to divide to 2 stages:
    + Stage 1: Find possible sequences of data points without thinking about their direction.
        Doing this by calculating distances between every pair of points, mark a point j as 
        a possible next point of point i by checking if the distance is smaller than some
        hyper-parameter epsilon.
        There may be many possible sequences, we will choose the one with smallest total length
        to be our prediction.
    + Stage 2: Find direction of the predicted sequence. Not sure yet how to do this, need to
        read more related work...

    Args:
        X (numpy.ndarray): 3D array of trajectories with shape
            (num_trajectories, num_steps, d)

    Returns:
        timely_arranged_X: 3D array of trajectories with shape
            (num_trajectories, num_steps, d), but the data points are in correct order.
    """
    num_trajectories = X.shape[0]
    timely_arranged_X = []

    # code for finding out num_steps


    return timely_arranged_X


def calculating_distance(X: np.ndarray) -> np.ndarray:
    """Calculating L2-distance between all pairs of points given in X

    Args:
        X (np.ndarray): 2D arrays of points with shape (num_steps, d). Each step has one point.

    Returns:
        distances (np.ndarray): 2D arrays with shape (num_steps, num_steps). Value [i][j]
            is a scalar showing L2-distance between the points at row i and row j of X.
            The diagonal values are zeros.
    """
    num_steps = X.shape[0]
    distances = np.zeros((num_steps, num_steps))
    
    for i in range(X.shape[0]):
        for j in range(X.shape[0]):
            if i == j or distances[i][j] != 0:
                continue
            distances[i][j] = np.sqrt(np.sum(np.square(X[i, :] - X[j, :]))) # L2-distance

    return distances


def back_tracking(
    num_steps: int,
    all_pair_distances: np.ndarray,
    first_step: bool = False,
    epsilon: float = 1e-3):
    """_summary_

    Args:
        num_steps (int): _description_
        all_pair_distances (np.ndarray): _description_
        first_step (bool, optional): _description_. Defaults to False.
        epsilon (float, optional): _description_. Defaults to 1e-3.

    Returns:
        _type_: _description_
    """

    global all_sequences, sequence

    if len(sequence) == num_steps:
        return sequence

    if not first_step:
        next_possible_points = np.nonzero( \
            np.where((all_pair_distances[sequence[-1], :] > 0) & (all_pair_distances[sequence[-1], :] < epsilon), \
                     all_pair_distances[sequence[-1], :], 0))
        next_possible_points = next_possible_points[0].tolist()
        for point in next_possible_points:
            if point in sequence:
                next_possible_points.remove(point)

    for i in range(all_pair_distances.shape[0]) if first_step else next_possible_points:
        if i in sequence:
            continue
        # next possible points are points with distances smaller than epsilon and larger than 0
        # with 0 being their self-distance
        next_possible_points = np.nonzero( \
            np.where((all_pair_distances[i, :] > 0) & (all_pair_distances[i, :] < epsilon), \
                     all_pair_distances[i, :], 0))
        next_possible_points = next_possible_points[0].tolist()
        for point in next_possible_points:
            if point in sequence:
                next_possible_points.remove(point)
        if len(next_possible_points) == 0:
            continue
        
        sequence.append(int(i))
        possible_sequence = back_tracking(num_steps, all_pair_distances, first_step=False, epsilon=epsilon)
        if len(possible_sequence) == num_steps:
            all_sequences.append(sequence.copy())

        sequence.pop()
    
    return []


all_sequences = []
sequence = []


def finding_possible_sequences(X: np.ndarray) -> list:
    """Find all sequences of data points that can be the true order sequence.

    Args:
        X (np.ndarray): Input data points, no temporal order.

    Returns:
        possible_sequences (list): All possible sequences.
    """
    print(X.shape)
    # temp fix to deal with only the first trajectory first
    X = X[0]
    global sequence
    all_pair_distances = calculating_distance(X)
    num_steps = all_pair_distances.shape[0]
    epsilon_quantile = 0.15
    while len(all_sequences) == 0:
        start = time.time()
        sequence = []
        epsilon = get_epsilon(all_pair_distances, epsilon_quantile)
        _ = back_tracking(num_steps, all_pair_distances, first_step=True, epsilon=epsilon)
        print(f"Epsilon: {epsilon:.4f}\tQuantile: {epsilon_quantile:.4f}\tTime:{(time.time() - start):.4f}\tNumber of sequences: {len(all_sequences)}")
        
        bad_epsilon_quantile = 1
        if len(all_sequences) >= 100:
            bad_epsilon_quantile = epsilon_quantile
            epsilon_quantile -= 3e-3
            all_sequences.clear()
        elif len(all_sequences) >= 50:
            bad_epsilon_quantile = epsilon_quantile
            epsilon_quantile -= 1e-3
            all_sequences.clear()
        else:
            epsilon_quantile += 7.5e-3
            if epsilon_quantile >= bad_epsilon_quantile:
                epsilon_quantile -= 2.5e-3

    smallest_length = 1e5
    for seq in all_sequences:
        length = 0
        for i in range(len(seq) - 1):
            length += all_pair_distances[seq[i], seq[i+1]]
        if length < smallest_length:
            chosen_seq = seq

    return all_sequences, chosen_seq


def get_epsilon(all_pair_distances: np.ndarray, quantile_value: float=0.15) -> float:
    """Function for determining the param epsilon. First naive idea: use value at 
    quantile 0.75 of all the distances.

    Args:
        all_pair_distances (np.ndarray): 2D arrays with shape (num_steps, num_steps).
            Value [i][j] is a scalar showing L2-distance between the points at row i
            and row j of X. The diagonal values are zeros.
        quantile_value (float, optional): Determining what quantile value to get our epsilon.
            Defaults to 0.15. This quantile value is added 2e-3 each time the algo fails
            to find possible sequences.

    Returns:
        epsilon: Parameter used in function finding_possible_sequences above.
    """
    # # Mean value
    # epsilon = np.sum(all_pair_distances)/float(num_steps*num_steps - num_steps)

    # Using quantile value for epsilon
    epsilon = np.quantile(all_pair_distances, quantile_value)

    return epsilon


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


if __name__ == "__main__":
    # data generated by data_generation.py of APPEX code
    d = 1
    T = 0.5
    dt = 0.01  # SDE step size
    num_trajectories = 1000
    A = np.random.randn(d, d)  # Random drift matrix
    G = np.random.randn(d, d)  # Random diffusion matrix
    points = generate_independent_points(d, d)
    X0_dist = [(point, 1 / len(points)) for point in points]
    X_appex = linear_additive_noise_data(
        num_trajectories=num_trajectories, d=d, T=T, dt_EM=dt, dt=dt,
        A=A, G=G, X0_dist=X0_dist)
    print(X_appex.shape)

    # Randomize segments between each trajectory (to get rid of the temporal order between segments)
    random_X, permutation_id = randomize_data(X_appex)

    right_percent = check_sorting_accuracy(X_appex, random_X, check_by_indices_order=True,
                                            reordered_order=permutation_id)
    print(right_percent)

    # # Experiment with Finding all possible sequences of data points (Stage 1)
    # all_possible_sequences, chosen_seq = finding_possible_sequences(X=X_appex)
    # print(len(all_possible_sequences))
    # print(chosen_seq)

    # Experiment with Direction Prediction through Residuals (Stage 2)
    time_start = time.time()
    num_steps = int(T/dt)   # Time period T / dt_EM - 1
    
    end = num_steps - 1
    start = 0
    order = permutation_id.copy()
    while end > start:
        for i in range(start, end):
            step_1 = random_X[:, i, :]
            if i + 1 >= num_steps:
                break
            step_2 = random_X[:, i+1, :]
            # forward direction
            residual = step_2 - step_1 * 0.5
            test_stat_1, threshold1 = Hsic().test(step_1, residual)
            # backward direction
            residual = step_1 - step_2 * 0.5
            test_stat_2, threshold2 = Hsic().test(step_2, residual)
            if test_stat_1 < test_stat_2:
                # print(f"Predicted direction: Step {i} --> {i+1}")
                temp = np.copy(step_1)
                random_X[:, i, :] = step_2
                random_X[:, i+1, :] = temp
                # change indices of order
                temp = np.copy(order[i])
                order[i] = np.copy(order[i+1])
                order[i+1] = np.copy(temp)
            else:
                # print(f"Predicted direction: Step {i+1} --> {i}")
                pass
            
            if i == end - 1:
                end = i
    
    print(f"Time taken for predicting direction: {time.time() - time_start}")
    right_percent = check_sorting_accuracy(X_appex, random_X, check_by_indices_order=True,
                                            reordered_order=order)
    print(right_percent)
    # print(mutual_info_regression(step_1, step_2))