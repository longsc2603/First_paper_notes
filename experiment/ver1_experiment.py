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


if __name__ == "__main__":
    # data generated by data_generation.py of APPEX code
    d = 10
    num_trajectories = 10
    A = np.array([[-1]])
    G = np.eye(d)
    points = generate_independent_points(d, d)
    X0_dist = [(point, 1 / len(points)) for point in points]
    X_appex = linear_additive_noise_data(
        num_trajectories=num_trajectories, d=d, T=2.5, dt_EM=0.05, dt=0.05,
        A=A, G=G, X0_dist=X0_dist)
    print(X_appex.shape)

    # # Experiment with Finding all possible sequences of data points (Stage 1)
    # all_possible_sequences, chosen_seq = finding_possible_sequences(X=X_appex)
    # print(len(all_possible_sequences))
    # print(chosen_seq)

    # # Experiment with Direction Prediction through Residuals (Stage 2)
    # start = time.time()
    # num_steps = int(1.25/0.05)   # Time period T / dt_EM - 1
    # for i in range(num_steps):
    #     step_1 = X_appex[:, i, :]
    #     step_2 = X_appex[:, i+1, :]
    #     # forward direction
    #     residual = step_2 - step_1 * 0.5
    #     test_stat_1, threshold1 = Hsic().test(step_1, residual)
    #     # backward direction
    #     residual = step_1 - step_2 * 0.5
    #     test_stat_2, threshold2 = Hsic().test(step_2, residual)
    #     if test_stat_1 < test_stat_2:
    #         print(f"Predicted direction: Step {i} --> {i+1}")
    #     else:
    #         print(f"Predicted direction: Step {i+1} --> {i}")
    # print(f"Time taken for predicting direction: {time.time() - start}")
    # # print(mutual_info_regression(step_1, step_2))