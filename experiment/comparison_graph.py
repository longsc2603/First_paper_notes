import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# # Method names
# methods = ['our_mle', 'our_ols', 'our_em', 'mst_mle', 'dpt_mle']
# method_line_color = ['red', 'blue', 'green', 'orange', 'black']
# markers = ['o', 's', '^', 'D', 'P']

# num_methods = len(methods)
# num_runs = 3
# x_axis = [10, 25, 50, 100, 250, 500]

# # Specify data, which setting to plot
# data = 'accuracy'     # 'accuracy', 'runtime', 'mae_a', or 'mae_h'
# setting = 'noisy'  # 'base' or 'noisy'

# # Logged results
# if setting == 'base':
#     log_folder = 'results/'
# elif setting == 'noisy':
#     log_folder = 'noise_results/'


# # Define figure and axes
# plt.figure(figsize=(8, 6))

# for id, method in enumerate(methods):
#     # Read the CSV file
#     df = pd.read_csv(f'{log_folder}{method}.csv')
    
#     # Extract the relevant columns
#     if setting == 'base':
#         num_steps = df['num_steps'].unique()
#         num_steps = [str(i) for i in num_steps]
#         num_params_setting = len(num_steps)
#     elif setting == 'noisy':
#         noise_level = df['noisy_measurements_sigma'].unique()
#         noise_level = [str(i) for i in noise_level]
#         num_params_setting = len(noise_level)
    
#     logging_data = df[data].values.reshape(num_params_setting, num_runs)
#     if data == 'accuracy':
#         logging_data = logging_data * 100  # Convert to percentage if accuracy
#     elif data == 'mae_a' or data == 'mae_h':
#         logging_data = np.log(logging_data)  # Convert to log scale if MAE

#     mean = logging_data.mean(axis=1)
#     std = logging_data.std(axis=1)
#     label_terms = method.split('_')
#     label_terms[0] = label_terms[0].capitalize() if label_terms[0] == 'our' else label_terms[0].upper() 
#     label_terms[1] = label_terms[1].upper()  # Uppercase the second term 
#     label = " + ".join(label_terms)

#     if setting == 'base':
#         plt.plot(num_steps, mean, label=label, marker=markers[id], color=method_line_color[id], linewidth=2)
#         # Plot std as dotted line
#         plt.fill_between(num_steps, mean - std, mean + std, color=method_line_color[id], alpha=0.2)
#     elif setting == 'noisy':
#         plt.plot(noise_level, mean, label=label, marker=markers[id], color=method_line_color[id], linewidth=2)
#         # Plot std as dotted line
#         plt.fill_between(noise_level, mean - std, mean + std, color=method_line_color[id], alpha=0.2)


# if data == 'accuracy':
#     title = 'Sorting Accuracy Across Runs'
#     ylabel = 'Accuracy (%)'
# elif data == 'runtime':
#     title = 'Iteration Runtime Across Runs'
#     ylabel = 'Runtime (s)'
# elif data == 'mae_a':
#     title = 'MAE of A (Log-scale) Across Runs'
#     ylabel = 'MAE of A (Log-scale)'
# elif data == 'mae_h':
#     title = 'MAE of H (Log-scale) Across Runs'
#     ylabel = 'MAE of H (Log-scale)'

# if setting == 'noisy':
#     xlabel = 'Noise Level'
# elif setting == 'base':
#     xlabel = 'Number of Timesteps'

# plt.title(title, fontsize=16)
# plt.xlabel(xlabel, fontsize=14)
# plt.ylabel(ylabel, fontsize=14)
# plt.legend(loc='upper left')

# # Layout
# plt.tight_layout()
# plt.savefig(f'{log_folder}{data}_{setting}_setting.pdf', dpi=150)




# Method names
methods = ['our_mle', 'mst_mle', 'dpt_mle']
method_line_color = ['red', 'blue', 'black']
markers = ['o', 's', 'P']

num_methods = len(methods)
num_runs = 5

# Specify data, which setting to plot
data = 'accuracy'     # 'accuracy', 'runtime', 'mae_a', or 'mae_h'
setting = 'base'  # 'base' or 'noisy'

# Logged results
if setting == 'base':
    log_folder = 'results/'
elif setting == 'noisy':
    log_folder = 'noise_results/'


# Define figure and axes
plt.figure(figsize=(8, 6))

for id, method in enumerate(methods):
    # Read the CSV file
    df = pd.read_csv(f'{log_folder}{method}_large_samples.csv')
    
    # Extract the relevant columns
    if setting == 'base':
        num_steps = df['num_trajectories'].unique()
        num_steps = [str(i) for i in num_steps]
        num_params_setting = len(num_steps)
        # num_steps = [0] + num_steps  # Add 0 for the base case
    elif setting == 'noisy':
        noise_level = df['noisy_measurements_sigma'].unique()
        noise_level = [str(i) for i in noise_level]
        num_params_setting = len(noise_level)
    
    logging_data = df[data].values.reshape(num_params_setting, num_runs)
    if data == 'accuracy':
        logging_data = logging_data * 100  # Convert to percentage if accuracy
    elif data == 'mae_a' or data == 'mae_h':
        logging_data = np.log(logging_data)  # Convert to log scale if MAE

    mean = logging_data.mean(axis=1)
    std = logging_data.std(axis=1)
    print(std, "std", method)
    low_std = mean - std
    high_std = mean + std
    # mean = np.insert(mean, 0, 0.0)
    # low_std = np.insert(low_std, 0, 0.0).astype(float)  # Add 0 for the base case
    # high_std = np.insert(high_std, 0, 0.0).astype(float)  # Add 0 for the base case
    print(mean, "mean", method)
    label_terms = method.split('_')
    label_terms[0] = label_terms[0].capitalize() if label_terms[0] == 'our' else label_terms[0].upper() 
    label_terms[1] = label_terms[1].upper()  # Uppercase the second term 
    label = " + ".join(label_terms)

    if setting == 'base':
        plt.plot(num_steps, mean, label=label, marker=markers[id], color=method_line_color[id], linewidth=2)
        # Plot std as dotted line
        plt.fill_between(np.array(num_steps), low_std, high_std, color=method_line_color[id], alpha=0.2)
        
    elif setting == 'noisy':
        plt.plot(noise_level, mean, label=label, marker=markers[id], color=method_line_color[id], linewidth=2)
        # Plot std as dotted line
        plt.fill_between(noise_level, mean - std, mean + std, color=method_line_color[id], alpha=0.2)


if data == 'accuracy':
    title = 'Sorting Accuracy Across Runs'
    ylabel = 'Accuracy (%)'
elif data == 'runtime':
    title = 'Iteration Runtime Across Runs'
    ylabel = 'Runtime (s)'
elif data == 'mae_a':
    title = 'MAE of A (Log-scale) Across Runs'
    ylabel = 'MAE of A (Log-scale)'
elif data == 'mae_h':
    title = 'MAE of H (Log-scale) Across Runs'
    ylabel = 'MAE of H (Log-scale)'

if setting == 'noisy':
    xlabel = 'Noise Level'
elif setting == 'base':
    xlabel = 'Number of Samples'

plt.title(title, fontsize=16)
plt.xlabel(xlabel, fontsize=14)
plt.ylabel(ylabel, fontsize=14)
plt.legend(loc='best')

# Layout
plt.tight_layout()
plt.savefig(f'{log_folder}{data}_large_samples.pdf', dpi=150)




# # Method names
# methods = ['our_mle', 'mst_mle', 'dpt_mle']
# method_line_color = ['red', 'blue', 'black']
# markers = ['o', 's', 'P']

# num_methods = len(methods)
# num_runs = 5
# x_axis = [60, 80, 100]

# # Specify data, which setting to plot
# data = 'rmse'     # 'teb' or 'rmse
# setting = 'base'  # 'base' or 'noisy'

# # Logged results
# if setting == 'base':
#     log_folder = 'results/'
# elif setting == 'noisy':
#     log_folder = 'noise_results/'


# # Define figure and axes
# plt.figure(figsize=(5, 2.5))

# for id, method in enumerate(methods):
#     # Read the CSV file
#     df = pd.read_csv(f'{log_folder}{data}.csv')
    
#     # Extract the relevant columns
#     if setting == 'base':
#         num_steps = df['num_steps'].unique()
#         num_steps = [str(i) for i in num_steps]
#         num_params_setting = len(num_steps)
#     elif setting == 'noisy':
#         noise_level = df['noisy_measurements_sigma'].unique()
#         noise_level = [str(i) for i in noise_level]
#         num_params_setting = len(noise_level)
    
#     method_names = df['method_name'].unique()

#     if data == 'teb':
#         logging_data = df['treatment_effect_bias'].values.reshape(len(method_names), num_params_setting, num_runs)
#     elif data == 'rmse':
#         logging_data = df['rmse'].values.reshape(len(method_names), num_params_setting, num_runs)
#     # logging_data = np.log(logging_data)  # Convert to log scale if MAE

#     mean = logging_data.mean(axis=2)
#     std = logging_data.std(axis=2)
#     print(mean.shape, std.shape)
#     label_terms = method.split('_')
#     label_terms[0] = label_terms[0].capitalize() if label_terms[0] == 'our' else label_terms[0].upper() 
#     label_terms[1] = label_terms[1].upper()  # Uppercase the second term 
#     label = " + ".join(label_terms)

#     if setting == 'base':
#         plt.plot(num_steps, mean[id, :], label=label, marker=markers[id], color=method_line_color[id], linewidth=2)
#         # Plot std as dotted line
#         plt.fill_between(num_steps, mean[id, :] - std[id, :] + np.ones(std[id, :].shape)*1.5e-14, \
#                          mean[id, :] + std[id, :] - np.ones(std[id, :].shape)*1.5e-14, color=method_line_color[id], alpha=0.2)
#     elif setting == 'noisy':
#         plt.plot(noise_level, mean[id, :], label=label, marker=markers[id], color=method_line_color[id], linewidth=2)
#         # Plot std as dotted line
#         plt.fill_between(noise_level, mean[id, :] - std[id, :], mean[id, :] + std[id, :], color=method_line_color[id], alpha=0.2)


# if data == 'accuracy':
#     title = 'Sorting Accuracy Across Runs'
#     ylabel = 'Accuracy (%)'
# elif data == 'runtime':
#     title = 'Iteration Runtime Across Runs'
#     ylabel = 'Runtime (s)'
# elif data == 'mae_a':
#     title = 'MAE of A (Log-scale) Across Runs'
#     ylabel = 'MAE of A (Log-scale)'
# elif data == 'mae_h':
#     title = 'MAE of H (Log-scale) Across Runs'
#     ylabel = 'MAE of H (Log-scale)'
# elif data == 'teb':
#     title = 'Treatment Effect Bias (TEB) Across Runs'
#     ylabel = 'TEB'
# elif data == 'rmse':
#     title = 'Root Mean Square Error (RMSE) Across Runs'
#     ylabel = 'RMSE'

# if setting == 'noisy':
#     xlabel = 'Noise Level'
# elif setting == 'base':
#     xlabel = 'Number of Timesteps'

# # plt.title(title)
# # plt.xlabel(xlabel)
# plt.ylabel(ylabel)
# plt.legend(loc='best')

# # Layout
# plt.tight_layout()
# plt.savefig(f'{log_folder}{data}.pdf', dpi=150)