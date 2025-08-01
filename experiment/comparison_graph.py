import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Method names
methods = ['our_mle', 'our_ols', 'our_em', 'mst_mle', 'dpt_mle']
method_line_color = ['red', 'blue', 'green', 'orange', 'black']
markers = ['o', 's', '^', 'D', 'P']

num_methods = len(methods)
num_runs = 3
x_axis = [10, 25, 50, 100, 250, 500]

# Specify data, which setting to plot
data = 'accuracy'     # 'accuracy', 'runtime', 'mae_a', or 'mae_h'
setting = 'noisy'  # 'base' or 'noisy'

# Logged results
if setting == 'base':
    log_folder = 'results/'
elif setting == 'noisy':
    log_folder = 'noise_results/'


# Define figure and axes
plt.figure(figsize=(8, 6))

for id, method in enumerate(methods):
    # Read the CSV file
    df = pd.read_csv(f'{log_folder}{method}.csv')
    
    # Extract the relevant columns
    if setting == 'base':
        num_steps = df['num_steps'].unique()
        num_steps = [str(i) for i in num_steps]
        num_params_setting = len(num_steps)
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
    label_terms = method.split('_')
    label_terms[0] = label_terms[0].capitalize() if label_terms[0] == 'our' else label_terms[0].upper() 
    label_terms[1] = label_terms[1].upper()  # Uppercase the second term 
    label = " + ".join(label_terms)

    if setting == 'base':
        plt.plot(num_steps, mean, label=label, marker=markers[id], color=method_line_color[id], linewidth=2)
        # Plot std as dotted line
        plt.fill_between(num_steps, mean - std, mean + std, color=method_line_color[id], alpha=0.2)
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
    xlabel = 'Number of Timesteps'

plt.title(title, fontsize=16)
plt.xlabel(xlabel, fontsize=14)
plt.ylabel(ylabel, fontsize=14)
plt.legend(loc='upper left')

# Layout
plt.tight_layout()
plt.savefig(f'{log_folder}{data}_{setting}_setting.pdf', dpi=150)
