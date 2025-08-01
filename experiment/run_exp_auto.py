import os

# Path to the Python file you want to run
script_path = "main_setting.py"
param_est_method = "mle"
sorting_method = "our"
setting = 'base'
result_log_file = "results/our_mle.csv"

if setting == 'base':
    # Base version
    T = [0.1, 0.25, 0.5, 1, 2.5, 5]
    for t in range(len(T)):
        for seed in range(3):
            # Arguments to pass (as a list of strings)
            random_seed = [167, 16, 17]
            args = f"--T {T[t]}" +  " --random_seed " + str(random_seed[seed]) \
                + f" --param_est_method {param_est_method}" \
                + f" --sorting_method {sorting_method}" \
                + f" --setting {setting}" \
                + f" --result_log_file {result_log_file}"

            # Build the command: ['python', script_path, arg1, arg2, ...]
            command = f"python {script_path} {args}"

            # Run the script and capture output
            os.system(command)


elif setting == 'noisy':
    # Noise version
    noise = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
    for t in range(len(noise)):
        for seed in range(3):
            # Arguments to pass (as a list of strings)
            random_seed = [167, 16, 17]
            args = "--T 0.5" +  " --random_seed " + str(random_seed[seed]) \
                + " --noisy_measurements_sigma " + str(noise[t]) \
                + f" --param_est_method {param_est_method}" \
                + f" --sorting_method {sorting_method}" \
                + f" --setting {setting}" \
                + f" --result_log_file {result_log_file}"

            # Build the command: ['python', script_path, arg1, arg2, ...]
            command = f"python {script_path} {args}"

            # Run the script and capture output
            os.system(command)
        