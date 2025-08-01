import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import time
import os
 
# ------------- Hyperparameters and Functions --------------
args = {
    "seed": 167,
    "n_samples": 500,
    "t_final": 15.0,
    "n_steps": 60,
    "obs_noise": 0.01,
    "bsv_noise": 0.05,        # between-subject std for parameters
    "gamma": 2.0,             # confounding level
    "window_size": 15,
    "max_chemo": 5.0,
    "max_radio": 1.0,
    "init_diam_min": 13,
    "init_diam_max": 15,
}
 
def seed_all(seed=42):
    np.random.seed(seed)
    random.seed(seed)
seed_all(args["seed"])
 
def calc_volume(diameter):
    return 4.0 / 3.0 * np.pi * (diameter / 2.0) ** 3.0
 
# Default parameter means
param_means = {
    'rho': 0.005,
    'K': calc_volume(30),
    'beta_c': 0.03,
    'alpha_r': 0.04,
    'beta_r': 0.004,
    'sigma_tumor': 0.15,
}
 
def sample_subject_params(args, bsv_noise=0.05):
    p = param_means.copy()
    for k in p.keys():
        # Sample log-normal or normal per Appendix F
        p[k] = np.abs(np.random.normal(p[k], bsv_noise * p[k]))
    return p
 
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
 
def get_action(y, gamma, chemo_max, radio_max):
    # Chemotherapy: continuous [0, chemo_max] (more likely with higher tumor)
    # Radiotherapy: binary {0, 1} (more likely with higher tumor)
    # Use gamma for confounding strength
    prob_radio = sigmoid(gamma * (y - 0.5 * args["init_vol_max"]) / args["init_vol_max"])
    radio = np.random.binomial(1, p=prob_radio)
    chemo = np.clip(np.random.normal(y / args["init_vol_max"] * chemo_max, 0.5), 0, chemo_max)
    return chemo, radio
 
def sde_tumor_growth(x, chemo, radio, params):
    """Return drift (dx/dt) for the SDE given state x and treatments."""
    if x < 1e-4: x = 1e-4  # avoid log(0)
    drift = (params['rho'] * np.log(params['K'] / x)
            - params['beta_c'] * chemo
            - (params['alpha_r'] * radio + params['beta_r'] * radio ** 2)) * x
    return drift
 
def simulate_tumor_sde(args, treat_on=True):
    t_eval = np.linspace(0, args["t_final"], args["n_steps"])
    init_vol_min = calc_volume(args["init_diam_min"])
    init_vol_max = calc_volume(args["init_diam_max"])
    args["init_vol_max"] = init_vol_max
 
    trajs = []
    for _ in tqdm(range(args["n_samples"]), desc="Sim subjects"):
        params = sample_subject_params(args, args["bsv_noise"])
        x = np.random.uniform(init_vol_min, init_vol_max)
        y = [x]
        for ti in range(1, len(t_eval)):
            # Assign treatments:
            if treat_on:
                chemo = args["max_chemo"]  # always high chemo
                radio = 1                  # always on
            else:
                chemo = 0                  # always off
                radio = 0
            # Euler–Maruyama step
            dt = t_eval[ti] - t_eval[ti-1]
            drift = sde_tumor_growth(x, chemo, radio, params)
            noise = params['sigma_tumor'] * np.random.normal(0, np.sqrt(dt))
            x = x + drift * dt + noise
            x = max(x, 0.0)
            y.append(x)
        trajs.append(y)
    return t_eval, np.array(trajs)
 
# ----------------- Plotting -----------------
if __name__ == "__main__":
    os.makedirs("figs", exist_ok=True)
 
    # Simulate treated and untreated groups
    t_eval, treat_trajs = simulate_tumor_sde(args, treat_on=True)
    _, notreat_trajs = simulate_tumor_sde(args, treat_on=False)
    print(treat_trajs.shape, notreat_trajs.shape)  # (n_samples, n_steps)
 
    fig, axes = plt.subplots(2, 1, figsize=(5, 4), sharex=True)
    for i in range(args["n_samples"]):
        axes[0].plot(t_eval, treat_trajs[i], color="tab:red", alpha=0.5, lw=1)
        axes[1].plot(t_eval, notreat_trajs[i], color="tab:blue", alpha=0.5, lw=1)
    axes[0].set_title("Tumor SDE Trajectories (Chemo+Radio ON)")
    axes[1].set_title("Tumor SDE Trajectories (Chemo+Radio OFF)")
    axes[1].set_xlabel("Time")
    axes[0].set_ylabel("Tumor Volume $X_t$")
    axes[1].set_ylabel("Tumor Volume $X_t$")
    fig.tight_layout()
    plt.savefig("figs/tumor_treat-notreat.pdf", dpi=150)
    plt.close(fig)
    print("Saved plot to figs/tumor_treat-notreat.pdf")
 