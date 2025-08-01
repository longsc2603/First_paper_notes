import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# ---- Hyperparameters for PKPD (One-compartment) ----
args_pkpd = {
    "seed": 167,
    "n_samples": 500,
    "t_final": 15.0,
    "n_steps": 60,
    "obs_noise": 0.5,
    "C0": 0.05,      # Decay rate for untreated
    "C1": 0.15,      # Decay rate for treated
    "V": 1.0,        # Volume parameter
    "init_diam_min": 13,
    "init_diam_max": 15,
}

def seed_all(seed=42):
    np.random.seed(seed)
seed_all(args_pkpd["seed"])

def calc_volume(diameter):
    return 4.0 / 3.0 * np.pi * (diameter / 2.0) ** 3.0

def simulate_pkpd(args, treat_on=True):
    t_eval = np.linspace(0, args["t_final"], args["n_steps"])
    init_vol_min = calc_volume(args["init_diam_min"])
    init_vol_max = calc_volume(args["init_diam_max"])

    # Decay parameter based on treatment
    if treat_on:
        decay_rate = args["C1"] / args["V"]
    else:
        decay_rate = args["C0"] / args["V"]

    trajs = []
    for _ in tqdm(range(args["n_samples"]), desc="Sim PKPD subjects"):
        x = np.random.uniform(init_vol_min, init_vol_max)
        y = [x]
        for ti in range(1, len(t_eval)):
            dt = t_eval[ti] - t_eval[ti-1]
            drift = decay_rate * x
            noise = args["obs_noise"] * np.random.normal(0, np.sqrt(dt))
            x = x + drift * dt + noise
            x = max(x, 0.0)
            y.append(x)
        trajs.append(y)
    return t_eval, np.array(trajs)

# ----------------- Plotting -----------------
if __name__ == "__main__":
    os.makedirs("figs", exist_ok=True)

    t_eval, treat_trajs = simulate_pkpd(args_pkpd, treat_on=True)
    _, notreat_trajs = simulate_pkpd(args_pkpd, treat_on=False)
    print(treat_trajs.shape, notreat_trajs.shape)  # (n_samples, n_steps)

    fig, axes = plt.subplots(2, 1, figsize=(5, 4), sharex=True)
    for i in range(args_pkpd["n_samples"]):
        axes[0].plot(t_eval, treat_trajs[i], color="tab:red", alpha=0.5, lw=1)
        axes[1].plot(t_eval, notreat_trajs[i], color="tab:blue", alpha=0.5, lw=1)
    axes[0].set_title("PKPD Tumor Trajectories (Treated: $a=1$)")
    axes[1].set_title("PKPD Tumor Trajectories (Untreated: $a=0$)")
    axes[1].set_xlabel("Time")
    axes[0].set_ylabel("Tumor Volume $x(t)$")
    axes[1].set_ylabel("Tumor Volume $x(t)$")
    fig.tight_layout()
    plt.savefig("figs/pkpd_treat-notreat.pdf", dpi=150)
    plt.close(fig)
    print("Saved plot to figs/pkpd_treat-notreat.pdf")
