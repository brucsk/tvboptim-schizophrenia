"""
check_ei_identifiability.py

Purpose
-------
Before comparing fitted wLRE_ij, wFFI_ij (and their E/I ratio) between groups,
check whether a single gradient-descent fit per subject is even a reliable
estimate of that subject's parameters.

For a handful of subjects per group, refit multiple times with different
random seeds / initial conditions against the *same* empirical target FC,
then compare:

    within-subject variance   (spread across seeds, same subject)
  vs
    between-subject variance  (spread across subjects, same group, using
                                the seed-averaged estimate per subject)

If within-subject noise is comparable to or larger than between-subject
spread, single-seed fits are not trustworthy for group comparison, and the
seed-averaged ("stabilized") parameters saved at the end of this script
should be used instead (see compare_ei_params downstream).

This mirrors the robustness check in Schirner et al. (2023), who found
CV_wLRE ~ 0.5 and CV_wFFI ~ 0.72 across refits of the same target FC.

Follows the structure/conventions of optim_multisubject_kluster.py and
utils.py in this codebase.
"""

## Imports ====================
import copy
import os
from pathlib import Path

import cloudpickle as pickle
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import pandas as pd

from tvboptim.observations.tvb_monitors.bold import HRFBold, BalloonWindkesselBold, HRFKernel, FirstOrderVolterraHRFKernel, GammaHRFKernel, DoubleExponentialHRFKernel, MixtureOfGammasHRFKernel

from utils import (
    load_and_organize_bold,
    load_structural_connectivity,
    lagged_fc_matrices,
    make_loss,
)
from model_definition.network_model_utils import (
    build_network_model,
    run_initial_simulation,
    setup_bold_monitor,
)
from tvboptim.types import Parameter, BoundedParameter
from tvboptim.optim.optax import OptaxOptimizer
from tvboptim.observations.tvb_monitors.bold import HRFBold, MixtureOfGammasHRFKernel

jax.config.update("jax_enable_x64", True)

## Single configuration point ====================
data_dir = "./"
cond0_filename = "TS_Control.npy"
cond1_filename = "TS_Schizo.npy"
sc_path = "SC_EnigmadK68.mat"
tl_path = "tract_lengths.csv"
centers_path = "centers.txt"

conds = ["CTR", "SCZ"]
n_cond = len(conds)
n_nodes = 68
n_sub_total = 48

N_SUBJECTS_CHECK = 48           # subjects per group used for this check (5-10)
SEED_LIST = list(range(5))     # random seeds per subject
INIT_JITTER_SD = 0.05          # relative sd of multiplicative jitter on initial wLRE/wFFI

# Simulation / fitting settings -- kept identical to the production pipeline
# so this check is representative of what group-level fits will look like.
# NOTE: N_SUBJECTS_CHECK * n_cond * len(SEED_LIST) fits are run below
# (here: 48*2*5 = 480), each for max_steps optimizer steps, so runtime is
# larger than the full 48-subject pipeline. Reduce max_steps for a
# cheaper/quicker sanity check if needed.
t1 = 314_000
dt = 4.0
bold_TR = 2000.0
transient_lim = 5
target_fic = 0.25
sigma = 0.01
learning_rate = 0.0325
max_steps = 300
alpha_fc0 = 1.0
beta_fc1 = 2.0
n_tau = 2

result_dir = f"./results/ei_identifiability_check_nsub_{N_SUBJECTS_CHECK}/"
os.makedirs(result_dir, exist_ok=True)
checkpoint_dir = os.path.join(result_dir, "fits")
os.makedirs(checkpoint_dir, exist_ok=True)

EPS = 1e-6

## Load data ====================
print("Loading BOLD data and structural connectivity...")
new_array = load_and_organize_bold(
    data_dir=data_dir,
    cond0_filename=cond0_filename,
    cond1_filename=cond1_filename,
    n_sub=n_sub_total,
    n_nodes=n_nodes,
)

weights, delays, labels = load_structural_connectivity(
    sc_filepath=sc_path, tl_filepath=tl_path, centers_filepath=centers_path
)
# Mask of structurally connected edges: wLRE_ij/wFFI_ij only receive a
# nonzero gradient where C_ij > 0. Edges with C_ij = 0 stay at whatever
# value they were (jitter-)initialized to and their "variance across seeds"
# is pure noise, not identifiability -- excluding them from the stats below.
sc_mask = np.asarray((weights > 0) & ~np.eye(n_nodes, dtype=bool))
print(f"Structurally connected edges used in stats: {sc_mask.sum()} / {n_nodes**2}")

# Empirical target FC (lag-0, lag-1) per subject/condition
Q0_emp_all = np.zeros((N_SUBJECTS_CHECK, n_nodes, n_nodes, n_cond))
Q1_emp_all = np.zeros((N_SUBJECTS_CHECK, n_nodes, n_nodes, n_cond))
for subj_idx in range(N_SUBJECTS_CHECK):
    for cond_idx in range(n_cond):
        X_emp = new_array[subj_idx, :, :, cond_idx].T
        Q_emp = lagged_fc_matrices(X_emp, n_tau=n_tau, diag_zero=False, diag_zero_Q0=False, z_score=True)
        Q0_emp_all[subj_idx, :, :, cond_idx] = Q_emp[0]
        Q1_emp_all[subj_idx, :, :, cond_idx] = Q_emp[1]

## Build shared network + baseline state ====================
print("Building network and running initial transient...")
network = build_network_model(weights=weights, labels=labels, sigma=sigma)
_, _, result_init = run_initial_simulation(t1=t1, dt=dt, network=network)

kernel = None  # No kernel needed for BalloonWindkesselBold
bold_monitor_opt = setup_bold_monitor(
    bold_TR=bold_TR, result_init=result_init, monitor_type=BalloonWindkesselBold, kernel=kernel
)
network.update_history(result_init)
model_opt, state_opt, _ = run_initial_simulation(t1=t1, dt=dt, network=network, verbose=False)

# Freeze a clean baseline state (post-transient, pre-fitting) that every
# single seeded fit below will independently start from -- deliberately not
# reused/mutated across iterations (see module docstring).
base_J_i = copy.deepcopy(state_opt.dynamics.J_i)
base_wLRE = copy.deepcopy(state_opt.coupling.coupling.wLRE)
base_wFFI = copy.deepcopy(state_opt.coupling.coupling.wFFI)


def run_seeded_fit(Q0_emp, Q1_emp, seed, checkpoint_path):
    """Fit wLRE, wFFI, J_i from an independent baseline state with a given seed.

    Varies both the noise realization used during simulation and the
    initial condition of wLRE/wFFI (multiplicative jitter), matching the
    "random initial conditions and noise generator seeds" description in
    Schirner et al. (2023).
    """
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, "rb") as f:
            return pickle.load(f)

    key = jax.random.key(seed)
    key_noise, key_jitter = jax.random.split(key)

    state = copy.deepcopy(state_opt)
    state.dynamics.J_i = Parameter(copy.deepcopy(base_J_i))

    jitter_lre = 1.0 + INIT_JITTER_SD * jax.random.normal(key_jitter, shape=base_wLRE.shape)
    key_jitter, subkey = jax.random.split(key_jitter)
    jitter_ffi = 1.0 + INIT_JITTER_SD * jax.random.normal(subkey, shape=base_wFFI.shape)

    init_wLRE = jnp.clip(base_wLRE * jitter_lre, 0.0, None)
    init_wFFI = jnp.clip(base_wFFI * jitter_ffi, 0.0, None)
    state.coupling.coupling.wLRE = BoundedParameter(init_wLRE, low=0.0, high=jnp.inf)
    state.coupling.coupling.wFFI = BoundedParameter(init_wFFI, low=0.0, high=jnp.inf)

    if hasattr(state, "noise") and hasattr(state.noise, "key"):
        state.noise.key = key_noise

    loss = make_loss(
        model_opt=model_opt,
        bold_monitor_opt=bold_monitor_opt,
        Q0_emp=Q0_emp,
        Q1_emp=Q1_emp,
        target_fic=target_fic,
        alpha_fc0=alpha_fc0,
        beta_fc1=beta_fc1,
    )

    optimizer = OptaxOptimizer(loss, optax.adamaxw(learning_rate=learning_rate))
    optimized_state, _ = optimizer.run(state, max_steps=max_steps)

    fit_result = {
        "wLRE": np.asarray(optimized_state.coupling.coupling.wLRE),
        "wFFI": np.asarray(optimized_state.coupling.coupling.wFFI),
        "J_i": np.asarray(optimized_state.dynamics.J_i),
    }
    with open(checkpoint_path, "wb") as f:
        pickle.dump(fit_result, f)
    return fit_result


## Run all seeded fits ====================
n_seeds = len(SEED_LIST)
wLRE_all = np.empty((n_cond, N_SUBJECTS_CHECK, n_seeds, n_nodes, n_nodes))
wFFI_all = np.empty((n_cond, N_SUBJECTS_CHECK, n_seeds, n_nodes, n_nodes))
J_i_all = np.empty((n_cond, N_SUBJECTS_CHECK, n_seeds, n_nodes))

for cond_idx in range(n_cond):
    for subj_idx in range(N_SUBJECTS_CHECK):
        for s, seed in enumerate(SEED_LIST):
            print(f"Fitting cond={conds[cond_idx]} subj={subj_idx} seed={seed}")
            ckpt = os.path.join(checkpoint_dir, f"fit_cond{cond_idx}_subj{subj_idx}_seed{seed}.pkl")
            fit = run_seeded_fit(
                Q0_emp=Q0_emp_all[subj_idx, :, :, cond_idx],
                Q1_emp=Q1_emp_all[subj_idx, :, :, cond_idx],
                seed=seed,
                checkpoint_path=ckpt,
            )
            wLRE_all[cond_idx, subj_idx, s] = fit["wLRE"]
            wFFI_all[cond_idx, subj_idx, s] = fit["wFFI"]
            J_i_all[cond_idx, subj_idx, s] = fit["J_i"]

# Normalized E/I index, bounded in [-1, 1] -- avoids blow-ups when wFFI -> 0
ratio_all = (wLRE_all - wFFI_all) / (wLRE_all + wFFI_all + EPS)

## Within-subject variability (across seeds) ====================
def edge_cv(array_seeds_last3, mask):
    """array_seeds_last3: (..., n_seeds, n_nodes, n_nodes). Returns CV per (..., node, node)."""
    mean = array_seeds_last3.mean(axis=-3)
    std = array_seeds_last3.std(axis=-3, ddof=1)
    cv = std / (np.abs(mean) + EPS)
    return mean, std, cv


wLRE_mean_within, wLRE_std_within, wLRE_cv_within = edge_cv(wLRE_all, sc_mask)
wFFI_mean_within, wFFI_std_within, wFFI_cv_within = edge_cv(wFFI_all, sc_mask)
ratio_mean_within, ratio_std_within, ratio_cv_within = edge_cv(ratio_all, sc_mask)

# wLRE_mean_within / wFFI_mean_within / ratio_mean_within are the seed-averaged
# ("stabilized") per-subject parameter estimates: shape (n_cond, N_SUBJECTS_CHECK, n_nodes, n_nodes)

def masked_median_per_subject(cv_array, mask):
    """cv_array: (n_cond, n_subj, n_nodes, n_nodes) -> (n_cond, n_subj) median over masked edges."""
    n_c, n_s = cv_array.shape[:2]
    out = np.empty((n_c, n_s))
    for c in range(n_c):
        for s in range(n_s):
            out[c, s] = np.median(cv_array[c, s][mask])
    return out


subj_within_cv_wLRE = masked_median_per_subject(wLRE_cv_within, sc_mask)
subj_within_cv_wFFI = masked_median_per_subject(wFFI_cv_within, sc_mask)
subj_within_cv_ratio = masked_median_per_subject(np.abs(ratio_std_within) / (np.abs(ratio_mean_within) + EPS), sc_mask)

## Between-subject variability (within group, using stabilized estimates) ====================
def between_subject_cv(mean_within, mask):
    """mean_within: (n_cond, n_subj, n_nodes, n_nodes) -> (n_cond,) median CV across subjects, masked edges."""
    n_c = mean_within.shape[0]
    out = np.empty(n_c)
    for c in range(n_c):
        subj_mean = mean_within[c].mean(axis=0)
        subj_std = mean_within[c].std(axis=0, ddof=1)
        cv = subj_std / (np.abs(subj_mean) + EPS)
        out[c] = np.median(cv[mask])
    return out


between_cv_wLRE = between_subject_cv(wLRE_mean_within, sc_mask)
between_cv_wFFI = between_subject_cv(wFFI_mean_within, sc_mask)
between_cv_ratio = between_subject_cv(ratio_mean_within, sc_mask)

## Summary ====================
summary_rows = []
for c in range(n_cond):
    summary_rows.append({
        "condition": conds[c],
        "param": "wLRE",
        "within_subject_CV_median": np.median(subj_within_cv_wLRE[c]),
        "between_subject_CV_median": between_cv_wLRE[c],
        "within_over_between": np.median(subj_within_cv_wLRE[c]) / (between_cv_wLRE[c] + EPS),
    })
    summary_rows.append({
        "condition": conds[c],
        "param": "wFFI",
        "within_subject_CV_median": np.median(subj_within_cv_wFFI[c]),
        "between_subject_CV_median": between_cv_wFFI[c],
        "within_over_between": np.median(subj_within_cv_wFFI[c]) / (between_cv_wFFI[c] + EPS),
    })
    summary_rows.append({
        "condition": conds[c],
        "param": "ratio",
        "within_subject_CV_median": np.median(subj_within_cv_ratio[c]),
        "between_subject_CV_median": between_cv_ratio[c],
        "within_over_between": np.median(subj_within_cv_ratio[c]) / (between_cv_ratio[c] + EPS),
    })

summary_df = pd.DataFrame(summary_rows)
summary_path = os.path.join(result_dir, "identifiability_summary.csv")
summary_df.to_csv(summary_path, index=False)
print("\n" + summary_df.to_string(index=False))

print(
    "\nInterpretation: within_over_between > ~1 means seed-to-seed "
    "(optimizer) noise is as large as, or larger than, subject-to-subject "
    "spread within a group -- single-seed fits should not be used for group "
    "comparison; use the seed-averaged ('stabilized') parameters saved below "
    "instead. Reference: Schirner et al. (2023) report CV_wLRE ~ 0.5 and "
    "CV_wFFI ~ 0.72 across refits of the same target."
)

## Diagnostic plot ====================
fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
labels_cv = [("wLRE", subj_within_cv_wLRE), ("wFFI", subj_within_cv_wFFI), ("ratio", subj_within_cv_ratio)]
colors = [plt.cm.cividis(0.3), plt.cm.cividis(0.75)]

for ax, (name, data) in zip(axes, labels_cv):
    positions = []
    all_vals = []
    for c in range(n_cond):
        pos = c
        positions.append(pos)
        all_vals.append(data[c])
    parts = ax.violinplot(all_vals, positions=positions, showmedians=True)
    for i, body in enumerate(parts["bodies"]):
        body.set_facecolor(colors[i % 2])
        body.set_alpha(0.7)
    for c in range(n_cond):
        jitter_x = pos + np.random.uniform(-0.05, 0.05, size=len(data[c]))
        ax.scatter(np.full(len(data[c]), c) + np.random.uniform(-0.05, 0.05, size=len(data[c])),
                   data[c], color="k", s=15, alpha=0.6, zorder=3)
    ax.set_xticks(range(n_cond))
    ax.set_xticklabels(conds)
    ax.set_title(f"Within-subject CV: {name}")
    ax.grid(True, alpha=0.3)

axes[0].set_ylabel("Median CV across seeds (per subject)")
plt.tight_layout()
plt.savefig(os.path.join(result_dir, "within_subject_cv.png"), dpi=300)
plt.close(fig)

## Save stabilized (seed-averaged) parameters for downstream group comparison ====================
np.save(os.path.join(result_dir, "wLRE_stable.npy"), wLRE_mean_within)   # (n_cond, n_subj, n_nodes, n_nodes)
np.save(os.path.join(result_dir, "wFFI_stable.npy"), wFFI_mean_within)
np.save(os.path.join(result_dir, "ratio_stable.npy"), ratio_mean_within)
np.save(os.path.join(result_dir, "J_i_stable.npy"), J_i_all.mean(axis=2))  # (n_cond, n_subj, n_nodes)
np.save(os.path.join(result_dir, "sc_mask.npy"), sc_mask)

print(f"\nStabilized (seed-averaged) parameters and summary saved to {result_dir}")
