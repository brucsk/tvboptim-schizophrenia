"""
extract_ei_arrays.py

Run this in the SAME environment/Python version that produced
part2_saved_state_*.pkl (i.e. wherever optim_multisubject_kluster.py ran).

The original pickle embeds full JAX/Equinox model objects, which contain
compiled Python code objects -- these are only safely unpicklable with a
matching Python version (mismatches raise errors like
"code expected at most 16 arguments, got 18"). This script unpickles the
saved state ONCE, in the environment that can still read it, and re-saves
just the plain numpy arrays needed downstream (wLRE, wFFI, J_i) into a
portable .npz file with no code objects, safe to load from any Python
version or machine afterwards.
"""

import glob
import os
import sys

import cloudpickle as pickle
import numpy as np

## Single configuration point ====================
pickle_path = None  # set explicitly, or auto-detect most recent match below
run_dir_glob = "./results/TVBOptim_RWW/kernel_test/**/part2_saved_state_*.pkl"

if pickle_path is None:
    candidates = sorted(glob.glob(run_dir_glob, recursive=True), key=os.path.getmtime)
    if not candidates:
        raise FileNotFoundError(f"No pickle found matching {run_dir_glob}. Set pickle_path explicitly.")
    pickle_path = candidates[-1]

print(f"Python version in this environment: {sys.version}")
print(f"Loading {pickle_path}")
with open(pickle_path, "rb") as f:
    saved = pickle.load(f)

optimized_states = saved["optimized_states"]  # (n_sub, n_cond) object array
n_sub, n_cond = optimized_states.shape
print(f"Found optimized_states with shape ({n_sub}, {n_cond})")

wLRE_shape = np.asarray(optimized_states[0, 0].coupling.coupling.wLRE).shape
J_i_shape = np.asarray(optimized_states[0, 0].dynamics.J_i).shape

wLRE_all = np.empty((n_sub, n_cond) + wLRE_shape)
wFFI_all = np.empty((n_sub, n_cond) + wLRE_shape)
J_i_all = np.empty((n_sub, n_cond) + J_i_shape)

for s in range(n_sub):
    for c in range(n_cond):
        state = optimized_states[s, c]
        wLRE_all[s, c] = np.asarray(state.coupling.coupling.wLRE)
        wFFI_all[s, c] = np.asarray(state.coupling.coupling.wFFI)
        J_i_all[s, c] = np.asarray(state.dynamics.J_i)

out_path = os.path.join(os.path.dirname(pickle_path), "ei_fitted_arrays.npz")
np.savez(out_path, wLRE=wLRE_all, wFFI=wFFI_all, J_i=J_i_all)
print(f"Saved portable arrays to {out_path}")
print("This .npz can now be loaded from any Python version/environment with np.load().")
