
import numpy as np
import jax
import os
import time

cpu = True
if cpu:
    N = 32
    os.environ['XLA_FLAGS'] = f'--xla_force_host_platform_device_count={N}'

# Import all required libraries
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import pandas as pd 
import jax.numpy as jnp
import copy
import optax
import scipy.io as sio
from scipy.io import loadmat
import equinox as eqx
from typing import Tuple
from itertools import product
import joblib
from joblib import Parallel, delayed
from contextlib import contextmanager
from tqdm.auto import tqdm
from pathlib import Path

# Jax enable x64
jax.config.update("jax_enable_x64", True)

# Import from tvboptim
from tvboptim.types import Parameter, BoundedParameter
from tvboptim.types.stateutils import show_parameters
from tvboptim.utils import set_cache_path, cache
from tvboptim.optim.optax import OptaxOptimizer
from tvboptim.optim.callbacks import MultiCallback, DefaultPrintCallback, SavingLossCallback

# Network dynamics imports
from tvboptim.experimental.network_dynamics import Network, solve, prepare
from tvboptim.experimental.network_dynamics.dynamics.tvb import ReducedWongWang
from tvboptim.experimental.network_dynamics.coupling.base import InstantaneousCoupling
from tvboptim.experimental.network_dynamics.coupling import LinearCoupling, FastLinearCoupling
from tvboptim.experimental.network_dynamics.graph import DenseDelayGraph, DenseGraph
from tvboptim.experimental.network_dynamics.solvers import Heun, BoundedSolver
from tvboptim.experimental.network_dynamics.noise import AdditiveNoise
from tvboptim.data import load_structural_connectivity, load_functional_connectivity
from tvboptim.experimental.network_dynamics.dynamics.base import AbstractDynamics
from tvboptim.experimental.network_dynamics.core.bunch import Bunch

# BOLD monitoring
from tvboptim.observations.tvb_monitors.bold import Bold

# Observation functions
from tvboptim.observations.observation import compute_fc, fc_corr, rmse

# Caching utilities
from tvboptim.utils import set_cache_path, cache

from utils import setup_directories
# Set cache path for tvboptim
set_cache_path("ei_tuning")

# Import later built utility functions
from utils import z_score_per_region, lagged_fc_matrices



def run_optimization_multisubject(learning_rate = 0.0325, max_steps = 120):
    # Set up directories
    results_dir, path_simulated_bold = setup_directories()
    
    
    # Create optimizer
    optimizer_test = OptaxOptimizer(
        loss,
        optax.adamaxw(learning_rate=learning_rate),
        callback=MultiCallback([DefaultPrintCallback(), SavingLossCallback()])
    )

    # Run optimization
    optimized_state, optimized_fit = optimizer_test.run(state_opt, max_steps=max_steps)

    return optimized_state, optimized_fit