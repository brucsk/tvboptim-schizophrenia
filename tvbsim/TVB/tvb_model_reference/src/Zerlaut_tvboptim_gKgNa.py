from tvboptim.experimental.network_dynamics.dynamics import AbstractDynamics

import jax.numpy as jnp
import jax.scipy as jsp
from jax import debug
import jax

from typing import Optional, Tuple, Union

from tvboptim.experimental.network_dynamics.core import Bunch

# TODO: make this more functional with the use of different functions attributed instead of different classes

def Zerlaut_factory(order, *args, **kwargs):
    if order == 1:
        return Zerlaut_TVBoptim_first_order(*args, **kwargs)
    elif order == 2:
        return Zerlaut_TVBoptim_second_order(*args, **kwargs)

class Zerlaut_TVBoptim_first_order(AbstractDynamics):

    # INTERFACE FOR TVBOPTIM
    # Multi-coupling: instantaneous and delayed
    STATE_NAMES = ("E","I","W_e","W_i","noise")
    INITIAL_STATE = (0.,0.,100.,0.,0.)  # Small initial synaptic activity

    AUXILIARY_NAMES = []

    DEFAULT_PARAMS = Bunch(
        weight_noise = 1e-4,
        tau_OU = 5.0,
        
        external_input_ex_ex = 0.315*1e-3,#0.315*1e-3, # KHz
        external_input_ex_in = 0.000,
        external_input_in_ex = 0.315*1e-3,
        external_input_in_in = 0.000,
        
        E_Na_e=50.,
        E_Na_i=50.,
        E_K_e=-90.,
        E_K_i=-90.,
        g_K_e=8.214285714285714,
        g_Na_e=1.7857142857142865,
        g_K_i=8.214285714285714,
        g_Na_i=1.7857142857142865,

        C_m = 200.0,  # membrane capacitance [pF]
        
        b_e = 5.0,  # excitatory adaptation current increment [pA]
        a_e = 0.0,  # excitatory adaptation conductance [nS]
        b_i = 0.0,  # inhibitory adaptation current increment [pA]
        a_i = 0.0,  # inhibitory adaptation conductance [nS]
        
        tau_w_e = 500.0,  # adaptation time constant of excitatory neurons [ms]
        tau_w_i = 1.0,  # adaptation time constant of inhibitory neurons [ms]
        
        E_e = 0.0,  # excitatory reversal potential [mV]
        E_i = -80.0,  # inhibitory reversal potential [mV]
        
        Q_e = 1.5,  # excitatory quantal conductance [nS]
        Q_i = 5.0,  # inhibitory quantal conductance [nS]
        
        tau_e = 5.0,  # excitatory decay [ms]
        tau_i = 5.0,  # inhibitory decay [ms]
        
        N_tot = 10000,  # cell number
        p_connect_e = 0.05,  # connectivity probability (excitatory)
        p_connect_i = 0.05,  # connectivity probability (inhibitory)
        g = 0.2,  # fraction of inhibitory cells
        
        K_ext_e = 400,  # number of excitatory connections from external population
        K_ext_i = 0,  # number of inhibitory connections from external population
        
        T = 20.0,  # time scale of describing network activity [ms]
        
        delta_mu_V_e = 0.,
        delta_mu_V_i = 0.,
        
        P_e = jnp.array([-0.04983106, 0.00506355, -0.02347012, 0.00229515, -0.00041053, 0.00743749, 0.00126506, 0.01054705, -0.04072161, -0.03659253]),
        P_i = jnp.array([-0.05149122, 0.00400369, -0.00835201, 0.00024142, -0.00050706, 0.00450271, 0.00284722, 0.00143454, -0.0153578, -0.01468669]),
            
        
        exps_e = jnp.array(
          [[0, 0, 0],
           [1, 0, 0],
           [0, 1, 0],
           [0, 0, 1],
           [2, 0, 0],
           [1, 1, 0],
           [1, 0, 1],
           [0, 2, 0],
           [0, 1, 1],
           [0, 0, 2]]),
        exps_i = jnp.array(
          [[0, 0, 0],
           [1, 0, 0],
           [0, 1, 0],
           [0, 0, 1],
           [2, 0, 0],
           [1, 1, 0],
           [1, 0, 1],
           [0, 2, 0],
           [0, 1, 1],
           [0, 0, 2]]),
    )

    COUPLING_INPUTS = {
        "instant": 1,
        "delayed": 1,
    }
    
    def __init__(self):
        super().__init__()

        
    def dynamics(
        self,
        t: float,
        state: jnp.ndarray,
        params: Bunch,
        coupling: Bunch,
        external: Bunch,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Compute AdEx mean-field dynamics with two coupling inputs.

        Parameters
        ----------
        t : float
            Current time (unused for autonomous system)
        state : jnp.ndarray
            Current state with shape ``[1, n_nodes]`` containing S (synaptic gating variable)
        params : Bunch
            Model parameters: a, b, d, gamma, tau_s, w, J_N, I_o
        coupling : Bunch
            Coupling inputs with attributes ``.instant[1, n_nodes]`` and ``.delayed[1, n_nodes]``
        external : Bunch
            External inputs (currently unused)

        Returns
        -------
        derivatives : jnp.ndarray
            State derivatives with shape ``[1, n_nodes]``
        """
        # Unpack parameters
        # TODO: question of optimization of parameters of functions making parameters changing through time
        E = state[0, :]
        I = state[1, :]
        W_e = state[2, :]
        W_i = state[3, :]
        noise = state[4, :]

        # fx_y: x input to population y
        fe_e, fi_e, fe_i, fi_i = self.inputs_merging(coupling, E, I, noise, params)

        fe_out, (mu_V_e, sig_V_e, tau_V_e) = self.TF_e(fe_e, fi_e, W_e, params)
        fi_out, (mu_V_i, sig_V_i, tau_V_i) = self.TF_i(fe_i, fi_i, W_i, params)

        # derivatives
        derivatives = jnp.array([
            (fe_out - E) / params.T,
            (fi_out - I) / params.T,
            -W_e/params.tau_w_e + params.b_e*E + params.a_e*(mu_V_e-(params.g_K_e*params.E_K_e + params.g_Na_e*params.E_Na_e)/(params.g_Na_e+params.g_K_e))/params.tau_w_e,
            -W_i/params.tau_w_i + params.b_i*I + params.a_i*(mu_V_i-(params.g_K_i*params.E_K_i + params.g_Na_i*params.E_Na_i)/(params.g_Na_i+params.g_K_i))/params.tau_w_i,
            -noise/params.tau_OU])

        return derivatives
    
    def TF_e(self, fe, fi, W, params):
        return self.TF(fe, fi, W, params, 
                       (params.g_K_e*params.E_K_e + params.g_Na_e*params.E_Na_e)/(params.g_Na_e+params.g_K_e), params.delta_mu_V_e, params.exps_e, params.P_e, params.g_K_e+params.g_Na_e)
    def TF_i(self, fe, fi, W, params):
        return self.TF(fe, fi, W, params,
                       (params.g_K_i*params.E_K_i + params.g_Na_i*params.E_Na_i)/(params.g_Na_i+params.g_K_i), params.delta_mu_V_i, params.exps_i, params.P_i, params.g_K_i+params.g_Na_i)
    
    def TF(self, fe, fi, W, params, E_L, delta_mu_V, exps, P, g_L):
        mu_V,sig_V,tau_V = self.get_fluct_regime_vars(
            fe,
            fi,
            W, 
            params,
            E_L,
            delta_mu_V,
            g_L
        )
        V_thre = self.threshold_func(mu_V, sig_V, tau_V*g_L/params.C_m, exps, P) * 1e3
        f_out = self.estimate_firing_rate(mu_V, sig_V, tau_V, V_thre)
        return jnp.squeeze(f_out), (mu_V, sig_V, tau_V)
        
    def inputs_merging(self, coupling, E, I, noise, params):
        """
            Returns a tuple of the total excitatory and inhibitory inputs to both populations in this order:
            exc to exc, inh to exc, exc to inh, inh to inh
        """
        # external exc input
        fe_ext = coupling.delayed[0, :] + params.weight_noise * noise
        fe_ext_e = (fe_ext + params.external_input_ex_ex) * params.K_ext_e
        fe_ext_e = jnp.clip(fe_ext_e, 1e-12, jnp.inf)
        fe_ext_i = (fe_ext + params.external_input_in_ex) * params.K_ext_e
        fe_ext_i = jnp.clip(fe_ext_i, 1e-12, jnp.inf)
        # external inh input (usually set to 0 via K_ext_i = 0)
        fi_ext = coupling.delayed[1, :]
        fi_ext_e = (fi_ext + params.external_input_ex_in) * params.K_ext_i
        fi_ext_i = (fi_ext + params.external_input_in_in) * params.K_ext_i
        
        # local exc input
        fe_local = (E+1.0e-6)*(1.-params.g)*params.p_connect_e*params.N_tot
        # local inh input
        fi_local = (I+1.0e-6)*params.g*params.p_connect_i*params.N_tot
        return fe_local+fe_ext_e, fi_local+fi_ext_e, fe_local+fe_ext_i, fi_local+fi_ext_i
        
    def get_fluct_regime_vars(self, fe, fi, W, params, E_L, delta_mu_V, g_L):
        """
        Compute the mean characteristic of neurons.
        Inspired from the next repository :
        https://github.com/yzerlaut/notebook_papers/tree/master/modeling_mesoscopic_dynamics
        :param fe: excitatory input
        :param fi: inhibitory input
        :param W: adaptation
        :param params: parameters of the mean-field
        :return: mean and variance of membrane voltage of neurons and autocorrelation time constant
        """
        # conductance fluctuation and effective membrane time constant
        mu_Ge, mu_Gi = params.Q_e*params.tau_e*fe, params.Q_i*params.tau_i*fi  # Eqns 5 from [MV_2018]
        mu_G = g_L+mu_Ge+mu_Gi  # Eqns 6 from [MV_2018]
        mu_G = jnp.maximum(mu_G, 1e-12)
        T_m = params.C_m/mu_G # Eqns 6 from [MV_2018]

        # membrane potential
        mu_V = (mu_Ge*params.E_e+mu_Gi*params.E_i+g_L*E_L-W)/mu_G  # Eqns 7 from [MV_2018]
        mu_V += delta_mu_V
        # post-synaptic membrane potential event s around muV
        U_e, U_i = params.Q_e/mu_G*(params.E_e-mu_V), params.Q_i/mu_G*(params.E_i-mu_V)
        # Standard deviation of the fluctuations
        # Eqns 8 from [MV_2018]
        var = fe*(U_e*params.tau_e)**2/(2.*(params.tau_e+T_m))+fi*(U_i*params.tau_i)**2/(2.*(params.tau_i+T_m))
        sigma_V = jnp.sqrt(var)
        sigma_V = jnp.sqrt(jnp.maximum(var, 1e-12))
        # Autocorrelation-time of the fluctuations Eqns 9 from [MV_2018]
        T_V_numerator = (fe*(U_e*params.tau_e)**2 + fi*(U_i*params.tau_i)**2)
        T_V_denominator = (fe*(U_e*params.tau_e)**2/(params.tau_e+T_m) + fi*(U_i*params.tau_i)**2/(params.tau_i+T_m))
        
        T_V = T_V_numerator/T_V_denominator
        T_V = jnp.maximum(1e-12, T_V)
        return mu_V, sigma_V, T_V
    
    def threshold_func(self, muV, sigmaV, TvN, exps, P):
        muV0, DmuV0 = -60.0, 10.0
        sV0, DsV0 = 4.0, 6.0
        TvN0, DTvN0 = 0.5, 1.
        # epsilon added if values too close to 0 because 0**0 is not well defined in differentiation
        eps = 1e-12
        V = ((muV-muV0)/DmuV0).reshape(-1,1)
        V = jnp.where(jnp.abs(V) < eps, V + eps, V)
        S = ((sigmaV-sV0)/DsV0).reshape(-1,1)
        S = jnp.where(jnp.abs(S) < eps, S + eps, S)
        T = ((TvN-TvN0)/DTvN0).reshape(-1,1)
        T = jnp.where(jnp.abs(T) < eps, T + eps, T)
        feats = (
            V**exps[:, 0] *
            S**exps[:, 1] *
            T**exps[:, 2]) 
        return feats @ P
    
    def estimate_firing_rate(self, muV, sigmaV, Tv, Vthre):
        return jsp.special.erfc((Vthre-muV) / (jnp.sqrt(2)*sigmaV)) / (2*Tv)

class Zerlaut_TVBoptim_second_order(Zerlaut_TVBoptim_first_order):

    STATE_NAMES = ("E","I","C_ee","C_ei","C_ii","W_e","W_i","noise")
    INITIAL_STATE = (0.,0.,0.,0.,0.,100.,0.,0.)  # Small initial synaptic activity

    def __init__(self):
        super().__init__()
        self.dTF_e_dfe = jax.vmap(jax.grad(self.TF_e, argnums=0, has_aux=True), in_axes=(0,0,0,None))
        self.dTF_e_dfi = jax.vmap(jax.grad(self.TF_e, argnums=1, has_aux=True), in_axes=(0,0,0,None))
        self.dTF_i_dfe = jax.vmap(jax.grad(self.TF_i, argnums=0, has_aux=True), in_axes=(0,0,0,None))
        self.dTF_i_dfi = jax.vmap(jax.grad(self.TF_i, argnums=1, has_aux=True), in_axes=(0,0,0,None))
        
        self.d2TF_e = jax.vmap(jax.hessian(self.TF_e, argnums=(0,1), has_aux=True), in_axes=(0,0,0,None))
        self.d2TF_i = jax.vmap(jax.hessian(self.TF_i, argnums=(0,1), has_aux=True), in_axes=(0,0,0,None))

    def dynamics(
        self,
        t: float,
        state: jnp.ndarray,
        params: Bunch,
        coupling: Bunch,
        external: Bunch,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        
        N_e = params.N_tot * (1 - params.g)
        N_i = params.N_tot * params.g
        
        E = state[0, :]
        I = state[1, :]
        C_ee = state[2, :]
        C_ei = state[3, :]
        C_ii = state[4, :]
        W_e = state[5,:]
        W_i = state[6,:]
        noise = state[7,:]

        # fx_y: x input to population y
        fe_e, fi_e, fe_i, fi_i = self.inputs_merging(coupling, E, I, noise, params)     

        fe_out, (mu_V_e, sig_V_e, tau_V_e) = self.TF_e(fe_e, fi_e, W_e, params)
        fi_out, (mu_V_i, sig_V_i, tau_V_i) = self.TF_i(fe_i, fi_i, W_i, params)

        # TODO: could optimize by taking into account in hessian computation that we computed the gradients already
        d2TF_e_values = self.d2TF_e(fe_e, fi_e, W_e, params)[0]
        d2TF_i_values = self.d2TF_i(fe_i, fi_i, W_i, params)[0]
        dTF_e_dfe_value = self.dTF_e_dfe(fe_e, fi_e, W_e, params)[0]
        dTF_e_dfi_value = self.dTF_e_dfi(fe_e, fi_e, W_e, params)[0]
        dTF_i_dfe_value = self.dTF_i_dfe(fe_i, fi_i, W_i, params)[0]
        dTF_i_dfi_value = self.dTF_i_dfi(fe_i, fi_i, W_i, params)[0]
        
        
        diff_e = fe_out - E
        diff_i = fi_out - I
        
        
        derivatives = jnp.array([
            (diff_e + .5*(C_ee*d2TF_e_values[0][0] + C_ei*(d2TF_e_values[0][1] + d2TF_e_values[1][0]) + C_ii*d2TF_e_values[1][1]))/params.T,
            (diff_i + .5*(C_ee*d2TF_i_values[0][0] + C_ei*(d2TF_i_values[0][1] + d2TF_i_values[1][0]) + C_ii*d2TF_i_values[1][1]))/params.T,
            (fe_out*(params.T**-1-fe_out)/N_e + diff_e**2 + 2.*(C_ee*dTF_e_dfe_value + C_ei*dTF_i_dfi_value - C_ee))/params.T,
            (diff_e*diff_i + C_ee*dTF_e_dfe_value + C_ei*(dTF_i_dfe_value + dTF_e_dfi_value) + C_ii*dTF_i_dfi_value - 2.*C_ei)/params.T,
            (fi_out*(params.T**-1-fi_out)/N_i + diff_i**2 + 2.*(C_ii*dTF_i_dfi_value + C_ei*dTF_e_dfe_value - C_ii))/params.T,
            -W_e/params.tau_w_e + params.b_e*E + params.a_e*(mu_V_e-params.g_K_e*params.E_K_e + params.g_Na_e*params.E_Na_e)/params.tau_w_e,
            -W_i/params.tau_w_i + params.b_i*I + params.a_i*(mu_V_i-params.g_K_i*params.E_K_i + params.g_Na_i*params.E_Na_i)/params.tau_w_i,
            -noise/params.tau_OU 
            ])
        return derivatives
    
    
    
    
    
    