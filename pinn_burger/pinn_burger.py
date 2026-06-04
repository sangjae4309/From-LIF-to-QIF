
from functools import partial
from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
import optax
import torch
from jax import jit, random, value_and_grad, vmap
from jaxtyping import Array, ArrayLike, Float, Int, UInt8
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets
from tqdm import trange as trange_script
from tqdm import trange as trange_scipt  # Works in .py scripts
import time
import scipy.io as sio
import numpy as np
import os
import pickle
import equinox as eqx

from spikegd.models import AbstractPhaseOscNeuron, AbstractPseudoPhaseOscNeuron
from spikegd.utils.plotting import formatter, petroff10

# %%
############################
### Data loading
############################

def normalize(x1,T):
  normalized_x1 = (x1 - jnp.min(x1))/(jnp.max(x1)-jnp.min(x1))
  normalized_x1=(1-normalized_x1)*T
  normalized_x1 = normalized_x1.reshape(-1,1)
  return normalized_x1


# no encoding
def generate_regression_data(num_samples: int, T):
    """Generate regression data with Gaussian receptive field encoding."""
    data = sio.loadmat('experiments/pinnst/burgers_shock.mat')
    x = data['x']
    x = x[:, 0] #100
    t = data['t']
    t = t[:, 0] #256


    n = 2 #downsample
    x_min = jnp.array([jnp.min(x)])
    x_max = jnp.array([jnp.max(x)])
    t_min = jnp.array([jnp.min(t)])
    t_max = jnp.array([jnp.max(t)])
    x_in = x[::n]
    t_in = t[::n]

    x = jnp.unique(jnp.concatenate([x_min, x_in, x_max]))
    t = jnp.unique(jnp.concatenate([t_min, t_in, t_max]))


    X1, X2 = jnp.meshgrid(x, t)
    X1 = X1.ravel()
    X2 = X2.ravel()
    print(X1.shape, X2.shape)
    X = jnp.column_stack((X1, X2))
    x1 = X[:, 0]
    x2 = X[:, 1]
    print(jnp.min(x1), jnp.max(x1), jnp.min(x2), jnp.max(x2))


    normalized_x1 = normalize(x1,T)
    normalized_x2 = normalize(x2,T)
    
   
    encoded_inputs = jnp.concat((normalized_x1, normalized_x2), axis=1)

    ys = data['usol']
    ys = ys[::n, ::n]
    ys = np.zeros(X1.shape)
    ys = ys.reshape(-1, 1)


    encoded_inputs = np.array(encoded_inputs)

    num = len(x1)
    # Generate random t values for boundaries (equivalent to 'y' in your torch code)
    key1, key2 = random.split(random.PRNGKey(42))
    t_vals_nn = np.linspace(0, 0.99, num)
   
    # Generate random x values for initial condition (equivalent to 'x' in your torch code) 
    x_vals_nn = np.linspace(-1, 1, num)

    # Boundary conditions - following torch.ones_like pattern
    x_bc1_nn = jnp.ones_like(t_vals_nn)     # x = +1 boundary (right)
    x_bc2_nn = -jnp.ones_like(t_vals_nn)    # x = -1 boundary (left)
    t_bc2_nn = jnp.zeros_like(x_vals_nn)    # t = 0 boundary (initial time)
    

    def normalize_global(x, xx, T):
        normalized_x = (x - jnp.min(xx))/(jnp.max(xx)-jnp.min(xx))
        normalized_x=(1-normalized_x)*T
        normalized_x = normalized_x.reshape(-1,1)
        return normalized_x


    x_bc1 = normalize_global(x_bc1_nn,x1,T)
    x_bc2 = normalize_global(x_bc2_nn,x1,T)
    t_bc2 = normalize_global(t_bc2_nn,x2,T)

  

    x_vals = normalize_global(x_vals_nn,x1,T)
    t_vals = normalize_global(t_vals_nn,x2,T)
   
   
    xt_right = jnp.stack([x_bc1, t_vals], axis=1).reshape(num, 2)
    xt_left = jnp.stack([x_bc2, t_vals], axis=1).reshape(num, 2)
    
    # Initial condition points: (t=0, x)
    xt_initial = jnp.stack([x_vals, t_bc2], axis=1).reshape(num, 2)
    xt_initial_label = jnp.stack([x_vals_nn, t_bc2_nn], axis=1).reshape(num, 2)
    xt_right = np.array(xt_right)
    xt_left = np.array(xt_left)
    xt_initial = np.array(xt_initial)


    # Create training output
    u_ic_label = -jnp.sin(jnp.pi * x_vals_nn) # u_ini = -sin(pi*x_ini)
    u_bc_right_label = np.zeros((num, 1)) 
    u_bc_left_label = np.zeros((num, 1)) 

    u_ic_label = np.array(u_ic_label).reshape(-1,1)



    return encoded_inputs, ys, xt_right, xt_left, xt_initial, u_bc_right_label, u_bc_left_label, u_ic_label



def load_data(data: callable, root: str, config: dict) -> tuple[DataLoader, DataLoader]:
    """
    Creates DataLoaders for regression data.
    
    This function generates synthetic regression data (learning y = x^2) instead of
    loading MNIST images. The generated data is wrapped in a TensorDataset to mimic
    the original data-loading format.
    
    Args:
        data: Unused here; kept for compatibility with the original signature.
        root: Unused for regression data.
        config: Dictionary containing configuration parameters.
            Expected keys include:
              - "Nbatch": Batch size.
              - "num_train": Number of training samples.
              - "num_test": Number of test samples.
    
    Returns:
        A tuple (train_loader, test_loader) of PyTorch DataLoaders.
    """
    Nbatch: int = config["Nbatch"]
    T: float = config["T"]

    # Training set: Generate regression data
    num_train = config.get("num_train", 1000)
    train_inputs, train_targets, xt_right, xt_left, xt_initial, u_bc_right_label, u_bc_left_label, u_ic_label = generate_regression_data(num_train, T)
    train_set = TensorDataset(torch.tensor(train_inputs), torch.tensor(train_targets), torch.tensor(xt_right), torch.tensor(xt_left), torch.tensor(xt_initial), torch.tensor(u_bc_right_label), torch.tensor(u_bc_left_label), torch.tensor(u_ic_label))
    train_loader = DataLoader(train_set, batch_size=Nbatch, shuffle=True)

    # Test set: Generate regression data
    num_test = config.get("num_test", 1000)
    test_inputs, test_targets, _, _, _, _,_,_ = generate_regression_data(num_test, T)
    test_set = TensorDataset(torch.tensor(test_inputs), torch.tensor(test_targets))
    test_loader = DataLoader(test_set, batch_size=100, shuffle=False)

    

    return train_loader, test_loader


# Save parameters (works with any JAX parameter structure)
def safe_save_params(params, filepath):
    """Save JAX parameters safely using pickle"""
    with open(filepath, 'wb') as f:
        pickle.dump(params, f)
    print(f"Parameters saved to {filepath}")

# Load parameters
def safe_load_params(filepath):
    """Load JAX parameters safely using pickle"""
    with open(filepath, 'rb') as f:
        params = pickle.load(f)
    print(f"Parameters loaded from {filepath}")
    return params



#
# %%
############################
### Initialization
############################


def init_weights(key: Array, config: dict) -> tuple[Array, list]:
    """
    Initializes input and network weights.
    """
    ### Unpack arguments
    Nin: int = config["Nin"]
    Nhidden: int = config["Nhidden"]
    Nlayer: int = config["Nlayer"]
    Nout: int = config["Nout"]
    w_scale: float = config["w_scale"]

    ### Initialize weights
    key, subkey = random.split(key)
    weights = []
    width = w_scale / jnp.sqrt(Nin)
    weights_in = random.uniform(subkey, (Nhidden, Nin), minval=-width, maxval=width)
    weights.append(weights_in)
    width = w_scale / jnp.sqrt(Nhidden)
    for _ in range(1, Nlayer - 1):
        key, subkey = random.split(key)
        weights_hidden = random.uniform(
            subkey, (Nhidden, Nhidden), minval=-width, maxval=width
        )
        weights.append(weights_hidden)
    key, subkey = random.split(key)
    weights_out = random.uniform(subkey, (Nout, Nhidden), minval=-width, maxval=width)
    weights.append(weights_out)

    return key, weights


def init_phi0(neuron: AbstractPhaseOscNeuron, config: dict) -> Array:
    """
    Initializes initial phase of neurons.
    """
    ### Unpack arguments
    Nhidden: int = config["Nhidden"]
    Nlayer: int = config["Nlayer"]
    Nout: int = config["Nout"]
    N = Nhidden * (Nlayer - 1) + Nout
    theta = neuron.Theta()

    ### Initialize initial phase
    phi0 = theta / 2 * jnp.ones(N)
    return phi0


# %%
############################
### Model
############################


def eventffwd(
    neuron: AbstractPhaseOscNeuron, p: list, input: Float[Array, " Nin"], config: dict
) -> tuple:
    """
    Simulates a feedforward network with time-to-first-spike input encoding.
    """
    ### Unpack arguments
    Nin_virtual: int = config["Nin_virtual"]
    Nhidden: int = config["Nhidden"]
    Nlayer: int = config["Nlayer"]  # currently has to be at least 2 (1hidden)
    Nout: int = config["Nout"]
    N = Nhidden * (Nlayer - 1) + Nout
    T: float = config["T"]
    weights: list = p[0]
    phi0: Array = p[1]
    x0 = phi0[jnp.newaxis]

   
    neurons_in = jnp.arange(Nin_virtual)
    # print(input.shape)
    times_in = input
    spikes_in = (times_in, neurons_in)

    ### Input weights
    weights_in = weights[0]
    weights_in_virtual = jnp.zeros((N, Nin_virtual))
    weights_in_virtual = weights_in_virtual.at[:Nhidden, :].set(
            weights_in
        )
  

    ### Network weights
    weights_net = jnp.zeros((N, N))
    for i in range(Nlayer - 2):
        slice_in = slice(i * Nhidden, (i + 1) * Nhidden)
        slice_out = slice((i + 1) * Nhidden, (i + 2) * Nhidden)
        weights_net = weights_net.at[slice_out, slice_in].set(weights[i + 1])
    weights_net = weights_net.at[N - Nout :, N - Nout - Nhidden : N - Nout].set(
        weights[-1]
    )

    # Run simulation
    out = neuron.event(x0, weights_net, weights_in_virtual, spikes_in, config)

    return out



def outfn(
    neuron: AbstractPseudoPhaseOscNeuron, out: tuple, p: list, config: dict
) -> Array:
    """
    Computes output spike times given simulation results.
    """
    ### Unpack arguments
    Nin: int = config["Nin"]
    Nhidden: int = config["Nhidden"]
    Nlayer: int = config["Nlayer"]
    Nout: int = config["Nout"]
    N = Nhidden * (Nlayer - 1) + Nout
    weights = p[0]
    times: Array = out[0]
    spike_in: Array = out[1]
    neurons: Array = out[2]
    x: Array = out[3]

    ### Run network as feedforward rate ANN
    Kord = jnp.sum(neurons >= 0)  # Number of ordinary spikes
    x_end = x[Kord]
    pseudo_rates = jnp.zeros(Nin)
    for i in range(Nlayer - 1):
        input = neuron.linear(pseudo_rates, weights[i])
        x_end_i = x_end[:, i * Nhidden : (i + 1) * Nhidden]
        pseudo_rates = neuron.construct_ratefn(x_end_i)(input)
    input = neuron.linear(pseudo_rates, weights[Nlayer - 1])

    ### Spike times for each learned neuron
    def compute_tout(i: ArrayLike) -> Array:
        ### Potential ordinary output spike times
        mask = (neurons == N - Nout + i) & (spike_in == False)  # noqa: E712
        Kout = jnp.sum(mask)  # Number of ordinary output spikes
        t_out_ord = times[jnp.argmax(mask)]

        ### Pseudospike time
        t_out_pseudo = neuron.t_pseudo(x_end[:, N - Nout + i], input[i], 1, config)

        ### Output spike time
        t_out = jnp.where(0 < Kout, t_out_ord, t_out_pseudo)

        return t_out

    t_outs = vmap(compute_tout)(jnp.arange(Nout))
    return t_outs



# burger shock 
# %   u_t = -u*(u)' + (0.01/pi)*u",
# % for x in [-1,1] and t in [0,1], subject to
# %   u = 0 at x = -1
# % and
# %   u = 0 at x = 1
# - mu = 0.01/pi, IC: -sin(pi*x), BC: u(−1, t) = u(1, t) = 0

def apply_boundary_conditions(neuron: AbstractPseudoPhaseOscNeuron, p: list, config: dict, input):
    """Apply zero boundary conditions: u(-1) = u(1) = 0"""

    # input[0] = (1-(x+1)/2)*T = T/2 - xT/2 -> x = 1-2input[0]/T
    # input[1] = (1-t/0.99)*T = T - (T/0.99)t -> t = (0.99/T)*(T-input[1])
    # du/dx = du/dinput[0] dinput[0]/dx = -(T/2) du/dinput[0]
    # d2u/dx2 = d/dx (-(T/2) du/dinput[0]) = (T^2/4) du/dinput[0]
    # du/dt = (d/dinput[1])(dinput[1]/dt) u = -T/0.99 du/dinput[1]
    # Raw network output
    T = config["T"]
    outs = eventffwd(neuron, p, input, config)
    t_outs = outfn(neuron, outs, p, config)
    pred_raw = t_outs[1] - t_outs[0]
   
    
    # Apply zero boundary conditions using the method: u(x) = (-1-x)(1-x) * N(x)
    # This ensures u(-1) = u(1) = 0 automatically
    # Initial condition function
    u_ic = -jnp.sin(jnp.pi * (1-2*input[0]/T))
    
    # Boundary multiplier (vanishes at x=-1,1)
    boundary_mult = (-1-(1-2*input[0]/T)) * (1-(1-2*input[0]/T))


    time_mult = jnp.exp(-3.0*((0.99/T)*(T-input[1]))) 

    pred = u_ic * time_mult + ((0.99/T)*(T-input[1])) * boundary_mult * pred_raw

    return pred

@eqx.filter_jit
def compute_derivatives(neuron: AbstractPseudoPhaseOscNeuron, p: list, config: dict, input):
    """Compute first and second derivatives using automatic differentiation"""

    # Define scalar functions for x and t derivatives (following your 1D pattern)
    def u_scalar_x(x_scalar):
        xt_vec = jnp.array([x_scalar, input[1]])  # Fix t, vary x
        result = apply_boundary_conditions(neuron, p, config, xt_vec)
        # result = forward(xt_vec)
        return result[0] if result.ndim > 0 else result  # Return scalar
    
    def u_scalar_t(t_scalar):
        xt_vec = jnp.array([input[0], t_scalar])  # Fix x, vary t
        result = apply_boundary_conditions(neuron, p, config, xt_vec)
        # result = forward(xt_vec)
        return result[0] if result.ndim > 0 else result  # Return scalar
   
    
    # First derivatives
    u_x = jax.grad(u_scalar_x)(input[0])
    u_t = jax.grad(u_scalar_t)(input[1])
    
    # Second derivative w.r.t. x
    u_xx = jax.grad(jax.grad(u_scalar_x))(input[0])
    
    return u_t, u_x, u_xx

def source_term(input):
    """Source term f(x) for the Poisson equation -d²u/dx² = f(x)"""
    return 0


def lossfn(neuron: AbstractPseudoPhaseOscNeuron,
    p:list, target: jnp.ndarray, config: dict, input_physics, input_right_bc, input_left_bc, input_ic, target_right_bc, target_left_bc, target_ic
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Computes the mean squared error (MSE) loss for regression and a simple accuracy metric.

    Args:
        t_out: Array of predicted output spike times of shape (Nout,). For regression, Nout=1.
        target: Scalar (or single-element array) representing the true value (e.g. x^2).
        config: Configuration dictionary. An optional key 'reg_threshold' can be provided,
                which defines a threshold for considering the prediction "accurate".

    Returns:
        A tuple (loss, correct) where:
            - loss: The squared error (MSE) between the prediction and target.
            - correct: A boolean flag (as a JAX array) indicating whether the absolute error 
              is below the given threshold.
    """


    """Compute physics-informed loss for the PDE: -d²u/dx² = f(x)"""
    
    # Get derivatives
    # input[0] = (1-(x+1)/2)*T = 1 - T/2 - xT/2 -> xT/2 = 1-T/2-input[0] -> x = 2/T - 1-2input[0]/T
    # input[1] = (1-t/0.99)*T = T - (T/0.99)t -> t = (0.99/T)*(T-input[1])
    # du/dx = du/dinput[0] dinput[0]/dx = -(T/2) du/dinput[0]
    # d2u/dx2 = d/dx (-(T/2) du/dinput[0]) = (T^2/4) d2u/dinput[0]2
    # du/dt = (d/dinput[1])(dinput[1]/dt) u = -T/0.99 du/dinput[1]
    T = 2.0
    u_t, u_x, u_xx = compute_derivatives(neuron, p, config, input_physics)
    u_x = -(T/2)* u_x
    u_xx = ((T**2)/4)*u_xx


    # go from spike time to original x and t
    u_t = (-T/0.99)*u_t


    u = apply_boundary_conditions(neuron, p, config, input_physics)
    u = u[0] if u.ndim > 0 else u  # Ensure scalar
   
    # PDE residual: u_t + u*u_x - (0.01/pi)*u_xx = 0
    pde_residual = u_t + u * u_x - (0.01 / jnp.pi) * u_xx


    u_bc_right = apply_boundary_conditions(neuron, p, config, input_right_bc)
    u_bc_left = apply_boundary_conditions(neuron, p, config, input_left_bc)
    u_ic = apply_boundary_conditions(neuron, p, config, input_ic)


    u_bc_right = u_bc_right[0] if u_bc_right.ndim > 0 else u_bc_right  # Ensure scalar   
    u_bc_left = u_bc_left[0] if u_bc_left.ndim > 0 else u_bc_left  # Ensure scalar   
    u_ic = u_ic[0] if u_ic.ndim > 0 else u_ic  # Ensure scalar   

    # Mean squared error of PDE residual
    physics_loss_val = pde_residual**2
    bc_loss = (u_bc_right - target_right_bc)**2 + (u_bc_left - target_left_bc)**2 
    ic_loss = (u_ic - target_ic)**2
    loss = physics_loss_val
   

 


    # Define a threshold for acceptable error (default threshold 0.1)
    threshold = config.get("reg_threshold", 0.1)
    correct = jnp.abs(target - target) < threshold
    
    return loss, correct



# ordinary spike
def apply_boundary_conditions_ord(neuron: AbstractPseudoPhaseOscNeuron, p: list, config: dict, input):
    """Apply zero boundary conditions: u(0) = u(1) = 0"""
    T: float = config["T"]
    # Raw network output
    outs = eventffwd(neuron, p, input, config)
    t_outs = outfn(neuron, outs, p, config)
    t_outs_ord = jnp.where(t_outs < T, t_outs, T)
    pred_raw = t_outs_ord[1] - t_outs_ord[0]
    
    # Apply zero boundary conditions using the method: u(x) = x(1-x) * N(x)
    # This ensures u(0) = u(1) = 0 automatically
    pred = input * (2 - input) * pred_raw
    return pred


def compute_derivatives_ord(neuron: AbstractPseudoPhaseOscNeuron, p: list, config: dict, input):
    """Compute first and second derivatives using automatic differentiation"""
  
    # Define function that takes scalar input but reshapes for network
    def u_scalar(t_scalar):
        t_vec = jnp.array([t_scalar])  # Convert scalar to vector for network
        result = apply_boundary_conditions_ord(neuron, p, config, t_vec)
        return result[0]  # Return scalar
    
    # Now grad works with scalar input/output
    du_dt = jax.grad(u_scalar)(input[0])
    d2u_dt2 = jax.grad(jax.grad(u_scalar))(input[0])

    
    return du_dt, d2u_dt2



def lossfn_ord(neuron: AbstractPseudoPhaseOscNeuron,
    p:list, target: jnp.ndarray, config: dict, input_physics
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Computes the mean squared error (MSE) loss for regression and a simple accuracy metric.

    Args:
        t_out: Array of predicted output spike times of shape (Nout,). For regression, Nout=1.
        target: Scalar (or single-element array) representing the true value (e.g. x^2).
        config: Configuration dictionary. An optional key 'reg_threshold' can be provided,
                which defines a threshold for considering the prediction "accurate".

    Returns:
        A tuple (loss, correct) where:
            - loss: The squared error (MSE) between the prediction and target.
            - correct: A boolean flag (as a JAX array) indicating whether the absolute error 
              is below the given threshold.
    """


    """Compute physics-informed loss for the PDE: -d²u/dx² = f(x)"""
    
    # Get second derivatives
    _, d2u_dt2 = compute_derivatives_ord(neuron, p, config, input_physics)

    d2u_dx2 = 4*d2u_dt2

    # PDE residual: -d²u/dx² - f(x) = 0
    pde_residual = -d2u_dx2 - source_term(input_physics)
    
    # Mean squared error of PDE residual
    physics_loss_val = pde_residual**2

    loss = physics_loss_val

    # loss = ((t_out[1]-t_out[0]) - target) ** 2

 


    # Define a threshold for acceptable error (default threshold 0.1)
    threshold = config.get("reg_threshold", 0.1)
    correct = jnp.abs(target - target) < threshold
    
    return loss, correct




def simulatefn(
    neuron: AbstractPseudoPhaseOscNeuron,
    p: list,
    input: Float[Array, "Batch Nin"],
    labels: Float[Array, " Batch"],
    config: dict,
    xt_right, xt_left, xt_initial, u_bc_right_label, u_bc_left_label, u_ic_label
) -> tuple[Array, Array]:
    """
    Simulates the network and computes the loss and accuracy for batched input.
    """

    
    
    outs = vmap(eventffwd, in_axes=(None, None, 0, None))(neuron, p, input, config)
    t_outs = vmap(outfn, in_axes=(None, 0, None, None))(neuron, outs, p, config)
    loss, correct = vmap(lossfn, in_axes=(None, None, 0, None, 0, 0, 0, 0, 0, 0, 0))(neuron, p, labels, config, input, xt_right, xt_left, xt_initial, u_bc_right_label, u_bc_left_label, u_ic_label)
    mean_loss = jnp.mean(loss)
    accuracy = jnp.mean(correct)
    return mean_loss, accuracy




def probefn(
    neuron: AbstractPseudoPhaseOscNeuron,
    p: list,
    input: Float[Array, "Batch Nin"],
    labels: Float[Array, " Batch"],
    config: dict,
    xt_right, xt_left, xt_initial, u_bc_right_label, u_bc_left_label, u_ic_label
) -> tuple:
    """
    Computes several metrics.
    """

    ### Unpack arguments
    T: float = config["T"]
    Nhidden: int = config["Nhidden"]
    Nlayer: int = config["Nlayer"]
    Nout: int = config["Nout"]
    N = Nhidden * (Nlayer - 1) + Nout
    Nbatch: int = config["Nbatch"]

    ### Batched functions
    @vmap
    def batch_eventffwd(input):
        return eventffwd(neuron, p, input, config)

    @vmap
    def batch_outfn(outs):
        return outfn(neuron, outs, p, config)

    @partial(vmap, in_axes=(None, None, 0, None, 0, 0, 0, 0, 0, 0, 0))
    def batch_lossfn(neuron, p, labels, config, input, xt_right, xt_left, xt_initial, u_bc_right_label, u_bc_left_label, u_ic_label):
            return lossfn(neuron, p, labels, config, input, xt_right, xt_left, xt_initial, u_bc_right_label, u_bc_left_label, u_ic_label)
    
    @partial(vmap, in_axes=(None, None, 0, 0))
    def batch_lossfn_ord(neuron, p, labels, input):
            return lossfn_ord(neuron, p, labels, config, input)

    ### Run network
    outs = batch_eventffwd(input)
    times: Array = outs[0]
    spike_in: Array = outs[1]
    neurons: Array = outs[2]
    t_outs = batch_outfn(outs)

    ### Loss and accuracy with pseudospikes
    loss, correct = batch_lossfn(neuron, p, labels, config, input, xt_right, xt_left, xt_initial, u_bc_right_label, u_bc_left_label, u_ic_label)
    mean_loss = jnp.mean(loss)
    acc = jnp.mean(correct)

    ### Loss and accuracy without pseudospikes
    t_out_ord = jnp.where(t_outs < T, t_outs, T)
    loss_ord, correct_ord = batch_lossfn_ord(neuron, p, labels, input)
    mean_loss_ord = jnp.mean(loss_ord)
    acc_ord = jnp.mean(correct_ord)

    ### Activity and silent neurons
    mask = (spike_in == False) & (neurons < N - Nout) & (neurons >= 0)  # noqa: E712
    activity = jnp.sum(mask) / (Nbatch * (N - Nout))
    silent_neurons = jnp.isin(
        jnp.arange(N - Nout), jnp.where(mask, neurons, -1), invert=True
    )

    ### Activity and silent neurons until first output spike
    t_out_first = jnp.min(t_out_ord, axis=1)
    mask = (
        (spike_in == False)  # noqa: E712
        & (neurons < N - Nout)
        & (neurons >= 0)
        & (times < t_out_first[:, jnp.newaxis])
    )
    activity_first = jnp.sum(mask) / (Nbatch * (N - Nout))
    silent_neurons_first = jnp.isin(
        jnp.arange(N - Nout), jnp.where(mask, neurons, -1), invert=True
    )

    ### Pack results in dictionary
    metrics = {
        "loss": mean_loss,
        "acc": acc,
        "loss_ord": mean_loss_ord,
        "acc_ord": acc_ord,
        "activity": activity,
        "activity_first": activity_first,
    }
    silents = {
        "silent_neurons": silent_neurons,
        "silent_neurons_first": silent_neurons_first,
    }

    return metrics, silents


# %%
############################
### Training
############################


def run(
    neuron: AbstractPseudoPhaseOscNeuron,
    config: dict,
    progress_bar: str | None = None
) -> dict:
    """
    Trains a feedforward network with time-to-first-spike encoding on MNIST.

    The pixel values are binned into `Nin_virtual+1` bins, each corresponding to an
    input spike time except for the last bin, which is ignored. The effect of all inputs
    in each bin is captured by a virtual input neuron under the hood to speed up the
    simulation. See `transform_image` and `eventffwd` for details. The trained
    parameters `p` are the feedforward weights of the network and the initial phases of
    the neurons.

    Args:
        neuron:
            Phase oscillator model including pseudodynamics.
        config:
            Simulation configuration. Needs to contain the following items:
                `seed`: Random seed
                `Nin`: Number of input neurons, has to be 28*28 for MNIST
                `Nin_virtual`: Number of virtual input neurons
                `Nhidden`: Number of hidden neurons per layer
                `Nlayer`: Number of layers
                `Nout`: Number of output neurons, has to be 10 for MNIST
                `w_scale`: Scale of the initial weights
                `T`: Trial duration
                `K`: Maximal number of simulated ordinary spikes
                `dt`: Integration time step (for state traces)
                `gamma`: Regularization strength
                `Nbatch`: Batch size
                `lr`: Learning rate
                `tau_lr`: Learning rate decay time constant
                `beta1`: Adabelief parameter
                `beta2`: Adabelief parameter
                `p_flip`: Probability of flipping input pixels
                `Nepochs`: Number of epochs
        progress_bar:
            Whether to use 'notebook' or 'script' tqdm progress bar or `None`.
    Returns:
        A dictionary containing detailed learning dynamics.
    """

    ### Unpack arguments
    seed: int = config["seed"]
    Nin_virtual: int = config["Nin_virtual"]
    Nhidden: int = config["Nhidden"]
    Nlayer: int = config["Nlayer"]
    Nout: int = config["Nout"]
    N = Nhidden * (Nlayer - 1) + Nout
    Nepochs: int = config["Nepochs"]
    p_flip: float = config["p_flip"]
    lr: float = config["lr"]
    tau_lr: float = config["tau_lr"]
    beta1: float = config["beta1"]
    beta2: float = config["beta2"]
    theta = neuron.Theta()
    if progress_bar == "notebook":
        trange = trange_notebook
    elif progress_bar == "script":
        trange = trange_script
    else:
        trange = range

    ### Set up the simulation

    # Gradient
    @jit
    @partial(value_and_grad, has_aux=True)
    def gradfn(
        p: list, input: Float[Array, "Batch Nin"], labels: Float[Array, " Batch"],  xt_right, xt_left, xt_initial, u_bc_right_label, u_bc_left_label, u_ic_label
    ) -> tuple[Array, Array]:
        loss, acc = simulatefn(neuron, p, input, labels, config, xt_right, xt_left, xt_initial, u_bc_right_label, u_bc_left_label, u_ic_label)
        return loss, acc

    # Regularization
    @jit
    def flip(key: Array, input: Array) -> tuple[Array, Array]:
        key, subkey = jax.random.split(key)
        mask = jax.random.bernoulli(subkey, p=p_flip, shape=input.shape)
        return key, jnp.where(mask, Nin_virtual - input, input)

    # Optimization step
    @jit
    def trial(
        p: list,
        input: Float[Array, "Batch Nin"],
        labels: Float[Array, " Batch"],
        opt_state: optax.OptState,
        xt_right, xt_left, xt_initial, u_bc_right_label, u_bc_left_label, u_ic_label
    ) -> tuple:
        (loss, acc), grad = gradfn(p, input, labels, xt_right, xt_left, xt_initial, u_bc_right_label, u_bc_left_label, u_ic_label)
        updates, opt_state = optim.update(grad, opt_state)
        p = optax.apply_updates(p, updates)  # type: ignore
        p[1] = jnp.clip(p[1], 0, theta)
        return loss, acc, p, opt_state

    # Probe network
    @jit
    def jprobefn(p, input, labels, xt_right, xt_left, xt_initial, u_bc_right_label, u_bc_left_label, u_ic_label):
        return probefn(neuron, p, input, labels, config, xt_right, xt_left, xt_initial, u_bc_right_label, u_bc_left_label, u_ic_label)

    def probe(p: list) -> dict:
        metrics = {
            "loss": 0.0,
            "acc": 0.0,
            "loss_ord": 0.0,
            "acc_ord": 0.0,
            "activity": 0.0,
            "activity_first": 0.0,
        }
        silents = {
            "silent_neurons": jnp.ones(N - Nout, dtype=bool),
            "silent_neurons_first": jnp.ones(N - Nout, dtype=bool),
        }
        steps = len(train_loader)
        for data in train_loader:
            # input, labels = jnp.array(data[0]), jnp.array(data[1])
            input, labels, xt_right, xt_left, xt_initial, u_bc_right_label, u_bc_left_label, u_ic_label = jnp.array(data[0]), jnp.array(data[1]), jnp.array(data[2]), jnp.array(data[3]), jnp.array(data[4]), jnp.array(data[5]), jnp.array(data[6]), jnp.array(data[7])
            metric, silent = jprobefn(p, input, labels, xt_right, xt_left, xt_initial, u_bc_right_label, u_bc_left_label, u_ic_label)
            metrics = {k: metrics[k] + metric[k] / steps for k in metrics}
            silents = {k: silents[k] & silent[k] for k in silents}
        for k, v in silents.items():
            metrics[k] = jnp.mean(v).item()
        return metrics

    ### Simulation

    # Data
    torch.manual_seed(seed)
    train_loader, test_loader = load_data(datasets.MNIST, "data", config)

    # Parameters
    key = random.PRNGKey(seed)
    key, weights = init_weights(key, config)
    phi0 = init_phi0(neuron, config)
    p = [weights, phi0]
    p_init = [weights, phi0]

    # Optimizer
    # schedule = optax.exponential_decay(lr, int(tau_lr * len(train_loader)), 1 / jnp.e)
    schedule1 = optax.piecewise_constant_schedule(
    init_value=lr,
    boundaries_and_scales={
        9*5000: 0.1,    # At step 3000, multiply by 0.1 (1e-3 * 0.1 = 1e-4)
        9*10000: 0.1     # At step 8000, multiply by 0.1 (1e-4 * 0.1 = 1e-5)
    
    }
    )
    # optim = optax.adabelief(schedule, b1=beta1, b2=beta2)
    optim = optax.adabelief(schedule1, b1=beta1, b2=beta2)
    # optim = optax.adam(schedule)
    opt_state = optim.init(p)

    # Metrics
    metrics: dict[str, Array | list] = {k: [v] for k, v in probe(p).items()}

  
    train_loss = []


    if os.path.exists('qif_burger_pinn.pkl'):
            print(f"Loading parameters from {'burger_pinn_params.pkl'}")
            p = safe_load_params('burger_pinn_params.pkl')
      
    else:
        print(f"No saved parameters found. Training from scratch.")

    # Training
    pre_loss = 10000000
    for epoch in trange(Nepochs):
        epoch_loss = 0
        batch_count = 0
        for data in train_loader:
            input, labels, xt_right, xt_left, xt_initial, u_bc_right_label, u_bc_left_label, u_ic_label = jnp.array(data[0]), jnp.array(data[1]), jnp.array(data[2]), jnp.array(data[3]), jnp.array(data[4]), jnp.array(data[5]), jnp.array(data[6]), jnp.array(data[7])
            key, input = flip(key, input)

            loss, acc, p, opt_state = trial(p, input, labels, opt_state, xt_right, xt_left, xt_initial, u_bc_right_label, u_bc_left_label, u_ic_label)

            epoch_loss += loss
            batch_count += 1
        avg_epoch_loss = epoch_loss / batch_count
        train_loss.append(loss)

        if epoch%100==0:
            print('loss:', avg_epoch_loss)
            if avg_epoch_loss <= pre_loss:
                safe_save_params(p, 'qif_burger_pinn.pkl')
                pre_loss = avg_epoch_loss
                

        # Probe network
        metric = probe(p)
        metrics = {k: v + [metric[k]] for k, v in metrics.items()}


    def normalize_test(x1,T):
      normalized_x1 = (x1 - jnp.min(x1))/(jnp.max(x1)-jnp.min(x1))
      normalized_x1=(1-normalized_x1)*T
      normalized_x1 = normalized_x1.reshape(-1,1)
      return normalized_x1


    data_test = sio.loadmat('experiments/pinnst/burgers_shock.mat')
    # encoded_inputs = data['encoded_inputs']
    x_test = data_test['x']
    x_test = x_test[:, 0]
    t_test = data_test['t']
    t_test = t_test[:, 0]
    u_test = data_test['usol']

    n_test = 1 #downsample
    x_min_test = jnp.array([jnp.min(x_test)])
    x_max_test = jnp.array([jnp.max(x_test)])
    t_min_test = jnp.array([jnp.min(t_test)])
    t_max_test = jnp.array([jnp.max(t_test)])
    x_in_test = x_test[::n_test]
    t_in_test = t_test[::n_test]

    x_test = jnp.unique(jnp.concatenate([x_min_test, x_in_test, x_max_test]))
    t_test = jnp.unique(jnp.concatenate([t_min_test, t_in_test, t_max_test]))
    print(x_test.shape, t_test.shape)


    X1_test, X2_test = jnp.meshgrid(x_test, t_test)
    X1_test = X1_test.ravel()
    X2_test = X2_test.ravel()
    print(X1_test.shape, X2_test.shape)
    X_test = jnp.column_stack((X1_test, X2_test))
    x1_test = X_test[:, 0]
    x2_test = X_test[:, 1]
    print(jnp.min(x1_test), jnp.max(x1_test), jnp.min(x2_test), jnp.max(x2_test))

    normalized_x1_test = normalize_test(x1_test,2.0)
    normalized_x2_test = normalize_test(x2_test,2.0)
   
    encoded_inputs_test = jnp.concat((normalized_x1_test, normalized_x2_test), axis=1)


    
    # Run the network simulation on each input using eventffwd.
    # vmap over the batch dimension (each input sample).
    outs_test = vmap(eventffwd, in_axes=(None, None, 0, None))(
        neuron, p, encoded_inputs_test, config
    )
    
    # # Decode the network’s continuous output using the regression output decoder.
    # # Each prediction will be an array of shape (1,), so squeeze to get a 1D array.
    t_outs_test = vmap(outfn, in_axes=(None, 0, None, None))(
        neuron, outs_test, p, config
    )
    preds_raw_test = jnp.squeeze(t_outs_test[:,1]-t_outs_test[:,0])

    u_ic_test = -jnp.sin(jnp.pi * (1-encoded_inputs_test[:,0]))
    
    # Boundary multiplier (vanishes at x=-1,1)
    boundary_mult_test = (-1-(1-encoded_inputs_test[:,0])) * (1-(1-encoded_inputs_test[:,0]))
    
    # Time evolution term
    # time_mult = jnp.tanh(((0.99/2)*(2-input[1])))  # or just t
    time_mult_test = jnp.exp(-3.0*((0.99/2)*(2-encoded_inputs_test[:,1])))  # or just t
    
    # Construct solution that automatically satisfies constraints
    # pred = u_ic * (1 - time_mult) + time_mult * boundary_mult * pred_raw
    pred_test = u_ic_test * time_mult_test + ((0.99/2)*(2-encoded_inputs_test[:,1])) * boundary_mult_test * preds_raw_test

    sio.savemat('experiments/pinnst/qif_pinn_burger_result.mat',{
    'x': x_test,
    't': t_test,
    'u_true': u_test,
    'u_pred': pred_test,
    'loss': train_loss
    })
   


    if jnp.any(jnp.isnan(jnp.array(metrics["loss"]))):
        print(
            "Warning: A NaN appeared. "
            "Likely not enough spikes have been simulated. "
            "Try increasing `K`."
        )
    metrics = {k: jnp.array(v) for k, v in metrics.items()}
    p_end = p
    metrics["p_init"] = p_init
    metrics["p_end"] = p_end

    return metrics


# %%
############################
### Examples
############################


def run_example(p: list, neuron: AbstractPseudoPhaseOscNeuron, config: dict) -> dict:
    """
    Simulates the network for a single example input given the parameters `p`.
    """

    ### Unpack arguments
    seed: int = config["seed"]
    Nhidden: int = config["Nhidden"]
    Nlayer: int = config["Nlayer"]
    Nout: int = config["Nout"]
    N = Nhidden * (Nlayer - 1) + Nout

    ### Set up the simulation
    @jit
    def jeventffwd(p, input):
        return eventffwd(neuron, p, input, config)

    @jit
    def joutfn(out, p):
        return outfn(neuron, out, p, config)

    ### Run simulation

    # Data
    torch.manual_seed(seed)
    _, test_loader = load_data(datasets.MNIST, "data", config)

    input, label = next(iter(test_loader))
    # input, label = jnp.array(input[2]), jnp.array(label[2])
    input, label = jnp.array(input[60]), jnp.array(label[60])
    out = jeventffwd(p, input)
    t_outs = joutfn(out, p)

    ### Prepare results
    times: Array = out[0]
    spike_in: Array = out[1]
    neurons: Array = out[2]

    trace_ts, trace_xs = neuron.traces(p[1][jnp.newaxis], out, config)
    trace_phis = trace_xs[:, 0]
    trace_Vs = neuron.iPhi(trace_phis)

    spiketimes = []
    for i in range(N):
        times_i = times[~spike_in & (neurons == i)]
        spiketimes.append(times_i)
    # predicted = jnp.argmin(t_outs)
    predicted = t_outs[1]-t_outs[0]

    ### Pack results in dictionary
    results = {
        "input": input,
        "label": label,
        "predicted": predicted,
        "trace_ts": trace_ts,
        "trace_phis": trace_phis,
        "trace_Vs": trace_Vs,
        "spiketimes": spiketimes,
    }
    print('input',results['input'])
    print('label', results['label'])
    print('out spike times', t_outs)
    print('pred', predicted)

    return results


# %%
############################
### Plotting
############################


def plot_spikes(ax: Axes, example: dict, config: dict) -> None:
    ### Unpack arguments
    T: float = config["T"]
    Nhidden: int = config["Nhidden"]
    Nlayer: int = config["Nlayer"]
    Nout: int = config["Nout"]
    N = (Nlayer - 1) * Nhidden + Nout
    spiketimes: Array = example["spiketimes"]

    ### Plot spikes
    tick_len = 2
    ax.eventplot(spiketimes, colors="k", linewidths=0.5, linelengths=tick_len)
    patch = Rectangle((0, Nhidden - 1 / 2), T, Nhidden, color="k", alpha=0.2, zorder=0)
    ax.add_patch(patch)
    ax.text(
        T,
        0,
        r"$1^\mathrm{st}$ hidden",
        ha="right",
        va="bottom",
        color="k",
        alpha=0.2,
        zorder=1,
    )
    ax.text(
        T,
        Nhidden - 1 / 2,
        r"$2^\mathrm{nd}$ hidden",
        ha="right",
        va="bottom",
        color="white",
        zorder=1,
    )
    ax.text(
        T,
        2 * Nhidden - 1 / 2,
        "Output",
        ha="right",
        va="bottom",
        color="k",
        alpha=0.2,
        zorder=1,
    )

    ### Formatting
    ax.set_xticks([0, T])
    ax.set_xlim(0, T)
    ax.set_xlabel("Time $t$", labelpad=-3)
    ax.set_yticks(
        [0, Nhidden - 1, 2 * Nhidden - 1, N - 1],
        [str(1), str(Nhidden), str(2 * Nhidden), str(N)],
    )
    ax.set_ylim(-tick_len / 2, N - 1 + tick_len / 2)
    ax.set_ylabel("Neuron", labelpad=-0.1)


def plot_error(ax: Axes, metrics: dict, config: dict) -> None:
    ### Unpack arguments
    Nepochs: int = config["Nepochs"]
    acc: Array = metrics["acc"]
    mean_acc = jnp.mean(acc, 0)
    std_acc = jnp.std(acc, 0)
    acc_ord: Array = metrics["acc_ord"]
    mean_acc_ord = jnp.mean(acc_ord, 0)
    std_acc_ord = jnp.std(acc_ord, 0)
    epochs = jnp.arange(1, Nepochs + 2)

    ### Plot classification error
    ax.plot(epochs, 1 - mean_acc_ord, label="Excl. pseudo", c="C0", zorder=1)
    ax.fill_between(
        epochs,
        1 - mean_acc_ord - std_acc_ord,
        1 - mean_acc_ord + std_acc_ord,
        alpha=0.3,
        color="C0",
    )
    ax.plot(epochs, 1 - mean_acc, label="Incl. pseudo", c="C1", zorder=0)
    ax.fill_between(
        epochs,
        1 - mean_acc - std_acc,
        1 - mean_acc + std_acc,
        alpha=0.3,
        color="C1",
    )
    ax.legend()

    ### Formatting
    ax.set_xlabel("Epochs + 1", labelpad=-1)
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(formatter)
    ax.set_ylim(0.01, 1)
    ax.set_ylabel("Test error", labelpad=-3)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(formatter)


def plot_traces(ax: Axes, example: dict, config: dict) -> None:
    ### Unpack arguments
    T: float = config["T"]
    Nhidden: int = config["Nhidden"]
    Nlayer: int = config["Nlayer"]
    Nout: int = config["Nout"]
    N = (Nlayer - 1) * Nhidden + Nout

    ### Unpack example
    trace_ts: Array = example["trace_ts"]
    trace_Vs: Array = example["trace_Vs"]

    ### Plot
    ax.axhline(0, c="gray", alpha=0.3, zorder=-1)
    ax.plot([-0.1, -0.1], [0, 1], c="k", clip_on=False)
    for i in range(10):
        ax.plot(trace_ts, trace_Vs[:, N - Nout + i], color=petroff10[i])
        ax.text((i % 5) * 0.15, -4 - (i // 5) * 3, str(i), color=petroff10[i])

    ### Formatting
    ax.set_xticks([0, T])
    ax.set_xlim(0, T)
    ax.set_xlabel("Time $t$", labelpad=-3)
    ax.set_yticks([])
    ax.set_ylim(-8, 8)
    ax.set_ylabel("Potential $V$")
    ax.spines["left"].set_visible(False)

