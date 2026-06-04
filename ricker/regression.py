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
import pickle


from spikegd.models import AbstractPhaseOscNeuron, AbstractPseudoPhaseOscNeuron
from spikegd.utils.plotting import formatter, petroff10
import scipy.io as sio
import numpy as np

# %%
############################
### Data loading
############################

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


def generate_regression_data(num_samples: int, T):
    """Generate regression data with Gaussian receptive field encoding."""
   
    x1 = np.linspace(-1, 1, 100)
    x2 = np.linspace(-1, 1, 100)
    X1, X2 = jnp.meshgrid(x1, x2)
    x1_min = -1
    x1_max = 1
    x2_min = -1
    x2_max = 1
    
    # without encoding
    normalized_x1 = (X1 - x1_min)/(x1_max-x1_min)
    normalized_x1=(1-normalized_x1)*T
    normalized_x1 = normalized_x1.reshape(-1,1)
    normalized_x2 = (X2 - x2_min)/(x2_max-x2_min)
    normalized_x2=(1-normalized_x2)*T
    normalized_x2 = normalized_x2.reshape(-1,1)
    encoded_inputs = jnp.concat((normalized_x1, normalized_x2), axis=1)
    # ricker
    ys = (1/((jnp.pi)*(0.8**4)))*(1-0.5*((X1**2+X2**2)/(0.8**2)))*jnp.exp(-0.5*((X1**2+X2**2)/(0.8**2)))
   
    ys = ys.reshape(-1, 1)
    encoded_inputs = np.array(encoded_inputs)
    ys = np.array(ys)


    return encoded_inputs, ys


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

    # Training set: Generate regression data
    num_train = config.get("num_train", 10000)
    train_inputs, train_targets = generate_regression_data(num_train, 2.0)
    # Convert JAX arrays to PyTorch tensors
    train_set = TensorDataset(torch.tensor(train_inputs), torch.tensor(train_targets))
    train_loader = DataLoader(train_set, batch_size=Nbatch, shuffle=True)

    # Test set: Generate regression data
    num_test = config.get("num_test", 10000)
    test_inputs, test_targets = generate_regression_data(num_test, 2.0)
    test_set = TensorDataset(torch.tensor(test_inputs), torch.tensor(test_targets))
    test_loader = DataLoader(test_set, batch_size=100, shuffle=True)

    return train_loader, test_loader





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
    # print(weights_in.shape)
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


def lossfn(
    t_out: jnp.ndarray, target: jnp.ndarray, config: dict
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
    # For regression, we assume t_out is an array of shape (1,)
    loss = ((t_out[1]-t_out[0]) - target) ** 2

    # Define a threshold for acceptable error (default threshold 0.1)
    threshold = config.get("reg_threshold", 0.1)
    correct = jnp.abs(t_out[0] - target) < threshold
    
    return loss, correct




def simulatefn(
    neuron: AbstractPseudoPhaseOscNeuron,
    p: list,
    input: Float[Array, "Batch Nin"],
    labels: Int[Array, " Batch"],
    config: dict,
) -> tuple[Array, Array]:
    """
    Simulates the network and computes the loss and accuracy for batched input.
    """
    outs = vmap(eventffwd, in_axes=(None, None, 0, None))(neuron, p, input, config)
    t_outs = vmap(outfn, in_axes=(None, 0, None, None))(neuron, outs, p, config)
    loss, correct = vmap(lossfn, in_axes=(0, 0, None))(t_outs, labels, config)
    mean_loss = jnp.mean(loss)
    accuracy = jnp.mean(correct)
    return mean_loss, accuracy


def probefn(
    neuron: AbstractPseudoPhaseOscNeuron,
    p: list,
    input: Float[Array, "Batch Nin"],
    labels: Int[Array, " Batch"],
    config: dict,
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

    @vmap
    def batch_lossfn(t_outs, labels):
        return lossfn(t_outs, labels, config)

    ### Run network
    outs = batch_eventffwd(input)
    times: Array = outs[0]
    spike_in: Array = outs[1]
    neurons: Array = outs[2]
    t_outs = batch_outfn(outs)

    ### Loss and accuracy with pseudospikes
    loss, correct = batch_lossfn(t_outs, labels)
    mean_loss = jnp.mean(loss)
    acc = jnp.mean(correct)

    ### Loss and accuracy without pseudospikes
    t_out_ord = jnp.where(t_outs < T, t_outs, T)
    loss_ord, correct_ord = batch_lossfn(t_out_ord, labels)
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
    progress_bar: str | None = None,
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
        p: list, input: Float[Array, "Batch Nin"], labels: Int[Array, " Batch"]
    ) -> tuple[Array, Array]:
        loss, acc = simulatefn(neuron, p, input, labels, config)
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
        labels: Int[Array, " Batch"],
        opt_state: optax.OptState,
    ) -> tuple:
        (loss, acc), grad = gradfn(p, input, labels)
        updates, opt_state = optim.update(grad, opt_state)
        p = optax.apply_updates(p, updates)  # type: ignore
        p[1] = jnp.clip(p[1], 0, theta)
        return loss, acc, p, opt_state

    # Probe network
    @jit
    def jprobefn(p, input, labels):
        return probefn(neuron, p, input, labels, config)

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
        steps = len(test_loader)
        for data in test_loader:
            input, labels = jnp.array(data[0]), jnp.array(data[1])
            metric, silent = jprobefn(p, input, labels)
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

  
    optim = optax.adabelief(lr, b1=beta1, b2=beta2)
    opt_state = optim.init(p)

    # Metrics
    metrics: dict[str, Array | list] = {k: [v] for k, v in probe(p).items()}

 
    train_loss = []

    # if os.path.exists('qif_ricker_params.pkl'):
    #         print(f"Loading parameters from {'qif_ripple_params.pkl'}")
    #         p = safe_load_params('qif_ripple_params.pkl')
      
    # else:
    #     print(f"No saved parameters found. Training from scratch.")

    # Training
    pre_loss = 10000000
    for epoch in trange(Nepochs):
        epoch_loss = 0
        batch_count = 0
        for data in train_loader:
            input, labels = jnp.array(data[0]), jnp.array(data[1])
            key, input = flip(key, input)

            loss, acc, p, opt_state = trial(p, input, labels, opt_state)
       
            epoch_loss += loss
            batch_count += 1
        avg_epoch_loss = epoch_loss / batch_count
        train_loss.append(loss)

        if epoch%100==0:
            print('loss:', avg_epoch_loss)
            if avg_epoch_loss <= pre_loss:
                safe_save_params(p, 'qif_ricker_params.pkl')
                pre_loss = avg_epoch_loss
        # Probe network
        metric = probe(p)
        metrics = {k: v + [metric[k]] for k, v in metrics.items()}
             

    T_snn = config['T']
    x1_test = np.linspace(-1,1,300)
    x2_test = np.linspace(-1,1,300)
    X1_test, X2_test = jnp.meshgrid(x1_test, x2_test)
    x1_test_min=-1
    x1_test_max=1
    x2_test_min=-1
    x2_test_max=1
    # without encoding
    normalized_x1_test = (X1_test - x1_test_min)/(x1_test_max-x1_test_min)
    normalized_x1_test=(1-normalized_x1_test)*T_snn
    normalized_x1_test = normalized_x1_test.reshape(-1,1)
    normalized_x2_test = (X2_test - x2_test_min)/(x2_test_max-x2_test_min)
    normalized_x2_test=(1-normalized_x2_test)*T_snn
    normalized_x2_test = normalized_x2_test.reshape(-1,1)
    
    encoded_inputs_test = jnp.concat((normalized_x1_test, normalized_x2_test), axis=1)
    encoded_inputs_test = np.array(encoded_inputs_test)
    # ricker
    ys = (1/((jnp.pi)*(0.8**4)))*(1-0.5*((X1_test**2+X2_test**2)/(0.8**2)))*jnp.exp(-0.5*((X1_test**2+X2_test**2)/(0.8**2)))
    Y_test = ys.reshape(300, 300)
    Y_test = np.array(Y_test)


    @jit
    def test_forward(p, inputs):
        outs = vmap(eventffwd, in_axes=(None, None, 0, None))(
            neuron, p, inputs, config
        )
        t_outs = vmap(outfn, in_axes=(None, 0, None, None))(
            neuron, outs, p, config
        )
        return t_outs


    t_outs_test = test_forward(p, encoded_inputs_test)

  

    preds_test = jnp.squeeze(t_outs_test[:,1]-t_outs_test[:,0])

    sio.savemat('experiments/regression_2d/qif_ricker_2d_result.mat',{
    'x1': x1_test,
    'x2': x2_test,
    'y_true': Y_test.reshape(300, 300),
    'y_pred': preds_test.reshape(300, 300),
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
    input, label = jnp.array(input[2]), jnp.array(label[2])
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
    predicted = jnp.argmin(t_outs)

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
