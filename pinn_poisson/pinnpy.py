import jax
import jax.numpy as jnp
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from jax import random
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np

from pinn import plot_error, plot_spikes, plot_traces, run, run_example, eventffwd, outfn
from spikegd.theta import ThetaNeuron
from spikegd.utils.plotting import (
    cm2inch,
    panel_label,
)

from spikegd.lif import LIFNeuron

from spikegd.olif import OscLIFNeuron

from spikegd.qif import QIFNeuron
import numpy as np
import os
import pickle

# Load parameters
def safe_load_params(filepath):
    """Load JAX parameters safely using pickle"""
    with open(filepath, 'rb') as f:
        params = pickle.load(f)
    print(f"Parameters loaded from {filepath}")
    return params

plt.style.use("spikegd.utils.plotstyle")


config_theta = {
    "seed": 0,
    # Neuron
    "tau": 6 / jnp.pi,
    "I0": 5 / 4,
    "eps": 1e-6,
    # Network
    "Nin": 1,
    "Nin_virtual": 1,  # #Virtual input neurons = #Pixel value bins - 1
    #"Nhidden": 64,
    "Nhidden": 4,
    "Nlayer": 2,  # Number of layers
    "Nout": 2,
    "w_scale": 0.9,  # Scaling factor of initial weights
    # Trial
    "T": 2.0,
    "K": 200,  # Maximal number of simulated ordinary spikes
    "dt": 0.001,  # Step size used to compute state traces
    # Training
    "gamma": 0,
    "Nbatch": 50,
    "lr": 1e-2,
    "tau_lr": 1e2,
    "beta1": 0.9,
    "beta2": 0.999,
    "p_flip": 0.0,
    "Nepochs": 500}



def run_theta(config: dict) -> dict:
    """
    Wrapper to train a network of Theta neurons with the given configuration.

    See docstring of `run` and article for more information.
    """
    tau, I0, eps = config["tau"], config["I0"], config["eps"]
    neuron = ThetaNeuron(tau, I0, eps)
    # metrics = run(neuron, config, progress_bar="notebook")
    metrics = run(neuron, config, progress_bar="script")
    return metrics


seed = 0
samples = 1 # Number of network realizations, decrease to save simulation time
key = random.PRNGKey(seed)
seeds = random.randint(key, (samples,), 0, jnp.uint32(2**32 - 1), dtype=jnp.uint32)
metrics_list = []
for seed in seeds:
    config_theta["seed"] = seed
    metrics = run_theta(config_theta)

    metrics_list.append(metrics)
metrics_example = metrics_list[0]
metrics = jax.tree.map(lambda *args: jnp.stack(args), *metrics_list)


def summarize_metrics(metrics: dict, epoch: int) -> None:
    """
    Print a summary of the metrics at the given epoch.
    """
    summary_metrics = {k: v for k, v in metrics.items() if k not in ["p_init", "p_end"]}
    summary_metrics = jax.tree.map(
        lambda x: jnp.array([jnp.mean(x[:, epoch]), jnp.std(x[:, epoch])]),
        summary_metrics,
    )
    for key, value in summary_metrics.items():
        print(f"{key:<25} {value[0]:.3f} ± {value[1]:.3f}")


print("**Results before training**")
summarize_metrics(metrics, 0)
print()
print("**Results after training**")
summarize_metrics(metrics, -1)


def run_example_theta(p: list, config: dict) -> dict:
    """
    Wrapper to run network on one example input.

    See docstring of `run_example` and article for more information.
    """
    tau, I0, eps = config["tau"], config["I0"], config["eps"]
    neuron = ThetaNeuron(tau, I0, eps)
    metrics = run_example(p, neuron, config)
    return metrics



def run_example_olif(p: list, config: dict) -> dict:
    """
    Wrapper to run network on one example input.

    See docstring of `run_example` and article for more information.
    """
    tau, I0, V_th = config["tau"], config["I0"], config["V_th"]
    neuron = OscLIFNeuron(tau, I0, V_th)
    metrics = run_example(p, neuron, config)
    return metrics



example_init = run_example_theta(metrics_example["p_init"], config_theta)
example_end = run_example_theta(metrics_example["p_end"], config_theta)




### Figure
fig = plt.figure(figsize=cm2inch(1.5 * 8.6, 1.5 * 6.0))
gs = gridspec.GridSpec(
    2,
    3,
    figure=fig,
    hspace=0.5,
    wspace=0.4,
    top=0.94,
    bottom=0.15,
    left=0.12,
    right=0.97,
)

#### Spike plot before learning
# Spike plot before learning for regression
ax = fig.add_subplot(gs[:, 0])

plot_spikes(ax, example_init, config_theta) 


ax.set_title("Epoch 0", pad=-1)
# Instead of an inset image, show a text annotation of the input value
ax.text(0.05, 0.95, f"Input: {example_init['input'][0]:.2f}", transform=ax.transAxes,
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
panel_label(fig, ax, "(a)", x=-0.4, y=0.07)


#### Spike plot after learning
ax = fig.add_subplot(gs[:, 1])

plot_spikes(ax, example_end, config_theta)
ax.set_title(f"Epoch {config_theta['Nepochs']}", pad=-1)


ax.tick_params(labelleft=False)
ax.set_ylabel("")

### Trace output
ax = fig.add_subplot(gs[0, 2])

plot_traces(ax, example_end, config_theta)


panel_label(fig, ax, "(b)", x=-0.4, y=0.0)

def plot_loss(ax, metrics: dict, config: dict) -> None:
    """
    Plots the mean squared error (MSE) loss over training epochs.
    """
    # Assume metrics["loss"] might be a 2D array with shape (runs, epochs)
    loss = metrics["loss"]
    # If there's only one run, squeeze out the extra dimension:
    loss = jnp.squeeze(loss)  # Now loss.shape should match the number of epochs
    epochs = jnp.arange(1, loss.size + 1)
    ax.plot(epochs, loss.reshape(-1,), label="MSE Loss", color="C0", zorder=1)
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Mean Squared Error")
    ax.set_title("Loss over Training")
    ax.legend()


#### Loss Plot (MSE)
ax = fig.add_subplot(gs[1, 2])

plot_loss(ax, metrics, config_theta)



panel_label(fig, ax, "(c)", x=-0.4, y=0.0)

plt.show()




import matplotlib.pyplot as plt
from jax import vmap
import jax.numpy as jnp
from spikegd.models import AbstractPhaseOscNeuron, AbstractPseudoPhaseOscNeuron, AbstractPseudoIFNeuron


def plot_parabola(neuron: AbstractPhaseOscNeuron, p, config):
# def plot_parabola(neuron: AbstractPseudoIFNeuron, p, config):
    """
    Plots the ground truth parabola y = x^2 and the learned prediction of the network.
    
    Args:
        neuron: The spiking neuron model.
        p: The network parameters (e.g. learned weights and initial phases).
        config: Configuration dictionary (must include keys like 'Nin', 'Nhidden', 'Nlayer',
                'Nout', 'T', etc. for simulation).
    """
    # Generate a set of test inputs uniformly in [0, 1]
    x_vals = jnp.linspace(0, 1, 5000)
    # print(x_vals)

    min_val = jnp.min(x_vals)
    max_val = jnp.max(x_vals)


    nomalized = (x_vals - jnp.min(x_vals))/(jnp.max(x_vals)-jnp.min(x_vals))
    encoded_inputs=(1-nomalized)*config_theta["T"]
    encoded_inputs=encoded_inputs.reshape(-1,1)


    
    # Run the network simulation on each input using eventffwd.
    # vmap over the batch dimension (each input sample).
    outs = vmap(eventffwd, in_axes=(None, None, 0, None))(
        neuron, p, encoded_inputs, config
    )
    
    # # Decode the network’s continuous output using the regression output decoder.
    # # Each prediction will be an array of shape (1,), so squeeze to get a 1D array.
    t_outs = vmap(outfn, in_axes=(None, 0, None, None))(
        neuron, outs, p, config
    )
    preds_raw = jnp.squeeze(t_outs[:,1]-t_outs[:,0])


    # Apply zero boundary conditions using the method: u(x) = x(1-x) * N(x)
    # This ensures u(0) = u(1) = 0 automatically
    preds = encoded_inputs.squeeze() * (2 - encoded_inputs.squeeze()) * preds_raw



    def forwardfn(neuron: AbstractPseudoPhaseOscNeuron, p: list, input, config: dict):
        """
        Computes the output of the network for single input.
        """
        outs = eventffwd(neuron, p, input, config)
        t_outs = outfn(neuron, outs, p, config)
        pred_raw = t_outs[1] - t_outs[0]
        pred = input * (2 - input) * pred_raw
        return pred
    
    def forwardfn_scalar(neuron, p, input, config):
        """Ensure scalar output"""
        result = forwardfn(neuron, p, input, config)
        # Handle various possible shapes
        while jnp.ndim(result) > 0:
            result = result[0] if result.shape[0] == 1 else jnp.squeeze(result)
        return result

    def first_derivative_scalar(neuron, p, input, config):
        """First derivative that returns scalar"""
        grad_fn = jax.grad(forwardfn_scalar, argnums=2)
        result = grad_fn(neuron, p, input, config)
        # Ensure scalar output
        while jnp.ndim(result) > 0:
            result = result[0] if result.shape[0] == 1 else jnp.squeeze(result)
        return result
    

    # Now compute derivatives
    dydt = jax.vmap(first_derivative_scalar, in_axes=(None, None, 0, None))(neuron, p, encoded_inputs, config)

    fn_d2ydt2 = jax.grad(first_derivative_scalar, argnums=2)
    d2ydt2 = jax.vmap(fn_d2ydt2, in_axes=(None, None, 0, None))(neuron, p, encoded_inputs, config)

    dydx = -2*dydt

   
    d2ydx2 = 4*d2ydt2
    
    
    # Compute the ground truth: y = x^2.
    # ground_truth = (x_vals.squeeze()) ** 2
    ground_truth = 2 * jnp.sin(jnp.pi * x_vals.squeeze())

    print('mse', jnp.mean((ground_truth-preds)**2))
    print('l2', jnp.linalg.norm(preds-ground_truth)/jnp.linalg.norm(ground_truth))
    pde = -2*((jnp.pi)**2)*jnp.sin(jnp.pi*x_vals.squeeze())
    pde = pde.reshape(-1, 1)
    print('residual', jnp.mean((pde-d2ydx2)**2))

    # Plot the curves
    plt.figure(figsize=(6, 4))
    plt.plot(x_vals.squeeze(), ground_truth, label="Ground Truth", color="blue")
    plt.plot(x_vals.squeeze(), preds, label="Learned Prediction", color="red", linestyle="dotted")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Comparison of Ground Truth and Learned Function")
    plt.legend()
    plt.show()
    plt.savefig('regression.png')

    # Plot 1st grad
    plt.figure(figsize=(6, 4))
    plt.plot(x_vals.squeeze(), dydx, label="pred: dy/dx", color="red")
    plt.plot(x_vals.squeeze(), 2*jnp.pi*jnp.cos(jnp.pi*x_vals.squeeze()), label="true: dy/dx", color="blue")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Gradient")
    plt.legend()
    plt.savefig('g.png')
    plt.show()

    # Plot 2nd grad
    plt.figure(figsize=(6, 4))
    plt.plot(x_vals.squeeze(), d2ydx2, label="pred: d2y/dx2", color="red")
    plt.plot(x_vals.squeeze(), -2*((jnp.pi)**2)*jnp.sin(jnp.pi*x_vals.squeeze()), label="true: d2y/dx2", color="blue")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Gradient")
    plt.legend()
    plt.savefig('g2.png')
    plt.show()



from spikegd.theta import ThetaNeuron
neuron = ThetaNeuron(config_theta["tau"], config_theta["I0"], config_theta["eps"])
plot_parabola(neuron, metrics_example["p_end"], config_theta)
