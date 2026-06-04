import jax
import jax.numpy as jnp
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from jax import random
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import scipy.io as sio

from experiments.pinnst.pinn_burger import plot_error, plot_spikes, plot_traces, run, run_example, eventffwd, outfn
from spikegd.theta import ThetaNeuron
from spikegd.utils.plotting import (
    cm2inch,
    panel_label,
)

from spikegd.lif import LIFNeuron

from spikegd.olif import OscLIFNeuron

from spikegd.qif import QIFNeuron

def check_gpu_availability():
    """Check if GPU is available and print device info"""
    print(f"JAX version: {jax.__version__}")
    print(f"Available devices: {jax.devices()}")
    print(f"Default backend: {jax.default_backend()}")
    
    # Check if GPU is available
    try:
        gpu_device = jax.devices('gpu')[0]
        print(f"GPU device found: {gpu_device}")
        return True
    except:
        print("No GPU device found, using CPU")
        return False

# Check GPU availability
gpu_available = check_gpu_availability()

if gpu_available:
    print("✓ GPU is available and will be used for computations")
else:
    print("⚠ GPU not available, using CPU (computations will be slower)")

plt.style.use("spikegd.utils.plotstyle")



config_theta = {
    "seed": 0,
    # Neuron
    "tau": 6 / jnp.pi,
    "I0": 5 / 4,
    "eps": 1e-6,
    # Network
    "Nin": 2,
    "Nin_virtual": 2,  # #Virtual input neurons = #Pixel value bins - 1
    "Nhidden": 64,
    "Nlayer": 6,  # Number of layers
    "Nout": 2,
    "w_scale": 0.9,  # Scaling factor of initial weights
    # Trial
    "T": 2.0,
    "K": 200,  # Maximal number of simulated ordinary spikes
    "dt": 0.001,  # Step size used to compute state traces
    # Training
    "gamma": 0,
    "Nbatch":731,
    "lr": 1e-3,
    "tau_lr": 1e2,
    "beta1": 0.9,
    "beta2": 0.999,
    "p_flip": 0.0,
    "Nepochs": 0}







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
    """
    Plots the ground truth parabola y = x^2 and the learned prediction of the network.
    
    Args:
        neuron: The spiking neuron model.
        p: The network parameters (e.g. learned weights and initial phases).
        config: Configuration dictionary (must include keys like 'Nin', 'Nhidden', 'Nlayer',
                'Nout', 'T', etc. for simulation).
    """
   

    def normalize(x1,T):
      normalized_x1 = (x1 - jnp.min(x1))/(jnp.max(x1)-jnp.min(x1))
      normalized_x1=(1-normalized_x1)*T
      normalized_x1 = normalized_x1.reshape(-1,1)
      return normalized_x1


    data = sio.loadmat('experiments/pinnst/burgers_shock.mat')
    # encoded_inputs = data['encoded_inputs']
    x = data['x']
    x = x[:, 0]
    t = data['t']
    t = t[:, 0]

    n = 1 #downsample
    x_min = jnp.array([jnp.min(x)])
    x_max = jnp.array([jnp.max(x)])
    t_min = jnp.array([jnp.min(t)])
    t_max = jnp.array([jnp.max(t)])
    x_in = x[::n]
    t_in = t[::n]

    x = jnp.unique(jnp.concatenate([x_min, x_in, x_max]))
    t = jnp.unique(jnp.concatenate([t_min, t_in, t_max]))
    print(x.shape, t.shape)


    X1, X2 = jnp.meshgrid(x, t)
    X1 = X1.ravel()
    X2 = X2.ravel()
    print(X1.shape, X2.shape)
    X = jnp.column_stack((X1, X2))
    x1 = X[:, 0]
    x2 = X[:, 1]
    print(jnp.min(x1), jnp.max(x1), jnp.min(x2), jnp.max(x2))

    T = config["T"]
    normalized_x1 = normalize(x1, T)
    normalized_x2 = normalize(x2, T)
    
   
    encoded_inputs = jnp.concat((normalized_x1, normalized_x2), axis=1)


    
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

    u_ic = -jnp.sin(jnp.pi * (1-encoded_inputs[:,0]))
    
    # Boundary multiplier (vanishes at x=-1,1)
    boundary_mult = (-1-(1-encoded_inputs[:,0])) * (1-(1-encoded_inputs[:,0]))
    
    # Time evolution term
    time_mult = jnp.exp(-3.0*((0.99/2)*(2-encoded_inputs[:,1])))  # or just t
    
    # Construct solution that automatically satisfies constraints
    pred = u_ic * time_mult + ((0.99/2)*(2-encoded_inputs[:,1])) * boundary_mult * preds_raw


    sio.savemat('experiments/pinnst/preds.mat', {'preds': pred})



from spikegd.theta import ThetaNeuron
neuron = ThetaNeuron(config_theta["tau"], config_theta["I0"], config_theta["eps"])
plot_parabola(neuron, metrics_example["p_end"], config_theta)

