import jax
import jax.numpy as jnp
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from jax import random
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from deeponetr import plot_error, plot_spikes, plot_traces, run, run_example, eventffwd, outfn_b, outfn_t
from spikegd.theta import ThetaNeuron
from spikegd.utils.plotting import (
    cm2inch,
    panel_label,
)

from spikegd.lif import LIFNeuron

from spikegd.olif import OscLIFNeuron

from spikegd.qif import QIFNeuron
import scipy.io as sio

plt.style.use("spikegd.utils.plotstyle")


config_theta = {
    "seed": 0,
    # Neuron
    "tau": 6 / jnp.pi,
    "I0": 5 / 4,
    "eps": 1e-6,
    # Network
    "Nin_b":51,
    "Nin_t": 1,
    "Nin_virtual_b": 51,
    "Nin_virtual_t": 1,
    "Nhidden_b": 64,
    "Nhidden_t": 64,
    "Nlayer_b": 2,  # Number of layers
    "Nlayer_t": 2,  # Number of layers
    "Nout_b": 128,
    "Nout_t": 128,
    "w_scale": 0.9,  # Scaling factor of initial weights
    # Trial
    "T": 3.0,
    "K": 200,  # Maximal number of simulated ordinary spikes
    "dt": 0.001,  # Step size used to compute state traces
    # Training
    "gamma": 0,
    "Nbatch": 50,
    "lr": 1e-3,
    "tau_lr": 1e2,
    "beta1": 0.9,
    "beta2": 0.999,
    "p_flip": 0.0,
    "Nepochs": 1500,
    "Num_points": 51}







def run_theta(config: dict) -> dict:
    """
    Wrapper to train a network of Theta neurons with the given configuration.

    See docstring of `run` and article for more information.
    """
    tau, I0, eps = config["tau"], config["I0"], config["eps"]
    neuron_b = ThetaNeuron(tau, I0, eps)
    neuron_t = ThetaNeuron(tau, I0, eps)
    # metrics = run(neuron, config, progress_bar="notebook")
    metrics = run(neuron_b, neuron_t, config, progress_bar="script")
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
    p_b = p[0]
    p_t = p[1]
    tau, I0, eps = config["tau"], config["I0"], config["eps"]
    neuron_b = ThetaNeuron(tau, I0, eps)
    neuron_t = ThetaNeuron(tau, I0, eps)
    metrics = run_example(p_b, p_t, neuron_b, neuron_t, config)
    return metrics






example_init = run_example_theta(metrics_example["p_init"], config_theta)
example_end = run_example_theta(metrics_example["p_end"], config_theta)





# ### Figure
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


def input_normalization(xs, T):
    normalized = (xs - jnp.min(xs))/(jnp.max(xs)-jnp.min(xs))
    normalized=(1-normalized)*T
   
    return normalized

def plot_solution(neuron_b: AbstractPhaseOscNeuron, neuron_t: AbstractPhaseOscNeuron, p, config):
# def plot_parabola(neuron: AbstractPseudoIFNeuron, p, config):
    """
    Plots the ground truth parabola y = x^2 and the learned prediction of the network.
    
    Args:
        neuron: The spiking neuron model.
        p: The network parameters (e.g. learned weights and initial phases).
        config: Configuration dictionary (must include keys like 'Nin', 'Nhidden', 'Nlayer',
                'Nout', 'T', etc. for simulation).
    """
    test_data = sio.loadmat('experiments/deeponet/test_data.mat')
    branch_X_test = test_data['f_test'][:800, :]
    trunk_X_test = test_data['x_test'].reshape(-1, 1)[:800, :]
    Y_test = test_data['y_test'][:800, :]

    train_data = sio.loadmat('experiments/deeponet/train_data.mat')
    branch_X_train = train_data['f_train'][:800, :]
    trunk_X_train = train_data['x_train'].reshape(-1, 1)[:800, :]
    Y = train_data['y_train'][:800, :]
    Y_min = jnp.min(Y)
    Y_max = jnp.max(Y)
   
    T = config_theta["T"] 
   
    branch_min = min(jnp.min(branch_X_test), jnp.min(branch_X_train))
    branch_max = max(jnp.max(branch_X_test), jnp.max(branch_X_train))
    test_input_b = (branch_X_test-branch_min)/(branch_max-branch_min)
    test_input_b = (1 - test_input_b)*T
    test_input_t = input_normalization(trunk_X_test, T)
    test_input_t = jnp.repeat(test_input_t.reshape(1, test_input_t.shape[0], test_input_t.shape[1]), test_input_b.shape[0], axis=0)
    

    p_b = p[0]
    p_t = p[1]
    # Run the network simulation on each input using eventffwd.
    # vmap over the batch dimension (each input sample).
    outs_b, outs_t = vmap(eventffwd, in_axes=(None, None, None, None, 0, 0, None))(neuron_b, neuron_t, p_b, p_t, test_input_b, test_input_t, config)
    t_outs_b = vmap(outfn_b, in_axes=(None, 0, None, None))(neuron_b, outs_b, p_b, config)
    t_outs_t = vmap(outfn_t, in_axes=(None, 0, None, None))(neuron_t, outs_t, p_t, config)
  
    t_outs_t_reshape = t_outs_t.reshape(t_outs_t.shape[0], t_outs_t.shape[2], -1, 2)
    t_outs_b_reshape = t_outs_b.reshape(t_outs_b.shape[0], -1, 2)
    t_outs_t_true = t_outs_t_reshape[:, :, :, 1] - t_outs_t_reshape[:, :, :, 0]
    t_outs_b_true = t_outs_b_reshape[:, :, 1] - t_outs_b_reshape[:, :, 0]
    t_final = jnp.einsum('bo,bto->bt', t_outs_b_true, t_outs_t_true)
    preds = t_final

   
    preds = preds * (Y_max - Y_min) + Y_min
    # Compute the ground truth: y = x^2.
    ground_truth = Y_test 

    print('mse', jnp.mean((ground_truth-preds)**2))
    print('l2', jnp.mean(jnp.linalg.norm(ground_truth - preds, axis=1) / jnp.linalg.norm(ground_truth, axis=1)))
    
    sio.savemat('experiments/deeponet/deeponet.mat', {'pred':preds, 'true': ground_truth})
    # Plot the curves
    plt.figure(figsize=(6, 4))
    plt.plot(trunk_X_test.squeeze(), ground_truth[50,:], label="Ground Truth", color="blue")
    plt.plot(trunk_X_test.squeeze(), preds[50, :], label="Learned Prediction", color="red", linestyle="dotted")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Comparison of Ground Truth and Learned Function")
    plt.legend()
    plt.show()
    plt.savefig('regression.png')




from spikegd.theta import ThetaNeuron
neuron_b = ThetaNeuron(config_theta["tau"], config_theta["I0"], config_theta["eps"])
neuron_t = ThetaNeuron(config_theta["tau"], config_theta["I0"], config_theta["eps"])
plot_solution(neuron_b, neuron_t, metrics_example["p_end"], config_theta)

