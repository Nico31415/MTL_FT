import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Define q and Q-norm functions
def q(z):
    return 2 - np.sqrt(4 + z**2) + z * np.arcsinh(z / 2)

def generalized_q_norm(beta_main, beta_aux, gamma, lam):
    root_term = np.sqrt(lam**2 + 4 * beta_aux**2)
    scale = gamma**2 - lam + root_term
    argument = (4 * beta_main) / (2 * root_term - lam + gamma**2)
    return scale * q(argument)

# Constants
beta_main_vals = np.logspace(-2, 2, 500)
beta_aux_vals = [0.001, 1, 1000]
gamma = 1e-3
colors = ['#E69F00', '#D55E00', '#CC79A7']

# Create figure and axis
fig, ax = plt.subplots(figsize=(6, 4))
plt.subplots_adjust(bottom=0.25)

# Initial lambda
initial_lambda = 0.0
lines = []

# Initial plot
for i, beta_aux in enumerate(beta_aux_vals):
    norm_vals = generalized_q_norm(beta_main_vals, beta_aux, gamma, initial_lambda)
    line, = ax.plot(beta_main_vals, norm_vals, label=f'{beta_aux:.0e}', color=colors[i])
    lines.append(line)

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('Magnitude')
ax.set_ylabel('Norm')
ax.set_xticks([0.1, 10])
ax.set_xticklabels(['0.1', '10'])
ax.set_yticks([1e-5, 10])
ax.set_yticklabels(['$10^{-5}$', '10'])
ax.set_title(f'λ = {initial_lambda:.2f}')
ax.axvline(x=1, color='grey', linestyle='--', alpha=0.5)
ax.legend(title='Auxiliary\ncoefficient')
ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.6)

# Slider axis
ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
lambda_slider = Slider(ax_slider, 'λ', -10.0, 10.0, valinit=initial_lambda, valstep=0.1)

# Update function
def update(val):
    lam = lambda_slider.val
    for i, beta_aux in enumerate(beta_aux_vals):
        new_vals = generalized_q_norm(beta_main_vals, beta_aux, gamma, lam)
        lines[i].set_ydata(new_vals)
    ax.set_title(f'λ = {lam:.2f}')
    fig.canvas.draw_idle()

lambda_slider.on_changed(update)

plt.show()
