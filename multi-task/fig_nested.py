import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Define q and Q-norm functions
def Q(z):
    return 2 - np.sqrt(4 + z**2) + z * np.arcsinh(z/2)

def Q_prime(z):
    return -1/np.sqrt(4 + z**2) * 2*z + np.arcsinh(z/2) + z/np.sqrt(1 + (z/2)**2)/2

def generalized_q_norm(beta, beta_aux, gamma, lam):
    root_term = np.sqrt(lam**2 + 4 * beta_aux**2)
    scale = gamma**2 - lam + root_term
    arg = (4 * beta) / (2 * root_term - lam + gamma**2)
    return scale * Q(arg)

def generalized_q_norm_derivative(beta, beta_aux, gamma, lam):
    eps = 1e-4
    norm1 = generalized_q_norm(np.exp(np.log(beta) + eps), beta_aux, gamma, lam)
    norm2 = generalized_q_norm(beta, beta_aux, gamma, lam)
    return (np.log(norm1) - np.log(norm2)) / eps

# Define MTL derivative
def mt_norm(beta, beta_aux):
    return np.sqrt(beta**2 + beta_aux**2) - np.abs(beta_aux)

def mt_derivative(beta, beta_aux):
    eps = 1e-4
    norm1 = mt_norm(np.exp(np.log(beta) + eps), beta_aux)
    norm2 = mt_norm(beta, beta_aux)
    return (np.log(norm1) - np.log(norm2)) / eps

# Set up plot
beta_aux_vals = np.logspace(-4, 2, 1000)
beta_fixed = 1.0
gamma = 1e-3

fig, ax = plt.subplots(figsize=(6, 4))
plt.subplots_adjust(bottom=0.25)

# Fixed MTL curve
mtl_curve = [mt_derivative(beta_fixed, b_init) for b_init in beta_aux_vals]
line_mtl, = ax.plot(beta_aux_vals, mtl_curve, color='purple', label='MTL', linewidth=0.75)

# Initial PT+FT curve
initial_lambda = 0.0
ptft_curve = [generalized_q_norm_derivative(beta_fixed, b_init, gamma, initial_lambda)
              for b_init in beta_aux_vals]
line_ptft, = ax.plot(beta_aux_vals, ptft_curve, color='darkgreen', label='PT+FT', linewidth=0.75)

# Axis config
ax.set_xscale('log')
ax.set_xlabel('Auxiliary magnitude')
ax.set_ylabel('Order of penalty')
ax.set_xticks([0.001, 10])
ax.set_xticklabels(['0.001', '10'])
ax.set_yticks([1, 2])
ax.set_title(f'λ = {initial_lambda:.2f}')
ax.axvline(x=1, color='grey', linestyle='--', alpha=0.5)
ax.legend()

# Secondary y-axis for feature dependence
ax2 = ax.twinx()
ax2.set_ylim(ax.get_ylim())
ax2.set_yticks([1, 0, -1])
ax2.set_ylabel('Feature\ndependence')

# Slider
ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
lambda_slider = Slider(ax_slider, 'λ', -10.0, 10.0, valinit=initial_lambda, valstep=0.1)

def update(val):
    lam = lambda_slider.val
    new_ptft = [generalized_q_norm_derivative(beta_fixed, b_init, gamma, lam)
                for b_init in beta_aux_vals]
    line_ptft.set_ydata(new_ptft)
    ax.set_title(f'λ = {lam:.2f}')
    fig.canvas.draw_idle()

lambda_slider.on_changed(update)

plt.show()
