import numpy as np
import matplotlib.pyplot as plt

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

# Set up plots
beta_aux_vals = np.logspace(-4, 2, 1000)
beta_fixed = 1.0
gamma = 1e-3
initial_lambda = 0.0

# Create first figure for the curves
fig1 = plt.figure(figsize=(6, 4))
ax1 = fig1.add_subplot(111)

# Fixed MTL curve
mtl_curve = [mt_derivative(beta_fixed, b_init) for b_init in beta_aux_vals]
line_mtl, = ax1.plot(beta_aux_vals, mtl_curve, color='purple', label='MTL', linewidth=0.75)

# Initial PT+FT curve
ptft_curve = [generalized_q_norm_derivative(beta_fixed, b_init, gamma, initial_lambda)
              for b_init in beta_aux_vals]
line_ptft, = ax1.plot(beta_aux_vals, ptft_curve, color='darkgreen', label='PT+FT', linewidth=0.75)

# Axis config for first plot
ax1.set_xscale('log')
ax1.set_xlabel('Auxiliary magnitude')
ax1.set_ylabel('Order of penalty')
ax1.set_xticks([0.001, 10])
ax1.set_xticklabels(['0.001', '10'])
ax1.set_yticks([1, 2])
ax1.set_title(f'λ = {initial_lambda:.2f}')
ax1.axvline(x=1, color='grey', linestyle='--', alpha=0.5)
ax1.legend()

# Secondary y-axis for feature dependence
ax1_twin = ax1.twinx()
ax1_twin.set_ylim(ax1.get_ylim())
ax1_twin.set_yticks([1, 0, -1])
ax1_twin.set_ylabel('Feature\ndependence')

# Create second figure for the heatmap
fig2 = plt.figure(figsize=(6, 4))
ax2 = fig2.add_subplot(111)

# Create heatmap data
lambda_vals = np.linspace(-2.5, 2.5, 100)
beta_aux_vals_heatmap = np.logspace(-4, 2, 100)
X, Y = np.meshgrid(beta_aux_vals_heatmap, lambda_vals)
Z = np.zeros_like(X)

for i, lam in enumerate(lambda_vals):
    for j, beta_aux in enumerate(beta_aux_vals_heatmap):
        Z[i, j] = generalized_q_norm_derivative(beta_fixed, beta_aux, gamma, lam)

# Plot heatmap
im = ax2.pcolormesh(X, Y, Z, shading='auto', cmap='viridis')
ax2.set_xscale('log')
ax2.set_xlabel('Auxiliary magnitude')
ax2.set_ylabel('λ value')
ax2.set_title('Order of penalty heatmap')
plt.colorbar(im, ax=ax2, label='Order of penalty')

plt.show()
