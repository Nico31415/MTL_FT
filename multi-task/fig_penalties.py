import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams
import matplotlib.gridspec as gridspec

# Set style similar to R's theme_classic
plt.style.use('seaborn-v0_8-whitegrid')
rcParams['font.size'] = 7
rcParams['axes.labelsize'] = 7
rcParams['xtick.labelsize'] = 7
rcParams['ytick.labelsize'] = 7
rcParams['legend.fontsize'] = 7
rcParams['figure.titlesize'] = 8

# Calculate dimensions to make subplots square
# We want 6 square subplots plus a small space for legend
# Each square subplot should be 3cm x 3cm
subplot_size_cm = 3
width_cm = subplot_size_cm * 6.2  # 6 squares plus 0.2 for legend
height_cm = subplot_size_cm  # Height of one square

# Convert to inches
width_inches = width_cm / 2.54
height_inches = height_cm / 2.54

# Define the functions from R
def Q(z):
    return 2 - np.sqrt(4 + z**2) + z * np.arcsinh(z/2)

def q_norm(beta, beta_init, gamma):
    return (np.abs(beta_init) + gamma**2) * Q(2*beta/(np.abs(beta_init) + gamma**2))

def mt_norm(beta, beta_init):
    return np.sqrt(beta**2 + beta_init**2) - np.abs(beta_init)

# Create figure with subplots and extra space for legend
fig = plt.figure(figsize=(width_inches, height_inches))
# Create a gridspec with extra space on the right for the legend
gs = gridspec.GridSpec(1, 7, width_ratios=[1, 1, 1, 1, 1, 1, 0.2], wspace=0.3)

# Plot A: mt_norm
ax1 = plt.subplot(gs[0])
beta_init_vals = [1e-3, 1, 1e3]
beta_vals = np.logspace(-2, 2, 1000)

for beta_init in beta_init_vals:
    y_vals = mt_norm(beta_vals, beta_init)
    ax1.plot(beta_vals, y_vals, label=f'{beta_init:.0e}')

ax1.axvline(x=1, color='grey', linestyle='--', alpha=0.5)
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_xlabel('Magnitude')
ax1.set_ylabel('Norm')
ax1.set_xticks([0.1, 10])
ax1.set_xticklabels(['0.1', '10'])
ax1.set_yticks([1e-5, 10])
ax1.set_yticklabels(['$10^{-5}$', '10'])
ax1.text(-0.2, 1.1, 'a', transform=ax1.transAxes, fontweight='bold')

# Plot B: q_norm
ax2 = plt.subplot(gs[1])
for beta_init in beta_init_vals:
    y_vals = q_norm(beta_vals, beta_init, 1e-3)
    ax2.plot(beta_vals, y_vals, label=f'{beta_init:.0e}')

ax2.axvline(x=1, color='grey', linestyle='--', alpha=0.5)
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlabel('Magnitude')
ax2.set_ylabel('Norm')
ax2.set_xticks([0.1, 10])
ax2.set_xticklabels(['0.1', '10'])
ax2.set_yticks([1e-5, 10])
ax2.set_yticklabels(['$10^{-5}$', '10'])
ax2.text(-0.2, 1.1, 'b', transform=ax2.transAxes, fontweight='bold')

# Plot C: Correlation plots
theta_vals = [1.0, 0.99, 0.9, 0.0]
beta_aux_vals = [0.001, 1, 1000]

# Create subplots for each correlation value
for i, theta in enumerate(theta_vals):
    ax = plt.subplot(gs[i+2])
    
    for beta_aux in beta_aux_vals:
        norms = []
        for beta in beta_vals:
            v_0 = 0
            m_0 = np.sqrt(beta_aux)
            m = None
            for root in np.roots([1, -m_0*theta, 0, v_0*beta, -beta**2]):
                if np.abs(np.imag(root)) < 1e-6 and np.real(root) > 0:
                    m = np.real(root)
            norm = (beta/m-v_0)**2 + m**2 + m_0**2 - 2*m*m_0*theta
            norms.append(norm)
        
        ax.plot(beta_vals, norms, label=f'{beta_aux:.0e}' if i == 0 else "")
    
    ax.axvline(x=1, color='grey', linestyle='--', alpha=0.5)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Magnitude')
    ax.set_ylabel('Norm' if i == 0 else '')
    ax.set_xticks([0.1, 10])
    ax.set_xticklabels(['0.1', '10'])
    ax.set_yticks([1e-5, 10])
    ax.set_yticklabels(['$10^{-5}$', '10'])
    ax.set_ylim(1e-6, None)
    ax.text(-0.2, 1.1, f'c{i+1}', transform=ax.transAxes, fontweight='bold')
    ax.text(0.02, 0.95, f'Corr.: {theta}', transform=ax.transAxes, ha='left', va='top')

# Add legend in the last subplot
ax_legend = plt.subplot(gs[6])
ax_legend.axis('off')  # Hide the axes
handles, labels = ax1.get_legend_handles_labels()
ax_legend.legend(handles, labels, title='Auxiliary\nmagnitude', 
                loc='center left', bbox_to_anchor=(0, 0.5))

plt.savefig('figures/fig-penalties.pdf', bbox_inches='tight', dpi=300)
plt.close() 