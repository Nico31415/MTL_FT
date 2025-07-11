"""
Interactive heat-maps for λ-balanced diagonal linear networks
=============================================================
Figure 1  : ℓ–order  as a function of  (β_aux  , λ)
Figure 2  : FD       as a function of  (β_aux  , λ)

Each figure has its own sliders for
    • β_main
    • γ          (sqrt initial-noise variance)
    • C_PT       (pre-training scale)

Run in a Jupyter notebook with `%matplotlib widget`
or as a regular Python script (the latter pops up GUI windows
if your matplotlib backend supports it).
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# -----------------------------------------------------------
# 1 .  FIXED GRID FOR THE TWO HEAT-MAPS
# -----------------------------------------------------------
lambda_vals   = np.linspace(-5.0, 5.0, 200)      # y-axis
beta_aux_vals = np.linspace( 0.0, 5.0, 200)      # x-axis
Λ, B_aux = np.meshgrid(lambda_vals, beta_aux_vals, indexing="ij")


# -----------------------------------------------------------
# 2 .  ANALYTIC FORMULAS  (no shortcuts hidden)
# -----------------------------------------------------------
def D(lam, beta_aux, C_pt, gamma):
    """Effective denominator D(λ, β_aux, C_PT, γ)."""
    return lam * C_pt * (1.0 + np.sqrt(1.0 + (beta_aux / C_pt) ** 2)) + gamma**2


def l_order_map(beta_main, C_pt, gamma):
    """Return ℓ-order evaluated on the global (Λ, B_aux) grid."""
    D_val   = D(Λ, B_aux, C_pt, gamma)
    y       = beta_main / D_val
    z_full  = 2.0 * y

    safe_den = 2.0 - np.sqrt(4.0 + z_full**2) + z_full * np.arcsinh(y)
    with np.errstate(divide="ignore", invalid="ignore"):
        ℓ = (2.0 * beta_main / D_val) * (np.arcsinh(y) / safe_den)
    ℓ[np.abs(D_val) < 1e-12] = np.nan   # mask the singular line
    return ℓ


def fd_map(beta_main, C_pt, gamma):
    """Return Feature-Dependence evaluated on the global grid."""
    D_val   = D(Λ, B_aux, C_pt, gamma)
    y       = beta_main / D_val
    z_full  = 2.0 * y

    safe_den   = 2.0 - np.sqrt(4.0 + z_full**2) + z_full * np.arcsinh(y)
    inner_term = 1.0 - (2.0 * beta_main / D_val) * (np.arcsinh(y) / safe_den)

    with np.errstate(divide="ignore", invalid="ignore"):
        fd = (Λ * B_aux**2) / (D_val * np.sqrt(C_pt**2 + B_aux**2)) * inner_term
    fd[np.abs(D_val) < 1e-12] = np.nan
    return fd


# -----------------------------------------------------------
# 3 .  CONVENIENCE: generic figure-builder
# -----------------------------------------------------------
def build_interactive_figure(title, data_fn):
    """
    Create an interactive figure with three sliders that control
    beta_main, gamma, and C_PT. `data_fn` is either l_order_map or fd_map.
    """
    # ---- initial slider values ----
    init_beta_main = 1.0
    init_gamma     = 0.0
    init_C_pt      = 1.0

    # ---- initial heat-map ----
    fig, ax = plt.subplots(figsize=(6, 5))
    plt.subplots_adjust(left=0.10, right=0.86, bottom=0.25)  # room for sliders
    img = ax.imshow(
        data_fn(init_beta_main, init_C_pt, init_gamma),
        origin="lower",
        extent=[beta_aux_vals.min(), beta_aux_vals.max(),
                lambda_vals.min(),  lambda_vals.max()],
        aspect="auto"
    )
    ax.set_title(title)
    ax.set_xlabel(r"$\beta_{\mathrm{aux}}$")
    ax.set_ylabel(r"$\lambda$")
    cbar = plt.colorbar(img, ax=ax, fraction=0.06, pad=0.04)
    cbar.set_label(title.split()[0])  # first word of title

    # ---- slider axes ----
    ax_beta_main = plt.axes([0.10, 0.15, 0.76, 0.03])
    ax_gamma     = plt.axes([0.10, 0.10, 0.76, 0.03])
    ax_C_pt      = plt.axes([0.10, 0.05, 0.76, 0.03])

    s_beta_main = Slider(ax_beta_main, r"$\beta_{\mathrm{main}}$", 0.1, 5.0,
                         valinit=init_beta_main, valstep=0.05)
    s_gamma     = Slider(ax_gamma,     r"$\gamma$",                0.0, 3.0,
                         valinit=init_gamma,     valstep=0.05)
    s_C_pt      = Slider(ax_C_pt,      r"$C_{\mathrm{PT}}$",       0.1, 5.0,
                         valinit=init_C_pt,      valstep=0.05)

    # ---- update callback ----
    def update(val):
        new_data = data_fn(s_beta_main.val, s_C_pt.val, s_gamma.val)
        img.set_data(new_data)
        # auto-rescale colour limits
        vmin, vmax = np.nanmin(new_data), np.nanmax(new_data)
        if np.isfinite(vmin) and np.isfinite(vmax):
            img.set_clim(vmin, vmax)
        fig.canvas.draw_idle()

    for sldr in (s_beta_main, s_gamma, s_C_pt):
        sldr.on_changed(update)

    return fig


# -----------------------------------------------------------
# 4 .  BUILD TWO INDEPENDENT FIGURES
# -----------------------------------------------------------
fig_L  = build_interactive_figure(r"$\ell$-order heat-map",      l_order_map)
fig_FD = build_interactive_figure(r"Feature-Dependence heat-map", fd_map)

plt.show()
