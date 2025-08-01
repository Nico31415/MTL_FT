import numpy as np
import matplotlib.pyplot as plt

# q(z)
def q(z):
    return 2 - np.sqrt(4 + z**2) + z * np.arcsinh(z / 2)

# q'(z)
def q_prime(z):
    return np.arcsinh(z / 2)

# c_FT
def compute_c_ft(lambda_pt, c_pt, beta_aux, gamma):
    beta_abs = abs(beta_aux)
    term1 = (lambda_pt / c_pt) + 1
    term2 = c_pt / beta_abs
    sqrt_term = np.sqrt(term2**2 + 1)
    return beta_abs * term1 * (term2 + sqrt_term) + gamma**2

# l-order
def compute_l_order(beta_main, sqrt_k):
    z = 2 * beta_main / sqrt_k
    return (2 * beta_main / sqrt_k) * (q_prime(z) / q(z))

# FD
def compute_fd(beta_aux, beta_main, sqrt_k, k, dk_dbeta_aux):
    z = 2 * beta_main / sqrt_k
    factor = (0.5 - (beta_main / sqrt_k) * (q_prime(z) / q(z)))
    return factor * (beta_aux / k) * dk_dbeta_aux

# Main compute wrapper
def compute_l_order_and_fd(lambda_pt, c_pt, beta_aux, beta_main, gamma, epsilon=1e-6):
    c_ft = compute_c_ft(lambda_pt, c_pt, beta_aux, gamma)
    sqrt_k = 2 * c_ft
    k = sqrt_k ** 2

    # Numerical derivative of k wrt beta_aux
    k_plus = (2 * compute_c_ft(lambda_pt, c_pt, beta_aux + epsilon, gamma))**2
    k_minus = (2 * compute_c_ft(lambda_pt, c_pt, beta_aux - epsilon, gamma))**2
    dk_dbeta_aux = (k_plus - k_minus) / (2 * epsilon)

    l_order = compute_l_order(beta_main, sqrt_k)
    fd = compute_fd(beta_aux, beta_main, sqrt_k, k, dk_dbeta_aux)

    return l_order, fd

# Main execution
def main():
    # User input
    lambda_pt = float(input("Enter λ_PT: "))
    c_pt = float(input("Enter c_PT: "))
    beta_main = float(input("Enter beta_main: "))
    gamma = float(input("Enter γ: "))

    # Sweep beta_aux on log scale
    beta_aux_values = np.logspace(-3, 1, 300)
    l_orders = []
    fds = []
    combined = []

    for beta_aux in beta_aux_values:
        l, f = compute_l_order_and_fd(lambda_pt, c_pt, beta_aux, beta_main, gamma)
        l_orders.append(l)
        fds.append(f)
        combined.append(l + f)

    # Shared y-axis
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)

    # Plot l-order
    axes[0].plot(beta_aux_values, l_orders)
    axes[0].set_xscale("log")
    axes[0].set_xlabel(r"$\beta_{\mathrm{aux}}$")
    axes[0].set_ylabel("Value")
    axes[0].set_title("l-order")
    axes[0].grid(True)

    # Plot FD
    axes[1].plot(beta_aux_values, fds)
    axes[1].set_xscale("log")
    axes[1].set_xlabel(r"$\beta_{\mathrm{aux}}$")
    axes[1].set_title("Feature Dependence")
    axes[1].grid(True)

    # Plot l-order + FD
    axes[2].plot(beta_aux_values, combined)
    axes[2].set_xscale("log")
    axes[2].set_xlabel(r"$\beta_{\mathrm{aux}}$")
    axes[2].set_title("l-order + FD")
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
