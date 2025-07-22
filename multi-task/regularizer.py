import numpy as np
from scipy.optimize import minimize

# ---- Implicit bias regularizer ----
def q(z):
    return 2 - np.sqrt(4 + z**2) + z * np.arcsinh(z / 2)

def regularizer_Q(beta, c):
    sqrt_k = 2 * c
    z = 2 * beta / sqrt_k
    return np.sum((sqrt_k / 8) * q(z))

# ---- New function for constrained optimization ----
def solve_constrained_opt(X, y, c):

    """
    Note: this code only works for N = d. otherwise the system is 
    under/over-constrained so there is not a unique solution. 
    However one can check that even in these cases the resulting 
    beta from gradient flow is one of the possible solutions. 
    """
    """
    Solves the constrained optimization problem:
        min_beta regularizer_Q(beta, c)
        s.t. X^T beta = y
    X: shape (d, N)
    y: shape (N,) or (o, N) with o=1
    c: shape (d,)
    Returns (optimal_beta, regularizer_value_at_optimum)
    """
    X = np.atleast_2d(X)
    d, N = X.shape
    y = np.atleast_1d(y)
    if y.ndim > 1:
        y = y.flatten()
    def J(beta):
        return regularizer_Q(beta, c)
    def constraint(beta):
        # b = beta.reshape(-1, 1)           # (d,1)
        # return (b.T@X).flatten() - y
        return X.T@beta - y  # shape (N,)
    cons = {'type': 'eq', 'fun': constraint}
    beta0 = np.random.randn(d)
    # print('Initial beta: ', beta0)
    res = minimize(J, 
                   beta0,
                   constraints=[cons], 
                   method='SLSQP',
                   options={'ftol': 1e-12, 'maxiter': 1000})
    beta_star = res.x
    reg_val = J(beta_star)
    return beta_star, reg_val

# # ---- Set up problem ----
# D = 10
# np.random.seed(42)

# x = np.random.randn(D)             # Input vector
# y = 1.0                            # Target scalar
# c = np.abs(np.random.rand(D)) + 0.1  # Regularization coefficients, must be > 0

# # ---- Define objective and constraint ----
# def J(beta):
#     return regularizer_Q(beta, c)

# def constraint(beta):
#     return beta.T @ x - y

# cons = {'type': 'eq', 'fun': constraint}

# # ---- Initial guess ----
# beta0 = np.random.randn(D)

# # ---- Run constrained optimization ----
# res = minimize(J, beta0, constraints=[cons], method='SLSQP')

# ---- Output results ----
# beta_star = res.x
# print("Optimal beta:", beta_star)
# print("Regularizer value at optimum:", J(beta_star))
# print("Constraint satisfied (βᵀx = y):", np.isclose(beta_star @ x, y))
