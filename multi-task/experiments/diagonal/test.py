import random, math

import numpy as np

def get_parameters(c, lmda):
    # if np.any(lmda == 0):
    #     raise ValueError("λ must be nonzero.")
    if np.any(c**2 < lmda**2):
        raise ValueError("Require c² ≥ λ² for real outputs.")
    v = np.sqrt((c + lmda) / 2)
    u = np.sqrt((c - lmda) / 2)
    return v, v, u, u  # v⁺, v⁻, u⁺, u⁻


def calc_lmbda(v_plus, v_minus, u_plus, u_minus):
    return (v_plus**2 - u_plus**2), (v_minus**2 - u_minus**2)

def calc_c(v_plus, v_minus, u_plus, u_minus):
    return v_plus*v_minus + u_plus*u_minus





print('hi')