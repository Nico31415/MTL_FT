import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from diagonal_network3 import DiagonalNetwork, compute_c, compute_lambda

@dataclass
class PretrainConfig:
    d: int = 5
    o: int = 1
    N: int = 5
    T: int = 500
    lr: float = 1e-6
    scale: float = 0.1
    teacher_scale: float = 5.0
    seed: int = 42

@dataclass
class FinetuneConfig:
    d: int = 5
    o: int = 1
    N: int = 5
    T: int = 500
    lr: float = 1e-6
    scale: float = 0.1
    teacher_scale: float = 5.0
    seed: int = 123  # Different seed for new data/teacher

def compute_analytical_c_ft(lambda_pt, c_pt, gamma, w_tilde):
    factor = np.sqrt(1 + (w_tilde / c_pt)**2)
    return lambda_pt + c_pt + (c_pt / lambda_pt + lambda_pt) * factor
    # TODO: Fill in your analytical formula here

def compute_beta(network):
    """
    Compute the effective beta parameter from the network's weight matrix.
    beta is just w_tilde = u_plus * v_plus - u_minus * v_minus
    """
    with torch.no_grad():
        w_tilde = network.u_plus * network.v_plus[:, None] - network.u_minus * network.v_minus[:, None]
        beta = w_tilde.cpu().numpy()
    return beta

if __name__ == "__main__":
    # ----- Pretraining Phase -----
    pre_cfg = PretrainConfig()
    pre_net = DiagonalNetwork(pre_cfg)
    print("[Pretraining] Initial c, lambda:")
    print("c_pt:", compute_c(pre_net))
    print("lambda_pt:", compute_lambda(pre_net))
    pre_net.train()
    print("[Pretraining] After training c, lambda:")
    c_pt_final = compute_c(pre_net)
    lambda_pt_final = compute_lambda(pre_net)
    print("c_pt:", c_pt_final)
    print("lambda_pt:", lambda_pt_final)

    # Save pretrained parameters
    pretrained_state = {k: v.clone() for k, v in pre_net.state_dict().items()}

    # ----- Finetuning Phase -----
    ft_cfg = FinetuneConfig()
    ft_net = DiagonalNetwork(ft_cfg)
    # Optionally: load some or all pretrained parameters (simulate transfer)
    # For example, copy all parameters:
    for k in pretrained_state:
        if k in ft_net.state_dict():
            ft_net.state_dict()[k].copy_(pretrained_state[k])
    print("[Finetuning] After reinit (before training) c, lambda:")
    print("c_ft:", compute_c(ft_net))
    print("lambda_ft:", compute_lambda(ft_net))
    ft_net.train()
    print("[Finetuning] After training c, lambda:")
    c_ft_final = compute_c(ft_net)
    lambda_ft_final = compute_lambda(ft_net)
    print("c_ft:", c_ft_final)
    print("lambda_ft:", lambda_ft_final)
    
    # Compare with analytical prediction
    gamma = 1e-3  # Set your gamma value
    with torch.no_grad():
        w_tilde = ft_net.u_plus * ft_net.v_plus[:, None] - ft_net.u_minus * ft_net.v_minus[:, None]
        w_tilde = w_tilde.cpu().numpy()
    c_ft_analytical = compute_analytical_c_ft(lambda_pt_final, c_pt_final, gamma, w_tilde)
    print(f"\n[Comparison] Analytical vs Actual c_ft:")
    print(f"Analytical c_ft: {c_ft_analytical}")
    print(f"Actual c_ft: {c_ft_final}")
    print(f"Difference factor: {(c_ft_analytical / c_ft_final)}")
    print(f"w_tilde used: {w_tilde}") 