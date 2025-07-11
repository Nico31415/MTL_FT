import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from diagonal_network3 import DiagonalNetwork, compute_c, compute_lambda, predict_parameters
import matplotlib.pyplot as plt

@dataclass
class PretrainConfig:
    d: int = 1
    o: int = 1
    N: int = 5
    T: int = 1000
    lr: float = 1e-3
    scale: float = 0.1
    teacher_scale: float = 5.0
    seed: int = 42

@dataclass
class FinetuneConfig:
    d: int = 1
    o: int = 1
    N: int = 5
    T: int = 1000
    lr: float = 1e-3
    scale: float = 0.1
    teacher_scale: float = 5.0
    seed: int = 123  # Different seed for new data/teacher

gamma = 5  # Set your gamma value

def compute_analytical_c_ft(lambda_pt, c_pt, gamma, beta_main):
    factor = 1 + np.sqrt(1 + (beta_main / c_pt)**2)

    return (lambda_pt + c_pt) * factor + gamma**2
    # return (1+lambda_pt)*factor + lambda_pt + c_pt / lambda_pt + c_pt + gamma**2
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

    # Print beta_main (learned beta after pretraining)
    with torch.no_grad():
        beta_main = pre_net.u_plus * pre_net.v_plus[:, None] - pre_net.u_minus * pre_net.v_minus[:, None]
        beta_main = beta_main.cpu().numpy()
    print("beta_main (learned beta after pretraining):", beta_main)

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
    # After copying pretrained parameters, reinitialize
    print("\n[Before Reinitialization]")
    print("u_plus:", ft_net.u_plus.detach().cpu().numpy())
    print("u_minus:", ft_net.u_minus.detach().cpu().numpy())
    print("v_plus:", ft_net.v_plus.detach().cpu().numpy())
    print("v_minus:", ft_net.v_minus.detach().cpu().numpy())

    with torch.no_grad():
        # Get the sum of pretrained v_plus and v_minus
        v_sum = pre_net.v_plus + pre_net.v_minus
        
        # Initialize u_plus and u_minus to gamma
        ft_net.u_plus.data = torch.full_like(ft_net.u_plus, gamma)
        ft_net.u_minus.data = torch.full_like(ft_net.u_minus, gamma)
        
        # Initialize v_plus and v_minus to the sum of pretrained values
        ft_net.v_plus.data = v_sum.clone()
        ft_net.v_minus.data = v_sum.clone()
        
        # The invariance constraint is automatically satisfied with the initialization.
        # No need to recompute u_minus.

    print("\n[After Reinitialization]")
    print("u_plus:", ft_net.u_plus.detach().cpu().numpy())
    print("u_minus:", ft_net.u_minus.detach().cpu().numpy())
    print("v_plus:", ft_net.v_plus.detach().cpu().numpy())
    print("v_minus:", ft_net.v_minus.detach().cpu().numpy())
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
    with torch.no_grad():
        w_tilde = ft_net.u_plus * ft_net.v_plus[:, None] - ft_net.u_minus * ft_net.v_minus[:, None]
        w_tilde = w_tilde.cpu().numpy()
    c_ft_analytical = compute_analytical_c_ft(lambda_pt_final, c_pt_final, gamma, w_tilde)
    print(f"\n[Comparison] Analytical vs Actual c_ft:")
    print(f"Analytical c_ft: {c_ft_analytical}")
    print(f"Actual c_ft: {c_ft_final}")
    print(f"Difference factor: {(c_ft_analytical / c_ft_final)}")
    print(f"w_tilde used: {w_tilde}")

    # --- Analytical parameter prediction after finetuning ---
    lambda_pred = compute_lambda(ft_net)
    c_pred = compute_c(ft_net)
    beta_pred = w_tilde
    v_plus_pred, u_plus_pred, v_minus_pred, u_minus_pred = predict_parameters(lambda_pred, c_pred, beta_pred)
    print("\n[Analytical Prediction after Finetuning]")
    print("u_plus_pred =", u_plus_pred)
    print("u_minus_pred =", u_minus_pred)
    print("v_plus_pred =", v_plus_pred)
    print("v_minus_pred =", v_minus_pred)

    with torch.no_grad():
        beta_aux = ft_net.u_plus * ft_net.v_plus[:, None] - ft_net.u_minus * ft_net.v_minus[:, None]
        beta_aux = beta_aux.cpu().numpy()
    print("beta_aux (learned beta after finetuning):", beta_aux)

    # Continual learning loss plot
    loss_pre = pre_net.loss_history
    loss_ft = ft_net.loss_history
    loss_all = np.concatenate([loss_pre, loss_ft])

    plt.figure(figsize=(8, 4))
    plt.plot(np.arange(len(loss_pre)), loss_pre, label='Pretraining loss (pretrain data)')
    plt.plot(np.arange(len(loss_pre), len(loss_pre) + len(loss_ft)), loss_ft, label='Finetuning loss (finetune data)')
    plt.axvline(len(loss_pre), color='red', linestyle='--', label='Start Finetuning')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Continual Learning Loss Curve')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Plot the evolution of all 4 parameters through pretraining and finetuning
    # Get histories for all parameters
    u_plus_pre = np.array(pre_net.u_plus_history)      # shape (T_pre+1, d, o)
    u_plus_ft = np.array(ft_net.u_plus_history)        # shape (T_ft+1, d, o)
    u_plus_all = np.concatenate([u_plus_pre, u_plus_ft], axis=0)

    u_minus_pre = np.array(pre_net.u_minus_history)    # shape (T_pre+1, d, o)
    u_minus_ft = np.array(ft_net.u_minus_history)      # shape (T_ft+1, d, o)
    u_minus_all = np.concatenate([u_minus_pre, u_minus_ft], axis=0)

    v_plus_pre = np.array(pre_net.v_plus_history)      # shape (T_pre+1, d)
    v_plus_ft = np.array(ft_net.v_plus_history)        # shape (T_ft+1, d)
    v_plus_all = np.concatenate([v_plus_pre, v_plus_ft], axis=0)

    v_minus_pre = np.array(pre_net.v_minus_history)    # shape (T_pre+1, d)
    v_minus_ft = np.array(ft_net.v_minus_history)      # shape (T_ft+1, d)
    v_minus_all = np.concatenate([v_minus_pre, v_minus_ft], axis=0)

    # Create 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Parameter Evolution (Pretrain + Finetune)', fontsize=16)

    # Plot u_plus
    axes[0, 0].set_title('u_plus')
    for i in range(u_plus_all.shape[1]):
        for j in range(u_plus_all.shape[2]):
            axes[0, 0].plot(u_plus_all[:, i, j], label=f'u_plus[{i},{j}]')
    axes[0, 0].axvline(len(u_plus_pre), color='red', linestyle='--', label='Start Finetuning')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('u_plus')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Plot u_minus
    axes[0, 1].set_title('u_minus')
    for i in range(u_minus_all.shape[1]):
        for j in range(u_minus_all.shape[2]):
            axes[0, 1].plot(u_minus_all[:, i, j], label=f'u_minus[{i},{j}]')
    axes[0, 1].axvline(len(u_minus_pre), color='red', linestyle='--', label='Start Finetuning')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('u_minus')
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Plot v_plus
    axes[1, 0].set_title('v_plus')
    for i in range(v_plus_all.shape[1]):
        axes[1, 0].plot(v_plus_all[:, i], label=f'v_plus[{i}]')
    axes[1, 0].axvline(len(v_plus_pre), color='red', linestyle='--', label='Start Finetuning')
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('v_plus')
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Plot v_minus
    axes[1, 1].set_title('v_minus')
    for i in range(v_minus_all.shape[1]):
        axes[1, 1].plot(v_minus_all[:, i], label=f'v_minus[{i}]')
    axes[1, 1].axvline(len(v_minus_pre), color='red', linestyle='--', label='Start Finetuning')
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].set_ylabel('v_minus')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.show() 