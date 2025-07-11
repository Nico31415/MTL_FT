import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from diagonal_network3 import DiagonalNetwork, compute_c, compute_lambda, predict_parameters
import matplotlib.pyplot as plt

# Set precision for better numerical stability
torch.set_default_dtype(torch.float64)
np.set_printoptions(precision=15)

@dataclass
class PretrainConfig:
    d: int = 1
    o: int = 1
    N: int = 5
    T: int = int(1e5)
    lr: float = 1e-3
    scale: float = 0.1
    teacher_scale: float = 1.0
    seed: int = 42

@dataclass
class FinetuneConfig:
    d: int = 1
    o: int = 1
    N: int = 5
    T: int = int(1e5)
    lr: float = 1e-3
    scale: float = 0.1
    teacher_scale: float = 1.0
    seed: int = 123  # Different seed for new data/teacher

gamma = 1.0  # Set your gamma value

def compute_analytical_c_ft(lambda_pt, c_pt, gamma, beta_aux):
    r = lambda_pt / c_pt
    s = c_pt / beta_aux
    return beta_aux * (r + 1) * (s + np.sqrt(s**2+1)) + gamma**2

def compute_beta(network):
    """
    Compute the effective beta parameter from the network's weight matrix.
    beta is just w_tilde = u_plus * v_plus - u_minus * v_minus
    """
    with torch.no_grad():
        w_tilde = network.u_plus * network.v_plus - network.u_minus * network.v_minus
        beta = w_tilde.detach().cpu().numpy()
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

    # Save final pretrained parameters for later use
    final_u_plus_pt = pre_net.u_plus.detach().cpu().numpy().copy()
    final_u_minus_pt = pre_net.u_minus.detach().cpu().numpy().copy()
    final_v_plus_pt = pre_net.v_plus.detach().cpu().numpy().copy()
    final_v_minus_pt = pre_net.v_minus.detach().cpu().numpy().copy()
    
    print("\n[Final Pretrained Parameters Saved]")

    # Print beta_aux (learned beta after pretraining)
    with torch.no_grad():
        beta_aux = pre_net.u_plus * pre_net.v_plus - pre_net.u_minus * pre_net.v_minus
        beta_aux = beta_aux.detach().cpu().numpy()
    print("beta_aux (learned beta after pretraining):", beta_aux)

    # Check predict_parameters function after pretraining
    lambda_pt_pred = compute_lambda(pre_net)
    c_pt_pred = compute_c(pre_net)
    beta_pt_pred = beta_aux
    v_plus_pt_pred, u_plus_pt_pred, v_minus_pt_pred, u_minus_pt_pred = predict_parameters(lambda_pt_pred, c_pt_pred, beta_pt_pred)
    
    # Compare with actual pretrained parameters
    # print("\n[Pretraining Parameter Comparison]")
    # print("u_plus - Analytical:", u_plus_pt_pred)
    # print("u_plus - Empirical:", pre_net.u_plus.detach().cpu().numpy())
    # print("u_minus - Analytical:", u_minus_pt_pred)
    # print("u_minus - Empirical:", pre_net.u_minus.detach().cpu().numpy())
    # print("v_plus - Analytical:", v_plus_pt_pred)
    # print("v_plus - Empirical:", pre_net.v_plus.detach().cpu().numpy())
    # print("v_minus - Analytical:", v_minus_pt_pred)
    # print("v_minus - Empirical:", pre_net.v_minus.detach().cpu().numpy())

    # Save pretrained parameters
    pretrained_state = {k: v.clone().detach() for k, v in pre_net.state_dict().items()}

    # ----- Finetuning Phase -----
    ft_cfg = FinetuneConfig()
    ft_net = DiagonalNetwork(ft_cfg)
    # Optionally: load some or all pretrained parameters (simulate transfer)
    # For example, copy all parameters:
    for k in pretrained_state:
        if k in ft_net.state_dict():
            ft_net.state_dict()[k].copy_(pretrained_state[k])
    # After copying pretrained parameters, reinitialize
    # print("\n[Before Reinitialization]")
    # print("u_plus:", ft_net.u_plus.detach().cpu().numpy())
    # print("u_minus:", ft_net.u_minus.detach().cpu().numpy())
    # print("v_plus:", ft_net.v_plus.detach().cpu().numpy())
    # print("v_minus:", ft_net.v_minus.detach().cpu().numpy())

    with torch.no_grad():
        # Get the sum of pretrained v_plus and v_minus
        v_sum = pre_net.v_plus + pre_net.v_minus
        
        # Initialize u_plus and u_minus to gamma
        ft_net.u_plus.data = torch.full_like(ft_net.u_plus, gamma, dtype=torch.float64)
        ft_net.u_minus.data = torch.full_like(ft_net.u_minus, gamma, dtype=torch.float64)
        
        # Initialize v_plus and v_minus to the sum of pretrained values
        ft_net.v_plus.data = v_sum.clone().detach()
        ft_net.v_minus.data = v_sum.clone().detach()
        
        # The invariance constraint is automatically satisfied with the initialization.
        # No need to recompute u_minus.

    # print("\n[After Reinitialization]")
    # print("u_plus:", ft_net.u_plus.detach().cpu().numpy())
    # print("u_minus:", ft_net.u_minus.detach().cpu().numpy())
    # print("v_plus:", ft_net.v_plus.detach().cpu().numpy())
    # print("v_minus:", ft_net.v_minus.detach().cpu().numpy())
    # print("[Finetuning] After reinit (before training) c, lambda:")
    # print("c_ft:", compute_c(ft_net))
    # print("lambda_ft:", compute_lambda(ft_net))
    
    # Save c_ft at the beginning (after reinitialization)
    c_ft_beginning = compute_c(ft_net)
    
    # Initialize array to store c_ft values during training
    c_ft_history = []
    
    # Save reinitialized parameters before finetuning
    reinit_u_plus_ft = ft_net.u_plus.detach().cpu().numpy().copy()
    reinit_u_minus_ft = ft_net.u_minus.detach().cpu().numpy().copy()
    reinit_v_plus_ft = ft_net.v_plus.detach().cpu().numpy().copy()
    reinit_v_minus_ft = ft_net.v_minus.detach().cpu().numpy().copy()
    
    print("\n[Reinitialized Parameters Saved (Before Finetuning)]")
    
    ft_net.train()
    print("[Finetuning] After training c, lambda:")
    c_ft_final = compute_c(ft_net)
    lambda_ft_final = compute_lambda(ft_net)
    print("c_ft:", c_ft_final)
    print("lambda_ft:", lambda_ft_final)
    
    # Save c_ft at the end (after finetuning)
    c_ft_end = compute_c(ft_net)
    
    print("[Finetuning] After training c, lambda:")
    c_ft_final = compute_c(ft_net)
    lambda_ft_final = compute_lambda(ft_net)
    print("c_ft:", c_ft_final)
    print("lambda_ft:", lambda_ft_final)
    
    # Save c_ft at the end (after finetuning)
    c_ft_end = compute_c(ft_net)
    
    print(f"\n[c_ft Evolution Summary]")
    print(f"c_ft_beginning: {c_ft_beginning}")
    print(f"c_ft_end: {c_ft_end}")
    print(f"c_ft_history length: {len(c_ft_history)}")

    # Compare with analytical prediction
    with torch.no_grad():
        w_tilde = ft_net.u_plus * ft_net.v_plus - ft_net.u_minus * ft_net.v_minus
        w_tilde = w_tilde.detach().cpu().numpy()
    c_ft_analytical = compute_analytical_c_ft(lambda_pt_final, c_pt_final, gamma, beta_aux)
    print(f"\n[Comparison] Analytical vs Actual c_ft:")
    print(f"Analytical c_ft: {c_ft_analytical}")
    print(f"Actual c_ft: {c_ft_final}")
    print(f"Difference factor: {(c_ft_analytical / c_ft_final)}")
    print(f"w_tilde used: {beta_aux}")

    # Define beta_main (learned beta after finetuning)
    with torch.no_grad():
        beta_main = ft_net.u_plus * ft_net.v_plus - ft_net.u_minus * ft_net.v_minus
        beta_main = beta_main.detach().cpu().numpy()
    print("beta_main (learned beta after finetuning):", beta_main)

    # Check predict_parameters function after finetuning
    lambda_ft_pred = compute_lambda(ft_net)
    c_ft_pred = compute_c(ft_net)
    beta_ft_pred = beta_main
    v_plus_ft_pred, u_plus_ft_pred, v_minus_ft_pred, u_minus_ft_pred = predict_parameters(lambda_ft_pred, c_ft_pred, beta_ft_pred)
    
    # Compare with actual finetuned parameters
    # print("\n[Finetuning Parameter Comparison]")
    # print("u_plus - Analytical:", u_plus_ft_pred)
    # print("u_plus - Empirical:", ft_net.u_plus.detach().cpu().numpy())
    # print("u_minus - Analytical:", u_minus_ft_pred)
    # print("u_minus - Empirical:", ft_net.u_minus.detach().cpu().numpy())
    # print("v_plus - Analytical:", v_plus_ft_pred)
    # print("v_plus - Empirical:", ft_net.v_plus.detach().cpu().numpy())
    # print("v_minus - Analytical:", v_minus_ft_pred)
    # print("v_minus - Empirical:", ft_net.v_minus.detach().cpu().numpy())

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
        beta_main = ft_net.u_plus * ft_net.v_plus - ft_net.u_minus * ft_net.v_minus
        beta_main = beta_main.cpu().numpy()
    print("beta_main (learned beta after finetuning):", beta_main)

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

    # Calculate global min and max for consistent scaling
    all_data = [
        u_plus_all.flatten(),
        u_minus_all.flatten(), 
        v_plus_all.flatten(),
        v_minus_all.flatten()
    ]
    global_min = min(np.min(data) for data in all_data)
    global_max = max(np.max(data) for data in all_data)
    
    # Add some padding to the range
    y_range = global_max - global_min
    y_min = global_min - 0.1 * y_range
    y_max = global_max + 0.1 * y_range

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
    axes[0, 0].set_ylim(y_min, y_max)
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
    axes[0, 1].set_ylim(y_min, y_max)
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Plot v_plus
    axes[1, 0].set_title('v_plus')
    for i in range(v_plus_all.shape[1]):
        axes[1, 0].plot(v_plus_all[:, i], label=f'v_plus[{i}]')
    axes[1, 0].axvline(len(v_plus_pre), color='red', linestyle='--', label='Start Finetuning')
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('v_plus')
    axes[1, 0].set_ylim(y_min, y_max)
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Plot v_minus
    axes[1, 1].set_title('v_minus')
    for i in range(v_minus_all.shape[1]):
        axes[1, 1].plot(v_minus_all[:, i], label=f'v_minus[{i}]')
    axes[1, 1].axvline(len(v_minus_pre), color='red', linestyle='--', label='Start Finetuning')
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].set_ylabel('v_minus')
    axes[1, 1].set_ylim(y_min, y_max)
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.show() 