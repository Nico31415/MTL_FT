import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Tuple

@dataclass
class NetworkConfig:
    d: int = 10  # input dimension
    o: int = 1   # output dimension
    N: int = 5   # number of samples
    T: int = 100000
    lr: float = 1e-9
    scale: float = 0.1
    teacher_scale: float = 5.0
    seed: int = 42

class DiagonalNetwork(nn.Module):
    def __init__(self, config: NetworkConfig):
        super().__init__()
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)

        self.config = config
        self._init_data()
        self._init_parameters()
        self._init_histories()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=config.lr)

    def _init_data(self):
        # Teacher network parameters
        self.teacher_u_plus  = np.random.randn(self.config.d, self.config.o) * self.config.teacher_scale
        self.teacher_u_minus = np.random.randn(self.config.d, self.config.o) * self.config.teacher_scale
        self.teacher_v_plus  = np.random.randn(self.config.d) * self.config.teacher_scale
        self.teacher_v_minus = np.random.randn(self.config.d) * self.config.teacher_scale

        # Compute true beta matrix (d x o)
        self.true_beta = (self.teacher_u_plus * self.teacher_v_plus[:, None]
                         - self.teacher_u_minus * self.teacher_v_minus[:, None])

        # Generate data
        self.X = np.random.randn(self.config.d, self.config.N)
        self.y = (self.true_beta.T @ self.X)

        self.X_torch = torch.tensor(self.X, dtype=torch.float32)
        self.y_torch = torch.tensor(self.y, dtype=torch.float32)

    def _init_parameters(self):
        d, o = self.config.d, self.config.o
        scale = self.config.scale

        # Initialize u_plus, v_plus, v_minus to random nonnegative values
        self.u_plus  = nn.Parameter(torch.abs(torch.randn(d, o)) * scale)
        self.v_plus  = nn.Parameter(torch.abs(torch.randn(d, 1)) * scale)
        self.u_minus = nn.Parameter(torch.abs(torch.randn(d, o)) * scale + torch.sqrt(torch.abs(self.v_plus**2 - self.u_plus**2))) 
        self.v_minus = nn.Parameter(torch.sqrt(self.u_minus**2 + self.v_plus**2 - self.u_plus**2).clone())
    

        # Optional: Print check for invariance
        with torch.no_grad():
            v_plus_np = self.v_plus.detach().cpu().numpy()
            u_plus_np = self.u_plus.detach().cpu().numpy()
            v_minus_np = self.v_minus.detach().cpu().numpy()
            u_minus_np = self.u_minus.detach().cpu().numpy()
            diff = (v_plus_np**2 - u_plus_np**2) - (v_minus_np**2 - u_minus_np**2)
            print('Max abs difference in invariants at init:', np.max(np.abs(diff)))

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Effective weight matrix W = u⁺∘v⁺ - u⁻∘v⁻
        # w_tilde = (self.u_plus * self.v_plus[:, None]
        #            - self.u_minus * self.v_minus[:, None])
        w_tilde = self.u_plus * self.v_plus - self.u_minus * self.v_minus
        preds = w_tilde.T @ self.X_torch
        error = preds - self.y_torch
        return w_tilde, preds, error

    def compute_loss(self, error: torch.Tensor) -> torch.Tensor:
        return 0.5 * (error ** 2).sum() / self.config.N

    def compute_analytical_gradients(self) -> Tuple[np.ndarray, ...]:
        # Extract numpy arrays
        u_plus  = self.u_plus.detach().cpu().numpy()      # (d, o)
        u_minus = self.u_minus.detach().cpu().numpy()     # (d, o)
        v_plus  = self.v_plus.detach().cpu().numpy()      # (d,)
        v_minus = self.v_minus.detach().cpu().numpy()     # (d,)
        X       = self.X                                # (d, N)
        Y       = self.y                                # (o, N)
        N       = self.config.N                         # samples
        lr      = self.config.lr

        # Build w_tilde and compute core gradient G
        # w_tilde = u_plus * v_plus[:, None] - u_minus * v_minus[:, None]
        w_tilde = self.u_plus * self.v_plus - self.u_minus * self.v_minus
        error   = (w_tilde.T).detach().numpy() @ X - Y                     # (o, N)
        G       = (X @ error.T) / N                    # (d, o)

        # Parameter-wise gradients
        du_plus  =  G * v_plus                 # dL/dU⁺
        du_minus = -G * v_minus               # dL/dU⁻
        dv_plus  =  np.sum(u_plus  * G, axis=1)         # dL/dv⁺
        dv_minus = -np.sum(u_minus * G, axis=1)         # dL/dv⁻

        # Velocities under gradient descent
        du_plus_dt  = -lr * du_plus
        du_minus_dt = -lr * du_minus
        dv_plus_dt  = -lr * dv_plus
        dv_minus_dt = -lr * dv_minus

        # Induced change in w_tilde
        dw_tilde_dt = (
            du_plus_dt * v_plus
            + u_plus * dv_plus_dt
            - du_minus_dt * v_minus
            - u_minus * dv_minus_dt
        )



        # Predicted loss decay
        dL_dt = np.sum(G * dw_tilde_dt)

        # Compute analytic invariants derivatives
        dC_dt = np.sum(du_plus_dt * u_minus + u_plus * du_minus_dt, axis=1) \
               + dv_plus_dt * v_minus + v_plus * dv_minus_dt
        ddelta_plus_dt = 2 * v_plus * dv_plus_dt - 2 * np.sum(u_plus * du_plus_dt, axis=1)
        ddelta_minus_dt = 2 * v_minus * dv_minus_dt - 2 * np.sum(u_minus * du_minus_dt, axis=1)

        return du_plus, du_minus, dv_plus, dv_minus, dw_tilde_dt, dL_dt, dC_dt, ddelta_plus_dt, ddelta_minus_dt

    def _init_histories(self):
        self.loss_history           = []
        self.w_tilde_history       = []
        self.u_plus_history        = []
        self.u_minus_history       = []
        self.v_plus_history        = []
        self.v_minus_history       = []

        self.du_plus_analytical    = []
        self.du_minus_analytical   = []
        self.dv_plus_analytical    = []
        self.dv_minus_analytical   = []
        self.dw_tilde_analytical   = []
        self.dL_dt_analytical      = []

        self.dC_dt_analytical         = []
        self.ddelta_plus_dt_analytical  = []
        self.ddelta_minus_dt_analytical = []

        self.c_history             = []
        self.delta_plus_history    = []
        self.delta_minus_history   = []

    def train(self):
        for t in range(self.config.T):
            self.optimizer.zero_grad()
            w_tilde, preds, error = self.forward()
            loss = self.compute_loss(error)

            # Record current state
            self.loss_history.append(loss.item())
            self.w_tilde_history.append(w_tilde.detach().cpu().numpy().copy())
            self.u_plus_history.append(self.u_plus.detach().cpu().numpy().copy())
            self.u_minus_history.append(self.u_minus.detach().cpu().numpy().copy())
            self.v_plus_history.append(self.v_plus.detach().cpu().numpy().copy())
            self.v_minus_history.append(self.v_minus.detach().cpu().numpy().copy())

            # Empirical invariants
            with torch.no_grad():
                c = (torch.sum(self.u_plus * self.u_minus, dim=1)
                     + self.v_plus * self.v_minus).cpu().numpy()
                delta_plus  = (self.v_plus**2 - torch.sum(self.u_plus**2, dim=1)).cpu().numpy()
                delta_minus = (self.v_minus**2 - torch.sum(self.u_minus**2, dim=1)).cpu().numpy()
                self.c_history.append(c.copy())
                self.delta_plus_history.append(delta_plus.copy())
                self.delta_minus_history.append(delta_minus.copy())

            # Compute analytic gradients and invariants derivatives before updating
            du_p, du_m, dv_p, dv_m, dw_dt, dL_dt, dC_dt, ddp_dt, ddm_dt = self.compute_analytical_gradients()
            self.du_plus_analytical.append(du_p.copy())
            self.du_minus_analytical.append(du_m.copy())
            self.dv_plus_analytical.append(dv_p.copy())
            self.dv_minus_analytical.append(dv_m.copy())
            self.dw_tilde_analytical.append(dw_dt.copy())
            self.dL_dt_analytical.append(dL_dt)

            self.dC_dt_analytical.append(dC_dt.copy())
            self.ddelta_plus_dt_analytical.append(ddp_dt.copy())
            self.ddelta_minus_dt_analytical.append(ddm_dt.copy())

            # Backprop and step
            loss.backward()
            self.optimizer.step()

    def plot_training_results(self):
        # Print basic training statistics instead of plotting
        # print(f"Training completed in {self.config.T} steps")
        # print(f"Final loss: {self.loss_history[-1]:.6f}")
        # print(f"Initial loss: {self.loss_history[0]:.6f}")
        # print(f"Loss reduction: {self.loss_history[0] - self.loss_history[-1]:.6f}")
        
        # Print final parameter values
        # print(f"\nFinal parameters:")
        # print(f"u_plus: {self.u_plus.detach().cpu().numpy()}")
        # print(f"u_minus: {self.u_minus.detach().cpu().numpy()}")
        # print(f"v_plus: {self.v_plus.detach().cpu().numpy()}")
        # print(f"v_minus: {self.v_minus.detach().cpu().numpy()}")
        
        # Print final invariants
        c_final = self.c_history[-1]
        delta_plus_final = self.delta_plus_history[-1]
        delta_minus_final = self.delta_minus_history[-1]
        # print(f"\nFinal invariants:")
        # print(f"C: {c_final}")
        # print(f"delta_plus: {delta_plus_final}")
        # print(f"delta_minus: {delta_minus_final}")

def compute_c(net):
    # v_plus * v_minus  → (d,)
    # (u_plus * u_minus).sum(dim=1)  → (d,)
    with torch.no_grad():
        return (
            net.v_plus * net.v_minus
            + (net.u_plus * net.u_minus)
        ).cpu().numpy()

def compute_lambda(net):
    # v_plus**2  → (d,)
    # (u_plus**2).sum(dim=1)  → (d,)
    with torch.no_grad():
        return (
            net.v_plus**2
            - net.u_plus**2
        ).cpu().numpy()

def predict_parameters(lambda_val, c, beta):
    lambda_val = np.array(lambda_val).flatten()  # shape (d,)
    c = np.array(c).flatten()                    # shape (d,)
    beta = np.array(beta).flatten()                       # shape (d, o)
    # d, o = beta.shape
    d = len(beta)
    o = 1

    abs_lambda = np.abs(lambda_val)
    sqrt_lambda = np.sqrt(abs_lambda)
    c_over_lambda = (c / abs_lambda)
    beta_over_c = (beta.T / c)  # Make sure shapes broadcast correctly

    # # Clip c_over_lambda to be >= 1 to avoid nan in arccosh
    # c_over_lambda_clipped = np.clip(c_over_lambda, 1, None)

    theta_plus = 0.5 * (np.arccosh(c_over_lambda) + np.arcsinh(beta_over_c))
    theta_minus = 0.5 * (np.arccosh(c_over_lambda) - np.arcsinh(beta_over_c))

    is_pos = lambda_val > 0
    is_neg = ~is_pos

    v_plus = np.zeros((d, o))
    u_plus = np.zeros((d, o))
    v_minus = np.zeros((d, o))
    u_minus = np.zeros((d, o))

    if np.any(is_pos):
        v_plus[is_pos]  = (sqrt_lambda[is_pos] * np.cosh(theta_plus[is_pos])).reshape(-1, 1)
        u_plus[is_pos]  = (sqrt_lambda[is_pos] * np.sinh(theta_plus[is_pos])).reshape(-1, 1)
        v_minus[is_pos] = (sqrt_lambda[is_pos] * np.cosh(theta_minus[is_pos])).reshape(-1, 1)
        u_minus[is_pos] = (sqrt_lambda[is_pos] * np.sinh(theta_minus[is_pos])).reshape(-1, 1)

    if np.any(is_neg):
        u_plus[is_neg]  = (sqrt_lambda[is_neg] * np.cosh(theta_plus[is_neg])).reshape(-1, 1)
        v_plus[is_neg]  = (sqrt_lambda[is_neg] * np.sinh(theta_plus[is_neg])).reshape(-1, 1)
        u_minus[is_neg] = (sqrt_lambda[is_neg] * np.cosh(theta_minus[is_neg])).reshape(-1, 1)
        v_minus[is_neg] = (sqrt_lambda[is_neg] * np.sinh(theta_minus[is_neg])).reshape(-1, 1)

    # v_plus_pred and v_minus_pred should be (d,), not (d, o)
    # For now, use the first output (j=0) for each input dimension
    # v_plus_pred = v_plus[:, 0]
    # v_minus_pred = v_minus[:, 0]
    v_plus_pred = v_plus
    v_minus_pred = v_minus
    u_plus_pred = u_plus
    u_minus_pred = u_minus

    return v_plus_pred, u_plus_pred, v_minus_pred, u_minus_pred

def main():
    config  = NetworkConfig()
    network = DiagonalNetwork(config)
    # Print c and lambda before training
    c_pt = compute_c(network)
    lambda_pt = compute_lambda(network)
    print(f"Before training: c_pt = {c_pt}, lambda_pt = {lambda_pt}")
    network.train()
    # Print c and lambda after training
    c_ft = compute_c(network)
    lambda_ft = compute_lambda(network)
    print(f"After training: c_ft = {c_ft}, lambda_ft = {lambda_ft}")
    # Print final learned parameters
    print("Final learned parameters:")
    print("u_plus =", network.u_plus.detach().cpu().numpy())
    print("u_minus =", network.u_minus.detach().cpu().numpy())
    print("v_plus =", network.v_plus.detach().cpu().numpy())
    print("v_minus =", network.v_minus.detach().cpu().numpy())

    # Compute final w_tilde (beta)
    with torch.no_grad():
        w_tilde_final = (network.u_plus * network.v_plus - network.u_minus * network.v_minus).cpu().numpy()

    # Predict parameters using your theory
    lambda_pred = compute_lambda(network)
    c_pred = compute_c(network)
    beta_pred = w_tilde_final
    v_plus_pred, u_plus_pred, v_minus_pred, u_minus_pred = predict_parameters(lambda_pred, c_pred, beta_pred)

    print("\nPredicted parameters (from theory):")
    print("u_plus_pred =", u_plus_pred)
    print("u_minus_pred =", u_minus_pred)
    print("v_plus_pred =", v_plus_pred)
    print("v_minus_pred =", v_minus_pred)


    print('Invariance check:')
    print('Empirical: ')
    print('lambda plus: ', network.v_plus.detach().cpu().numpy()**2 - network.u_plus.detach().cpu().numpy()**2)
    print('lambda minus: ', network.v_minus.detach().cpu().numpy()**2 - network.u_minus.detach().cpu().numpy()**2)
    # print('c: ', network.v_plus.detach().cpu().numpy() * network.v_minus.detach().cpu().numpy() + (network.u_plus.detach().cpu().numpy() * network.u_minus.detach().cpu().numpy()))
    print('Analytical: ')
    print('lambda plus: ', v_plus_pred**2 - u_plus_pred**2)
    print('lambda minus: ', v_minus_pred**2 - u_minus_pred**2)
    # print('c: ', v_plus_pred * v_minus_pred + (u_plus_pred * u_minus_pred))

    print('Differences between predicted and true parameters:')
    print('u_plus_pred - u_plus_true =', np.linalg.norm(u_plus_pred - network.u_plus.detach().cpu().numpy()))
    print('u_minus_pred - u_minus_true =', np.linalg.norm(u_minus_pred - network.u_minus.detach().cpu().numpy()))
    print('v_plus_pred - v_plus_true =', np.linalg.norm(v_plus_pred - network.v_plus.detach().cpu().numpy()))
    print('v_minus_pred - v_minus_true =', np.linalg.norm(v_minus_pred - network.v_minus.detach().cpu().numpy()))

    network.plot_training_results()

if __name__ == "__main__":
    main()
