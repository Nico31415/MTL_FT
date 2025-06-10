import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Tuple

@dataclass
class NetworkConfig:
    d: int = 2   # input dimension
    o: int = 5   # output dimension
    N: int = 5   # number of samples
    T: int = 500
    lr: float = 1e-3
    scale: float = 0.01
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

        # independent initialization for each branch
        self.u_plus  = nn.Parameter(torch.randn(d, o) * scale)
        self.u_minus = nn.Parameter(torch.randn(d, o) * scale)
        self.v_plus  = nn.Parameter(torch.randn(d)     * scale)
        self.v_minus = nn.Parameter(torch.randn(d)     * scale)

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Effective weight matrix W = u⁺∘v⁺ - u⁻∘v⁻
        w_tilde = (self.u_plus * self.v_plus[:, None]
                   - self.u_minus * self.v_minus[:, None])
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
        w_tilde = u_plus * v_plus[:, None] - u_minus * v_minus[:, None]
        error   = w_tilde.T @ X - Y                     # (o, N)
        G       = (X @ error.T) / N                    # (d, o)

        # Parameter-wise gradients
        du_plus  =  G * v_plus[:, None]                 # dL/dU⁺
        du_minus = -G * v_minus[:, None]                # dL/dU⁻
        dv_plus  =  np.sum(u_plus  * G, axis=1)         # dL/dv⁺
        dv_minus = -np.sum(u_minus * G, axis=1)         # dL/dv⁻

        # Velocities under gradient descent
        du_plus_dt  = -lr * du_plus
        du_minus_dt = -lr * du_minus
        dv_plus_dt  = -lr * dv_plus
        dv_minus_dt = -lr * dv_minus

        # Induced change in w_tilde
        dw_tilde_dt = (
            du_plus_dt * v_plus[:, None]
            + u_plus * dv_plus_dt[:, None]
            - du_minus_dt * v_minus[:, None]
            - u_minus * dv_minus_dt[:, None]
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
        # Convert histories to arrays
        u_plus_history       = np.array(self.u_plus_history)
        u_minus_history      = np.array(self.u_minus_history)
        v_plus_history       = np.array(self.v_plus_history)
        v_minus_history      = np.array(self.v_minus_history)
        w_tilde_history      = np.array(self.w_tilde_history)

        empirical_dloss    = np.diff(self.loss_history)
        du_plus_empirical  = -np.diff(u_plus_history, axis=0) / self.config.lr
        du_minus_empirical = -np.diff(u_minus_history, axis=0) / self.config.lr
        dv_plus_empirical  = -np.diff(v_plus_history, axis=0) / self.config.lr
        dv_minus_empirical = -np.diff(v_minus_history, axis=0) / self.config.lr
        dw_tilde_empirical =  np.diff(w_tilde_history, axis=0)

        du_plus_analytical  = np.array(self.du_plus_analytical[:-1])
        du_minus_analytical = np.array(self.du_minus_analytical[:-1])
        dv_plus_analytical  = np.array(self.dv_plus_analytical[:-1])
        dv_minus_analytical = np.array(self.dv_minus_analytical[:-1])
        dw_tilde_analytical = np.array(self.dw_tilde_analytical[:-1])

        # Plot parameters & rates (omitted for brevity)... (same as before)

        # Plot invariants derivatives
        c_arr     = np.array(self.c_history)
        dp_arr    = np.array(self.delta_plus_history)
        dm_arr    = np.array(self.delta_minus_history)
        dc_emp    = np.diff(c_arr, axis=0)
        ddp_emp   = np.diff(dp_arr, axis=0)
        ddm_emp   = np.diff(dm_arr, axis=0)

        dC_anal   = np.array(self.dC_dt_analytical[:-1])
        ddp_anal  = np.array(self.ddelta_plus_dt_analytical[:-1])
        ddm_anal  = np.array(self.ddelta_minus_dt_analytical[:-1])

        plt.figure(figsize=(10,9))
        titles = ['dC/dt','ddelta_plus/dt','ddelta_minus/dt']
        for idx,(emp,anal,title) in enumerate([(dc_emp,dC_anal,'dC/dt'),(ddp_emp,ddp_anal,'ddelta_plus/dt'),(ddm_emp,ddm_anal,'ddelta_minus/dt')]):
            plt.subplot(3,1,idx+1)
            for i in range(emp.shape[1]):
                plt.plot(emp[:,i], label=f'Emp {title}[{i}]')
                plt.plot(anal[:,i],'--',label=f'Anl {title}[{i}]')
            plt.title(title), plt.legend(), plt.grid(True)
        plt.tight_layout(), plt.show()


def main():
    config  = NetworkConfig()
    network = DiagonalNetwork(config)
    network.train()
    network.plot_training_results()

if __name__ == "__main__":
    main()
