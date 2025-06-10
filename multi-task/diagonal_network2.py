import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Tuple

@dataclass
class NetworkConfig:
    d: int = 2  # input dimension
    o: int = 5  # output dimension
    N: int = 5  # number of samples
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
        self.teacher_u_plus = np.random.randn(self.config.d, self.config.o) * self.config.teacher_scale
        self.teacher_u_minus = np.random.randn(self.config.d, self.config.o) * self.config.teacher_scale
        self.teacher_v_plus = np.random.randn(self.config.d) * self.config.teacher_scale
        self.teacher_v_minus = np.random.randn(self.config.d) * self.config.teacher_scale

        # Compute true beta matrix (d x o)
        self.true_beta = (self.teacher_u_plus * self.teacher_v_plus[:, None] - 
                         self.teacher_u_minus * self.teacher_v_minus[:, None])

        # Generate data
        self.X = np.random.randn(self.config.d, self.config.N)
        self.y = self.true_beta.T @ self.X  # (o x N)

        self.X_torch = torch.tensor(self.X, dtype=torch.float32)
        self.y_torch = torch.tensor(self.y, dtype=torch.float32)

    def _init_parameters(self):
        d, o = self.config.d, self.config.o
        scale = self.config.scale

        # param_value = torch.randn(d, o) * scale
        param_value = torch.randn(d) * scale
        param_value_u = param_value.unsqueeze(1).repeat(1, o)


        self.u_plus = nn.Parameter(param_value_u.clone())
        self.v_plus = nn.Parameter(param_value.clone())

        self.u_minus = nn.Parameter(param_value_u.clone())
        self.v_minus = nn.Parameter(param_value.clone())

        # self.u_plus = nn.Parameter(torch.randn(d, o) * scale)
        # self.u_minus = nn.Parameter(torch.randn(d, o) * scale)
        # self.v_plus = nn.Parameter(torch.randn(d) * scale)
        # self.v_minus = nn.Parameter(torch.randn(d) * scale)

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Compute w_tilde matrix (d x o)
        w_tilde = (self.u_plus * self.v_plus[:, None] - 
                  self.u_minus * self.v_minus[:, None])
        # Compute predictions (o x N)
        preds = w_tilde.T @ self.X_torch
        error = preds - self.y_torch
        return w_tilde, preds, error

    def compute_loss(self, error: torch.Tensor) -> torch.Tensor:
        # error is (o x N) matrix
        # Compute squared Frobenius norm and divide by N for mean
        return 0.5 * torch.sum(error ** 2) / self.config.N
    
    def compute_analytical_gradients(self) -> Tuple[np.ndarray, ...]:
        """
        Returns analytic (pre-step) gradients for all four parameters plus
        the predicted Δw_tilde and dL/dt.
        """
        X, Y, N = self.X, self.y, self.config.N
        lr      = self.config.lr

        # Current parameters (numpy)
        u_plus  = self.u_plus.detach().numpy()      # (d, o)
        u_minus = self.u_minus.detach().numpy()     # (d, o)
        v_plus  = self.v_plus.detach().numpy()      # (d,)
        v_minus = self.v_minus.detach().numpy()     # (d,)

        # Build w_tilde with *current* parameters
        w_tilde = u_plus * v_plus[:, None] - u_minus * v_minus[:, None]

        # Residual and core gradient
        residual = Y - w_tilde.T @ X                # (o, N)
        sum_xr   = X @ residual.T / N               # (d, o)
        dL_dw    = -sum_xr                          # correct sign

        # Parameter-wise analytic gradients
        du_plus  = dL_dw * v_plus[:, None]          # (d, o)
        du_minus = -dL_dw * v_minus[:, None]        # (d, o)  (+ sign overall)
        dv_plus  = np.sum(u_plus  * dL_dw, axis=1)  # (d,)
        dv_minus = -np.sum(u_minus * dL_dw, axis=1) # (d,)

        # Predicted parameter velocities (-lr · gradient)
        du_plus_dt  = -lr * du_plus
        du_minus_dt = -lr * du_minus
        dv_plus_dt  = -lr * dv_plus
        dv_minus_dt = -lr * dv_minus

        # Predicted Δw_tilde from those velocities
        dw_tilde_dt = (
            du_plus_dt * v_plus[:, None]
            + u_plus * dv_plus_dt[:, None]
            - du_minus_dt * v_minus[:, None]
            - u_minus * dv_minus_dt[:, None]
        ) 

        # Predicted loss rate
        dL_dt = np.sum(dL_dw * dw_tilde_dt)

        return du_plus, du_minus, dv_plus, dv_minus, dw_tilde_dt, dL_dt

    # def compute_analytical_gradients(self, w_tilde: torch.Tensor) -> Tuple[np.ndarray, ...]:
    #     X = self.X  # (d x N)
    #     Y = self.y  # (o x N)
    #     N = self.config.N

    #     # Convert to numpy
    #     u_plus = self.u_plus.detach().numpy()[:, 0]  # (d,)
    #     v_plus = self.v_plus.detach().numpy()       # (d,)
    #     u_minus = self.u_minus.detach().numpy()[:, 0]
    #     v_minus = self.v_minus.detach().numpy()

    #     # Compute residual
    #     pred = w_tilde.detach().numpy().T @ X  # (o x N)
    #     residual = Y - pred                    # (o x N)

    #     # Compute dL/dW
    #     sum_xr = X @ residual.T / N            # (d x o)

    #     # Gradients for individual parameters
    #     du_plus = sum_xr * v_plus[:, None]              # (d x o)
    #     dv_plus = np.sum(u_plus[:, None] * sum_xr, axis=1)  # (d,)

    #     du_minus = -sum_xr * v_minus[:, None]
    #     dv_minus = -np.sum(u_minus[:, None] * sum_xr, axis=1)

    #     # Compute dw_tilde/dt (what the optimizer does to w_tilde)
    #     # w_tilde = u_plus * v_plus - u_minus * v_minus
    #     # So dw_tilde/dt = du_plus/dt * v_plus + u_plus * dv_plus/dt - du_minus/dt * v_minus - u_minus * dv_minus/dt
    #     dw_tilde = (-self.config.lr * (
    #         du_plus * v_plus[:, None] + 
    #         u_plus[:, None] * dv_plus[:, None] - 
    #         du_minus * v_minus[:, None] - 
    #         u_minus[:, None] * dv_minus[:, None]
    #     ))

    #     # Compute dL/dt using Frobenius inner product
    #     dL_dw = sum_xr  # (d x o)
    #     dL_dt = np.sum(dL_dw * dw_tilde)

    #     return du_plus, du_minus, dv_plus, dv_minus, dw_tilde, dL_dt

    def _init_histories(self):
        self.loss_history = []
        self.w_tilde_history = []
        self.u_plus_history = []
        self.u_minus_history = []
        self.v_plus_history = []
        self.v_minus_history = []

        self.du_plus_analytical = []
        self.du_minus_analytical = []
        self.dv_plus_analytical = []
        self.dv_minus_analytical = []
        self.dw_tilde_analytical = []
        self.dL_dt_analytical = []

        self.c_history = []
        self.delta_plus_history = []
        self.delta_minus_history = []

    def train(self):
        for t in range(self.config.T):
            self.optimizer.zero_grad()
            w_tilde, preds, error = self.forward()
            loss = self.compute_loss(error)

            self.loss_history.append(loss.item())
            self.w_tilde_history.append(w_tilde.detach().numpy().copy())
            self.u_plus_history.append(self.u_plus.detach().numpy().copy())
            self.u_minus_history.append(self.u_minus.detach().numpy().copy())
            self.v_plus_history.append(self.v_plus.detach().numpy().copy())
            self.v_minus_history.append(self.v_minus.detach().numpy().copy())

            with torch.no_grad():
                # c = (self.u_plus * self.u_minus + 
                #      self.v_plus[:, None] * self.v_minus[:, None]).detach().numpy()
                # c = (torch.sum(self.u_plus * self.u_minus, dim=1) + self.v_plus * self.v_minus).detach().numpy()
                c = (
                torch.sum(self.u_plus * self.u_minus, dim=1)  # Σ_j u⁻_ij u⁺_ij, shape (d,)
                + self.v_plus * self.v_minus                  # v⁻_i v⁺_i,   shape (d,)
                ).detach().cpu().numpy()
                self.c_history.append(c.copy())
                delta_plus = (self.v_plus**2 - 
                            torch.sum(self.u_plus**2, dim=1)).detach().numpy()
                delta_minus = (self.v_minus**2 - 
                             torch.sum(self.u_minus**2, dim=1)).detach().numpy()
                # self.c_history.append(c.copy())
                self.delta_plus_history.append(delta_plus.copy())
                self.delta_minus_history.append(delta_minus.copy())

            loss.backward()
            self.optimizer.step()

            du_plus, du_minus, dv_plus, dv_minus, dw_tilde, dL_dt = self.compute_analytical_gradients()
            self.du_plus_analytical.append(du_plus.copy())
            self.du_minus_analytical.append(du_minus.copy())
            self.dv_plus_analytical.append(dv_plus.copy())
            self.dv_minus_analytical.append(dv_minus.copy())
            self.dw_tilde_analytical.append(dw_tilde.copy())
            self.dL_dt_analytical.append(dL_dt)

    def plot_training_results(self):
        u_plus_history = np.array(self.u_plus_history)
        u_minus_history = np.array(self.u_minus_history)
        v_plus_history = np.array(self.v_plus_history)
        v_minus_history = np.array(self.v_minus_history)
        w_tilde_history = np.array(self.w_tilde_history)

        empirical_dloss = np.diff(self.loss_history)
        # Divide empirical changes by learning rate and flip sign to match analytical gradients
        du_plus_empirical = -np.diff(u_plus_history, axis=0) / self.config.lr
        du_minus_empirical = -np.diff(u_minus_history, axis=0) / self.config.lr
        dv_plus_empirical = -np.diff(v_plus_history, axis=0) / self.config.lr
        dv_minus_empirical = -np.diff(v_minus_history, axis=0) / self.config.lr
        dw_tilde_empirical = np.diff(w_tilde_history, axis=0)

        du_plus_analytical = np.array(self.du_plus_analytical[:-1])
        du_minus_analytical = np.array(self.du_minus_analytical[:-1])
        dv_plus_analytical = np.array(self.dv_plus_analytical[:-1])
        dv_minus_analytical = np.array(self.dv_minus_analytical[:-1])
        dw_tilde_analytical = np.array(self.dw_tilde_analytical[:-1])

        # Figure 1: Parameter dynamics
        fig1 = plt.figure(figsize=(10, 4*self.config.d))
        plt.subplot(5, self.config.d, 1)
        for i in range(self.config.d):
            plt.plot(w_tilde_history[:, i], label=f'w_tilde[{i}]')
        plt.title("Dynamics of w_tilde")
        plt.legend()
        plt.grid(True)

        plt.subplot(5, self.config.d, self.config.d + 1)
        for i in range(self.config.d):
            plt.plot(u_plus_history[:, i], label=f'u_plus[{i}]')
        plt.title("Dynamics of u_plus")
        plt.legend()
        plt.grid(True)

        plt.subplot(5, self.config.d, 2*self.config.d + 1)
        for i in range(self.config.d):
            plt.plot(u_minus_history[:, i], label=f'u_minus[{i}]')
        plt.title("Dynamics of u_minus")
        plt.legend()
        plt.grid(True)

        plt.subplot(5, self.config.d, 3*self.config.d + 1)
        for i in range(self.config.d):
            plt.plot(v_plus_history[:, i], label=f'v_plus[{i}]')
        plt.title("Dynamics of v_plus")
        plt.legend()
        plt.grid(True)

        plt.subplot(5, self.config.d, 4*self.config.d + 1)
        for i in range(self.config.d):
            plt.plot(v_minus_history[:, i], label=f'v_minus[{i}]')
        plt.title("Dynamics of v_minus")
        plt.legend()
        plt.grid(True)

        fig1.tight_layout()

        # Figure 2: Loss
        fig2 = plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(self.loss_history)
        plt.title("Loss over Training Steps")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(empirical_dloss, label='Empirical dL/dt')
        plt.plot(self.dL_dt_analytical[:-1], '--', label='Analytical dL/dt')
        plt.title("Rate of Change of Loss")
        plt.xlabel("Step")
        plt.ylabel("dL/dt")
        plt.legend()
        plt.grid(True)
        fig2.tight_layout()

        # Figure 3: Rate of change of w_tilde
        fig3 = plt.figure(figsize=(10, 4))
        for i in range(self.config.d):
            plt.plot(dw_tilde_empirical[:, i], label=f'Empirical w_tilde[{i}]')
            plt.plot(dw_tilde_analytical[:, i], '--', label=f'Analytical w_tilde[{i}]')
        plt.title("Rate of Change of w_tilde")
        plt.xlabel("Step")
        plt.ylabel("dw_tilde/dt")
        plt.legend()
        plt.grid(True)
        fig3.tight_layout()

        # Figure 4: Rate of change of u_plus
        fig4 = plt.figure(figsize=(10, 4))
        for i in range(self.config.d):
            plt.plot(du_plus_empirical[:, i], label=f'Empirical u_plus[{i}]')
            plt.plot(du_plus_analytical[:, i], '--', label=f'Analytical u_plus[{i}]')
        plt.title("Rate of Change of u_plus")
        plt.xlabel("Step")
        plt.ylabel("du_plus/dt")
        plt.legend()
        plt.grid(True)
        fig4.tight_layout()

        # Figure 5: Rate of change of u_minus
        fig5 = plt.figure(figsize=(10, 4))
        for i in range(self.config.d):
            plt.plot(du_minus_empirical[:, i], label=f'Empirical u_minus[{i}]')
            plt.plot(du_minus_analytical[:, i], '--', label=f'Analytical u_minus[{i}]')
        plt.title("Rate of Change of u_minus")
        plt.xlabel("Step")
        plt.ylabel("du_minus/dt")
        plt.legend()
        plt.grid(True)
        fig5.tight_layout()

        # Figure 6: Rate of change of v_plus
        fig6 = plt.figure(figsize=(10, 4))
        for i in range(self.config.d):
            plt.plot(dv_plus_empirical[:, i], label=f'Empirical v_plus[{i}]')
            plt.plot(dv_plus_analytical[:, i], '--', label=f'Analytical v_plus[{i}]')
        plt.title("Rate of Change of v_plus")
        plt.xlabel("Step")
        plt.ylabel("dv_plus/dt")
        plt.legend()
        plt.grid(True)
        fig6.tight_layout()

        # Figure 7: Rate of change of v_minus
        fig7 = plt.figure(figsize=(10, 4))
        for i in range(self.config.d):
            plt.plot(dv_minus_empirical[:, i], label=f'Empirical v_minus[{i}]')
            plt.plot(dv_minus_analytical[:, i], '--', label=f'Analytical v_minus[{i}]')
        plt.title("Rate of Change of v_minus")
        plt.xlabel("Step")
        plt.ylabel("dv_minus/dt")
        plt.legend()
        plt.grid(True)
        fig7.tight_layout()

        # # Figure 8: c and delta diagnostics
        # fig8 = plt.figure(figsize=(10, 9))
        # c_array = np.array(self.c_history)
        # delta_plus_array = np.array(self.delta_plus_history)
        # delta_minus_array = np.array(self.delta_minus_history)

        # plt.subplot(3, 1, 1)
        # for i in range(self.config.d):
        #     plt.plot(c_array[:, i], label=f'c[{i}]')
        # plt.title("c = u⁺ ∘ u⁻ + v⁺ ∘ v⁻")
        # plt.xlabel("Step")
        # plt.grid(True)
        # plt.legend()

        # plt.subplot(3, 1, 2)
        # for i in range(self.config.d):
        #     plt.plot(delta_plus_array[:, i], label=f'δ⁺[{i}]')
        # plt.title("delta⁺ = v⁺² - u⁺²")
        # plt.xlabel("Step")
        # plt.grid(True)
        # plt.legend()

        # plt.subplot(3, 1, 3)
        # for i in range(self.config.d):
        #     plt.plot(delta_minus_array[:, i], label=f'δ⁻[{i}]')
        # plt.title("delta⁻ = v⁻² - u⁻²")
        # plt.xlabel("Step")
        # plt.grid(True)
        # plt.legend()

        # fig8.tight_layout()
        # plt.show()


        # Figure 8: c and delta diagnostics
        fig8 = plt.figure(figsize=(10, 9))

        c_array = np.array(self.c_history)               # (steps, d)
        delta_plus_array = np.array(self.delta_plus_history)
        delta_minus_array = np.array(self.delta_minus_history)
        steps = np.arange(len(c_array))

        # First plot: c
        plt.subplot(3, 1, 1)
        if self.config.d <= 20:
            for i in range(self.config.d):
                plt.plot(steps, c_array[:, i], label=f'c[{i}]')
        else:
            mean = c_array.mean(axis=1)
            std = c_array.std(axis=1)
            plt.plot(steps, mean, label='mean(c)')
            plt.fill_between(steps, mean - std, mean + std, alpha=0.3, label='±1 std')
        plt.title("c = u⁺ ∘ u⁻ + v⁺ ∘ v⁻")
        plt.xlabel("Step")
        plt.grid(True)
        plt.legend()

        # Second plot: delta⁺
        plt.subplot(3, 1, 2)
        if self.config.d <= 20:
            for i in range(self.config.d):
                plt.plot(steps, delta_plus_array[:, i], label=f'δ⁺[{i}]')
        else:
            mean = delta_plus_array.mean(axis=1)
            std = delta_plus_array.std(axis=1)
            plt.plot(steps, mean, label='mean(δ⁺)')
            plt.fill_between(steps, mean - std, mean + std, alpha=0.3, label='±1 std')
        plt.title("delta⁺ = v⁺² - ∥u⁺∥²")
        plt.xlabel("Step")
        plt.grid(True)
        plt.legend()

        # Third plot: delta⁻
        plt.subplot(3, 1, 3)
        if self.config.d <= 20:
            for i in range(self.config.d):
                plt.plot(steps, delta_minus_array[:, i], label=f'δ⁻[{i}]')
        else:
            mean = delta_minus_array.mean(axis=1)
            std = delta_minus_array.std(axis=1)
            plt.plot(steps, mean, label='mean(δ⁻)')
            plt.fill_between(steps, mean - std, mean + std, alpha=0.3, label='±1 std')
        plt.title("delta⁻ = v⁻² - ∥u⁻∥²")
        plt.xlabel("Step")
        plt.grid(True)
        plt.legend()

        fig8.tight_layout()
        plt.show()


def main():
    config = NetworkConfig()
    network = DiagonalNetwork(config)
    network.train()
    network.plot_training_results()

if __name__ == "__main__":
    main()
