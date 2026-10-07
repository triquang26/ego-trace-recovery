import numpy as np
import torch
from torch import Tensor


def clamped_knots(num_controls: int, degree: int) -> np.ndarray:
    interior = num_controls - degree - 1
    inner = np.arange(1, interior + 1) / (interior + 1)
    return np.concatenate([np.zeros(degree + 1), inner, np.ones(degree + 1)])


def basis_matrix(times: np.ndarray, knots: np.ndarray, degree: int, num_controls: int) -> np.ndarray:
    times = np.asarray(times, dtype=np.float64)
    basis = np.zeros((len(times), len(knots) - 1))
    for i in range(len(knots) - 1):
        basis[:, i] = (knots[i] <= times) & (times < knots[i + 1])
    last = np.nonzero(knots[:-1] < knots[1:])[0][-1]
    basis[times >= knots[-1], last] = 1.0
    for p in range(1, degree + 1):
        nxt = np.zeros((len(times), len(knots) - p - 1))
        for i in range(len(knots) - p - 1):
            left = knots[i + p] - knots[i]
            right = knots[i + p + 1] - knots[i + 1]
            if left > 0:
                nxt[:, i] += (times - knots[i]) / left * basis[:, i]
            if right > 0:
                nxt[:, i] += (knots[i + p + 1] - times) / right * basis[:, i + 1]
        basis = nxt
    return basis[:, :num_controls]


class BSplineTargets:
    def __init__(self, future_steps: int, free_controls: int, degree: int, regularization: float,
                 min_valid_steps: int, max_condition: float):
        self.knots = clamped_knots(free_controls + 1, degree)
        times = np.arange(1, future_steps + 1) / future_steps
        full = basis_matrix(times, self.knots, degree, free_controls + 1)
        self.free_basis = torch.from_numpy(full[:, 1:]).double()
        difference = np.eye(free_controls) - np.eye(free_controls, k=-1)
        self.difference = torch.from_numpy(difference).double()
        self.regularization = regularization
        self.min_valid_steps = min_valid_steps
        self.max_condition = max_condition

    def fit(self, trace: Tensor, weight: Tensor) -> tuple[Tensor, Tensor]:
        basis = self.free_basis.to(trace.device)
        reg = self.regularization * (self.difference.T @ self.difference).to(trace.device)
        w = weight.double()
        y = torch.where(w[..., None] > 0, trace.double(), torch.zeros_like(trace, dtype=torch.float64))
        lhs = torch.einsum("km,...k,kn->...mn", basis, w, basis) + reg
        rhs = torch.einsum("km,...k,...kc->...mc", basis, w, y)
        controls = torch.linalg.solve(lhs, rhs)
        condition = torch.linalg.cond(lhs)
        fit_valid = ((weight > 0).sum(-1) >= self.min_valid_steps) & (condition <= self.max_condition)
        fit_valid &= torch.isfinite(controls).all(-1).all(-1)
        controls = torch.where(fit_valid[..., None, None], controls, torch.zeros_like(controls))
        return controls.to(trace.dtype), fit_valid

    def decode(self, controls: Tensor) -> Tensor:
        basis = self.free_basis.to(controls.device, controls.dtype)
        return torch.einsum("km,...mc->...kc", basis, controls)
