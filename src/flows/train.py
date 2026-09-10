from __future__ import annotations

import numpy as np
import torch
from torch import nn

from . import const
from .data import encode


def _time_feat(t: torch.Tensor) -> torch.Tensor:
    if t.ndim == 1:
        t = t[:, None]
    return torch.cat(
        [t, torch.sin(2 * torch.pi * t), torch.cos(2 * torch.pi * t), torch.sin(4 * torch.pi * t)],
        dim=-1,
    )


class Velocity(nn.Module):
    def __init__(self, width: int = const.WIDTH, cond_dim: int = const.COND_DIM):
        super().__init__()
        d = const.N_DIM + 4 + cond_dim  # x, Fourier t, c
        self.net = nn.Sequential(
            nn.Linear(d, width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
            nn.Linear(width, const.N_DIM),
        )

    def forward(self, x, t, c):
        return self.net(torch.cat([x, _time_feat(t), c], dim=-1))


def train_cfm(
    x: np.ndarray,
    c: np.ndarray,
    epochs: int = const.EPOCHS,
    width: int = const.WIDTH,
    cond_dim: int = const.COND_DIM,
    seed: int = const.SEED,
) -> Velocity:
    if epochs > const.MAX_EPOCHS:
        raise ValueError(f"cap: epochs {epochs} > MAX_EPOCHS {const.MAX_EPOCHS}")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = Velocity(width=width, cond_dim=cond_dim)
    opt = torch.optim.Adam(model.parameters(), lr=const.LR)
    n = len(x)
    enc = encode(c, cond_dim)
    model.train()
    for _ in range(epochs):
        order = rng.permutation(n)
        for i in range(0, n, const.BATCH):
            sl = order[i : i + const.BATCH]
            x1 = torch.from_numpy(x[sl])
            cc = torch.from_numpy(enc[sl])
            x0 = torch.randn_like(x1)
            t = torch.rand(len(sl))
            xt = (1.0 - t[:, None]) * x0 + t[:, None] * x1
            target = x1 - x0
            loss = ((model(xt, t, cc) - target) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
    model.eval()
    return model


@torch.no_grad()
def sample(
    model: Velocity,
    conds: np.ndarray,
    steps: int = const.STEPS,
    cond_dim: int = const.COND_DIM,
    seed: int = const.SEED,
) -> np.ndarray:
    torch.manual_seed(seed)
    n = len(conds)
    x = torch.randn(n, const.N_DIM)
    c = torch.from_numpy(encode(conds, cond_dim))
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((n,), i * dt)
        x = x + dt * model(x, t, c)
    return x.numpy().astype(np.float32)
