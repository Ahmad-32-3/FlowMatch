from __future__ import annotations

import numpy as np

from . import const


def angle(k, n: int = const.N_CONDS) -> float:
    return 2.0 * np.pi * float(k) / n


def true_mean(k, radius: float = const.RADIUS) -> np.ndarray:
    th = angle(k)
    return np.array([radius * np.cos(th), radius * np.sin(th)], dtype=np.float32)


def encode(conds: np.ndarray, cond_dim: int = const.COND_DIM) -> np.ndarray:
    th = 2.0 * np.pi * conds.astype(np.float32) / const.N_CONDS
    if cond_dim == 1:
        return (th / np.pi).astype(np.float32)[:, None]
    return np.stack([np.cos(th), np.sin(th)], axis=1).astype(np.float32)


def check_no_leak(train_conds, hold_conds) -> None:
    overlap = set(int(c) for c in train_conds) & set(int(c) for c in hold_conds)
    if overlap:
        raise ValueError(f"leak: holdout conditions {sorted(overlap)} in train")


def check_split(train_ids, test_ids) -> None:
    overlap = set(int(i) for i in train_ids) & set(int(i) for i in test_ids)
    if overlap:
        raise ValueError("leak: test ids in train")


def in_region(x: np.ndarray, conds: np.ndarray, tau: float = const.TAU) -> np.ndarray:
    means = np.stack([true_mean(int(k)) for k in conds])
    dist = np.linalg.norm(x - means, axis=1)
    return dist < tau


def success_pct(x: np.ndarray, conds: np.ndarray, tau: float = const.TAU) -> float:
    hits = in_region(x, conds, tau)
    return float(100.0 * hits.mean()) if len(hits) else 0.0


def make_split(
    n_per: int = const.N_PER_TRAIN,
    seed: int = const.SEED,
    train_conds=const.TRAIN_CONDS,
    hold_conds=const.HOLD_CONDS,
) -> dict:
    train_conds = tuple(int(c) for c in train_conds)
    hold_conds = tuple(int(c) for c in hold_conds)
    check_no_leak(train_conds, hold_conds)
    rng = np.random.default_rng(seed)
    xs, cs, ids = [], [], []
    i = 0
    for k in train_conds:
        mu = true_mean(k)
        x = rng.normal(mu, const.SIGMA, size=(n_per, const.N_DIM)).astype(np.float32)
        xs.append(x)
        cs.append(np.full(n_per, k, dtype=np.int32))
        ids.append(np.arange(i, i + n_per, dtype=np.int32))
        i += n_per
    x = np.concatenate(xs)
    c = np.concatenate(cs)
    train_ids = np.concatenate(ids)
    n = len(x)
    if n > const.MAX_N:
        raise ValueError(f"cap: n={n} > MAX_N={const.MAX_N}")
    hold_ids = np.arange(i, i + const.N_SAMPLE * len(hold_conds), dtype=np.int32)
    check_split(train_ids, hold_ids)
    return {
        "x": x,
        "c": c,
        "train_ids": train_ids,
        "hold_ids": hold_ids,
        "train_conds": np.array(train_conds, dtype=np.int32),
        "hold_conds": np.array(hold_conds, dtype=np.int32),
        "n_train": n,
        "seed": seed,
    }
