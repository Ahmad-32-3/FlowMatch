from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import const
from .data import make_split, success_pct, true_mean


def sample_baseline(n: int, train_x: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mu = train_x.mean(axis=0)
    std = float(train_x.std())
    return rng.normal(mu, std, size=(n, const.N_DIM)).astype(np.float32)


def holdout_conds_rep(hold_conds: np.ndarray, n_sample: int) -> np.ndarray:
    return np.repeat(hold_conds.astype(np.int32), n_sample)


def evaluate(
    split: dict | None = None,
    epochs: int = const.EPOCHS,
    width: int = const.WIDTH,
    steps: int = const.STEPS,
    cond_dim: int = const.COND_DIM,
    n_sample: int = const.N_SAMPLE,
    seed: int = const.SEED,
    ablations: bool = False,
) -> dict:
    from .train import sample, train_cfm

    split = split or make_split(seed=seed)
    hold = split["hold_conds"]
    conds = holdout_conds_rep(hold, n_sample)
    model = train_cfm(split["x"], split["c"], epochs=epochs, width=width, cond_dim=cond_dim, seed=seed)
    gen = sample(model, conds, steps=steps, cond_dim=cond_dim, seed=seed + 1)
    flow_pct = success_pct(gen, conds)
    base = sample_baseline(len(conds), split["x"], seed + 2)
    base_pct = success_pct(base, conds)
    row = {
        "success_pct": round(flow_pct, 2),
        "baseline_pct": round(base_pct, 2),
        "n_train": int(split["n_train"]),
        "n_sample": int(len(conds)),
        "n_conds": const.N_CONDS,
        "train_conds": [int(c) for c in split["train_conds"]],
        "hold_conds": [int(c) for c in hold],
        "epochs": int(epochs),
        "width": int(width),
        "steps": int(steps),
        "cond_dim": int(cond_dim),
        "tau": const.TAU,
        "sigma": const.SIGMA,
        "radius": const.RADIUS,
        "seed": int(seed),
        "illustrative": False,
        "blocker": None,
        "samples": gen.tolist(),
        "baseline_samples": base.tolist(),
        "sample_conds": conds.tolist(),
        "train_x": split["x"].tolist(),
        "train_c": split["c"].tolist(),
    }
    if ablations:
        row["ablation"] = run_ablations(split, seed=seed, epochs=epochs, model=model)
    return row


def run_ablations(
    split: dict,
    seed: int = const.SEED,
    epochs: int = const.EPOCHS,
    model=None,
) -> list[dict]:
    from .train import sample, train_cfm

    hold = split["hold_conds"]
    conds = holdout_conds_rep(hold, const.N_SAMPLE)
    base = sample_baseline(len(conds), split["x"], seed + 2)
    base_pct = round(success_pct(base, conds), 2)
    if model is None:
        model = train_cfm(split["x"], split["c"], epochs=epochs, seed=seed)
    rows = []
    for steps in (8, const.STEPS, 32):
        gen = sample(model, conds, steps=steps, seed=seed + 1)
        rows.append(
            {
                "tag": f"steps={steps}",
                "steps": int(steps),
                "width": const.WIDTH,
                "cond_dim": const.COND_DIM,
                "success_pct": round(success_pct(gen, conds), 2),
                "baseline_pct": base_pct,
            }
        )
    default_pct = next(r["success_pct"] for r in rows if r["steps"] == const.STEPS)
    for width in (32, const.WIDTH, 96):
        if width == const.WIDTH:
            rows.append(
                {
                    "tag": f"width={width}",
                    "steps": const.STEPS,
                    "width": int(width),
                    "cond_dim": const.COND_DIM,
                    "success_pct": default_pct,
                    "baseline_pct": base_pct,
                }
            )
            continue
        rows.append(
            _ablate_row(split, conds, base_pct, seed, epochs, width=width, tag=f"width={width}")
        )
    for cond_dim in (1, 2):
        if cond_dim == const.COND_DIM:
            rows.append(
                {
                    "tag": f"cond_dim={cond_dim}",
                    "steps": const.STEPS,
                    "width": const.WIDTH,
                    "cond_dim": int(cond_dim),
                    "success_pct": default_pct,
                    "baseline_pct": base_pct,
                }
            )
            continue
        rows.append(
            _ablate_row(
                split, conds, base_pct, seed, epochs, cond_dim=cond_dim, tag=f"cond_dim={cond_dim}"
            )
        )
    return rows


def _ablate_row(
    split,
    conds,
    base_pct,
    seed,
    epochs,
    steps=const.STEPS,
    width=const.WIDTH,
    cond_dim=const.COND_DIM,
    tag="",
) -> dict:
    from .train import sample, train_cfm

    model = train_cfm(split["x"], split["c"], epochs=epochs, width=width, cond_dim=cond_dim, seed=seed)
    gen = sample(model, conds, steps=steps, cond_dim=cond_dim, seed=seed + 1)
    return {
        "tag": tag,
        "steps": int(steps),
        "width": int(width),
        "cond_dim": int(cond_dim),
        "success_pct": round(success_pct(gen, conds), 2),
        "baseline_pct": base_pct,
    }


def illustrative(blocker: str) -> dict:
    # Placeholder shape only. Captions must say illustrative.
    hold = list(const.HOLD_CONDS)
    n = const.N_SAMPLE * len(hold)
    rng = np.random.default_rng(0)
    conds = holdout_conds_rep(np.array(hold), const.N_SAMPLE)
    gen = np.stack([true_mean(int(k)) + rng.normal(0, const.SIGMA, 2) for k in conds]).astype(np.float32)
    base = rng.normal(0, 1.6, size=(n, 2)).astype(np.float32)
    split = make_split()
    return {
        "success_pct": 92.0,
        "baseline_pct": 11.0,
        "n_train": split["n_train"],
        "n_sample": n,
        "n_conds": const.N_CONDS,
        "train_conds": list(const.TRAIN_CONDS),
        "hold_conds": hold,
        "epochs": const.EPOCHS,
        "width": const.WIDTH,
        "steps": const.STEPS,
        "cond_dim": const.COND_DIM,
        "tau": const.TAU,
        "sigma": const.SIGMA,
        "radius": const.RADIUS,
        "seed": const.SEED,
        "illustrative": True,
        "blocker": blocker,
        "samples": gen.tolist(),
        "baseline_samples": base.tolist(),
        "sample_conds": conds.tolist(),
        "train_x": split["x"].tolist(),
        "train_c": split["c"].tolist(),
        "ablation": [
            {"tag": "steps=8", "steps": 8, "width": 64, "cond_dim": 2, "success_pct": 88.0, "baseline_pct": 11.0},
            {"tag": "steps=16", "steps": 16, "width": 64, "cond_dim": 2, "success_pct": 92.0, "baseline_pct": 11.0},
            {"tag": "steps=32", "steps": 32, "width": 64, "cond_dim": 2, "success_pct": 93.0, "baseline_pct": 11.0},
            {"tag": "width=32", "steps": 16, "width": 32, "cond_dim": 2, "success_pct": 86.0, "baseline_pct": 11.0},
            {"tag": "width=64", "steps": 16, "width": 64, "cond_dim": 2, "success_pct": 92.0, "baseline_pct": 11.0},
            {"tag": "width=96", "steps": 16, "width": 96, "cond_dim": 2, "success_pct": 93.0, "baseline_pct": 11.0},
            {"tag": "cond_dim=1", "steps": 16, "width": 64, "cond_dim": 1, "success_pct": 84.0, "baseline_pct": 11.0},
            {"tag": "cond_dim=2", "steps": 16, "width": 64, "cond_dim": 2, "success_pct": 92.0, "baseline_pct": 11.0},
        ],
    }


def print_report(m: dict) -> None:
    tag = "ILLUSTRATIVE" if m.get("illustrative") else "measured"
    print(
        f"flow success_pct {m['success_pct']:.1f}  |  baseline success_pct {m['baseline_pct']:.1f}  [{tag}]"
    )
    print(
        f"n_train {m['n_train']}  n_sample {m['n_sample']}  hold_conds {m['hold_conds']}  "
        f"steps {m['steps']} width {m['width']} cond_dim {m['cond_dim']}"
    )
    if m.get("ablation"):
        print("ablation:")
        for r in m["ablation"]:
            print(
                f"  {r['tag']:<14} flow {r['success_pct']:.1f}  baseline {r['baseline_pct']:.1f}"
            )


def write_metrics(m: dict, path: str = const.METRICS_PATH) -> None:
    slim = {k: v for k, v in m.items() if k not in {"samples", "baseline_samples", "train_x", "train_c", "sample_conds"}}
    Path(path).write_text(json.dumps(slim, indent=2), encoding="utf-8")


def _scatter_payload(m: dict, n: int = 80) -> dict:
    rng = np.random.default_rng(const.SEED)
    tx = np.array(m["train_x"])
    tc = np.array(m["train_c"])
    gen = np.array(m["samples"])
    base = np.array(m["baseline_samples"])
    sc = np.array(m["sample_conds"])
    idx_t = rng.choice(len(tx), size=min(n, len(tx)), replace=False)
    idx_g = rng.choice(len(gen), size=min(n, len(gen)), replace=False)
    return {
        "train": [{"x": float(tx[i, 0]), "y": float(tx[i, 1]), "c": int(tc[i])} for i in idx_t],
        "flow": [{"x": float(gen[i, 0]), "y": float(gen[i, 1]), "c": int(sc[i])} for i in idx_g],
        "baseline": [{"x": float(base[i, 0]), "y": float(base[i, 1]), "c": int(sc[i])} for i in idx_g],
        "means": [
            {"c": int(k), "x": float(true_mean(k)[0]), "y": float(true_mean(k)[1]), "hold": int(k) in set(m["hold_conds"])}
            for k in range(const.N_CONDS)
        ],
        "tau": const.TAU,
    }


def write_data_ts(m: dict, path: str = const.DATA_TS) -> None:
    p = Path(path)
    if not p.parent.exists():
        return
    abl = m.get("ablation") or []
    scatter = _scatter_payload(m)
    ill = bool(m.get("illustrative"))
    body = _data_ts_body(m, abl, scatter, ill)
    p.write_text(body, encoding="utf-8")
    print(f"wrote {path}")


def _data_ts_body(m: dict, abl: list, scatter: dict, ill: bool) -> str:
    note = "placeholder until the pipeline runs" if ill else "measured on the frozen holdout protocol"
    abl_json = json.dumps(
        [
            {
                "tag": r["tag"],
                "steps": int(r["steps"]),
                "width": int(r["width"]),
                "condDim": int(r["cond_dim"]),
                "successPct": float(r["success_pct"]),
                "baselinePct": float(r["baseline_pct"]),
            }
            for r in abl
        ],
        indent=2,
    )
    scatter_json = json.dumps(scatter)
    return f"""// Generated by scripts/run.py. Numbers match metrics.json.

export const ILLUSTRATIVE = {str(ill).lower()}

export const PROTOCOL = {{
  nConds: {int(m["n_conds"])},
  trainConds: {json.dumps(m["train_conds"])},
  holdConds: {json.dumps(m["hold_conds"])},
  nTrain: {int(m["n_train"])},
  nSample: {int(m["n_sample"])},
  epochs: {int(m["epochs"])},
  width: {int(m["width"])},
  steps: {int(m["steps"])},
  condDim: {int(m["cond_dim"])},
  tau: {float(m["tau"])},
  sigma: {float(m["sigma"])},
  radius: {float(m["radius"])},
  seed: {int(m["seed"])},
  floorPct: {const.FLOOR_PCT},
}}

export const METRICS = {{
  successPct: {float(m["success_pct"])},
  baselinePct: {float(m["baseline_pct"])},
}}

export type AblationRow = {{
  tag: string
  steps: number
  width: number
  condDim: number
  successPct: number
  baselinePct: number
}}

export const ABLATION: AblationRow[] = {abl_json}

export type Pt = {{ x: number; y: number; c: number }}
export type Mean = {{ c: number; x: number; y: number; hold: boolean }}

export const SCATTER = {scatter_json} as {{
  train: Pt[]
  flow: Pt[]
  baseline: Pt[]
  means: Mean[]
  tau: number
}}

export type Counter = {{ key: string; label: string; value: number; unit: string; note: string }}
export const COUNTERS: Counter[] = [
  {{ key: 'flow', label: 'Held-out hit rate', value: METRICS.successPct, unit: '%', note: '{note}' }},
  {{ key: 'base', label: 'Naive cloud hit rate', value: METRICS.baselinePct, unit: '%', note: 'same sample budget, ignores the condition' }},
  {{ key: 'ntrain', label: 'Train points', value: PROTOCOL.nTrain, unit: '', note: 'capped synthetic 2D mixture' }},
  {{ key: 'nsamp', label: 'Holdout samples scored', value: PROTOCOL.nSample, unit: '', note: 'held-out conditions only' }},
]

export const DECISIONS = [
  {{ first: 'Score train likelihood / NLL as the headline', built: 'Score holdout region hit rate against a frozen ball radius' }},
  {{ first: 'A full latent-diffusion stack', built: 'A thin rectified-flow velocity net on 2D' }},
  {{ first: 'Hold out points from conditions the model already saw', built: 'Hold out whole conditions and sample under those' }},
  {{ first: 'A fancy conditional sampler as the only comparison', built: 'An isotropic Gaussian in the same plane, same sample count' }},
] as const

export type Tool = {{ name: string; tag: string; plain: string; tech: string }}
export const STACK: Tool[] = [
  {{
    name: 'synthetic mixture',
    tag: 'data',
    plain: 'Eight clusters on a circle. The mean depends on the condition.',
    tech: 'x ~ N(R [cos θ_c, sin θ_c], σ² I) with R={const.RADIUS}, σ={const.SIGMA}. Train conditions {json.dumps(m["train_conds"])}; holdout {json.dumps(m["hold_conds"])}.',
  }},
  {{
    name: 'rectified flow',
    tag: 'fit',
    plain: 'A small network learns the straight-line velocity from noise to data, given the condition.',
    tech: 'xt = (1-t) x0 + t x1, target v = x1 - x0, MLP on concat(x, t, cos θ, sin θ). {int(m["epochs"])} epochs, width {int(m["width"])}.',
  }},
  {{
    name: 'Euler sample',
    tag: 'sample',
    plain: 'Start from noise and take small steps along the learned velocity, with the condition held fixed.',
    tech: '{int(m["steps"])} Euler steps from t=0 to t=1. Success is landing inside radius τ={const.TAU} of the true mean.',
  }},
  {{
    name: 'isotropic Gaussian',
    tag: 'baseline',
    plain: 'The naive sampler: one blob fit to all train points, no condition.',
    tech: 'Mean and scalar std of train x. Same number of draws as the flow. Hit rate is usually near chance over eight balls.',
  }},
  {{
    name: 'pytest leak check',
    tag: 'test',
    plain: 'If a holdout condition sneaks into train, the test fails on purpose.',
    tech: 'check_no_leak and check_split raise ValueError on overlap. Caps live in const.py.',
  }},
  {{
    name: 'Vite, React, motion',
    tag: 'page',
    plain: 'Builds this page and draws the charts from data.ts. No live API.',
    tech: 'React and Tailwind on Vite. Bklit-style bars and a Kokonut-style result bento, restyled to the field-kit palette.',
  }},
]

export const NEXT = [
  'Swap the circle of Gaussians for a real tabular conditional, same holdout region score.',
  'Try a continuous condition that is not an angle, and keep the same ball metric.',
  'If the interpolation gap gets larger, raise the train budget before adding a bigger architecture.',
]

export const SECTORS = [
  {{ name: 'simulation', why: 'Draw extra samples from a fitted conditional without rerunning the simulator.' }},
  {{ name: 'anomaly regions', why: 'Ask whether generated points land where the physics says they should.' }},
  {{ name: 'tabular imputation', why: 'A small flow is a density you can sample under a known condition.' }},
]
"""
