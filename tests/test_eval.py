import numpy as np
import pytest

from flows import const
from flows.data import check_no_leak, check_split, in_region, make_split, success_pct, true_mean


def test_leak_injection_fails():
    split = make_split(n_per=8, seed=1)
    leaked = np.concatenate([split["train_conds"], split["hold_conds"]])
    with pytest.raises(ValueError, match="leak"):
        check_no_leak(leaked, split["hold_conds"])


def test_id_leak_injection_fails():
    split = make_split(n_per=8, seed=2)
    leaked = np.concatenate([split["train_ids"], split["hold_ids"]])
    with pytest.raises(ValueError, match="leak"):
        check_split(leaked, split["hold_ids"])


def test_holdout_conds_are_disjoint():
    split = make_split(n_per=8, seed=0)
    check_no_leak(split["train_conds"], split["hold_conds"])
    check_split(split["train_ids"], split["hold_ids"])
    assert set(split["hold_conds"].tolist()) == set(const.HOLD_CONDS)
    assert len(split["x"]) == 8 * len(split["train_conds"])


def test_true_draws_hit_their_balls():
    split = make_split(n_per=40, seed=3)
    assert success_pct(split["x"], split["c"]) >= 95


def test_wrong_condition_misses_the_ball():
    mu = true_mean(0)
    x = np.tile(mu, (20, 1))
    assert in_region(x, np.zeros(20, dtype=int)).all()
    assert not in_region(x, np.full(20, 4, dtype=int)).any()


def test_caps_match_design():
    assert const.N_PER_TRAIN * len(const.TRAIN_CONDS) <= const.MAX_N
    assert const.EPOCHS <= const.MAX_EPOCHS
    assert const.N_DIM == 2
    assert const.FLOOR_PCT == 85.0
    assert const.SEED == 0


def test_evaluate_beats_baseline_and_clears_floor():
    pytest.importorskip("torch")
    from flows.data import make_split
    from flows.eval import evaluate

    split = make_split(seed=0)
    m = evaluate(split, n_sample=64, seed=0)
    assert m["success_pct"] >= const.FLOOR_PCT
    assert m["success_pct"] > m["baseline_pct"]
    assert m["n_train"] <= const.MAX_N
    assert set(m["hold_conds"]).isdisjoint(set(m["train_conds"]))


def test_ablation_same_split_has_columns():
    pytest.importorskip("torch")
    from flows.eval import run_ablations

    split = make_split(seed=0)
    again = make_split(seed=0)
    assert np.array_equal(split["x"], again["x"])
    rows = run_ablations(split, seed=0, epochs=8)
    tags = [r["tag"] for r in rows]
    assert any(t.startswith("steps=") for t in tags)
    assert any(t.startswith("width=") for t in tags)
    assert any(t.startswith("cond_dim=") for t in tags)
    assert len({r["baseline_pct"] for r in rows}) == 1
    assert all("success_pct" in r for r in rows)
