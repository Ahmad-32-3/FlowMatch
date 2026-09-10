"""Train CFM, score holdout success_pct vs baseline, write metrics.json.

CLI: python scripts/run.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from flows import const
from flows.eval import evaluate, illustrative, print_report, write_data_ts, write_metrics


def main() -> None:
    try:
        import torch  # noqa: F401
    except ImportError as e:
        m = illustrative(str(e))
        write_metrics(m)
        print_report(m)
        print("ILLUSTRATIVE")
        print("BLOCKER:", m["blocker"])
        write_data_ts(m)
        return

    if const.EPOCHS > const.MAX_EPOCHS:
        sys.exit(f"HARD FAIL: epochs {const.EPOCHS} over cap {const.MAX_EPOCHS}")
    if const.N_PER_TRAIN * len(const.TRAIN_CONDS) > const.MAX_N:
        sys.exit("HARD FAIL: n_train over cap")

    m = evaluate(ablations=True)
    if not m["illustrative"] and m["success_pct"] < const.FLOOR_PCT:
        sys.exit(
            f"HARD FAIL: success_pct {m['success_pct']:.1f} < {const.FLOOR_PCT} "
            f"(baseline {m['baseline_pct']:.1f})"
        )
    write_metrics(m)
    print_report(m)
    write_data_ts(m)


if __name__ == "__main__":
    main()
