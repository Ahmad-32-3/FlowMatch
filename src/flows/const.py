# Frozen protocol. Change only with one ADR line in DESIGN.md, then re-measure.

N_DIM = 2
N_CONDS = 8
# Held-out conditions sit between trained neighbors (45 deg) so CFM can interpolate.
HOLD_CONDS = (2, 6)
TRAIN_CONDS = tuple(k for k in range(N_CONDS) if k not in HOLD_CONDS)
RADIUS = 2.5
SIGMA = 0.22
TAU = 0.70  # ball radius around true mean; ~3.2 sigma

N_PER_TRAIN = 160
N_SAMPLE = 128  # samples per holdout condition
MAX_N = 10_000
MAX_EPOCHS = 40
EPOCHS = 40
BATCH = 64
LR = 2e-3
SEED = 0
WIDTH = 64
STEPS = 16  # Euler steps at sample time
COND_DIM = 2  # cos/sin of condition angle
FLOOR_PCT = 85.0

METRICS_PATH = "metrics.json"
DATA_TS = "web/src/data.ts"
