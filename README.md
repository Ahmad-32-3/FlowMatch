# FlowMatch

I teach a small generative model to draw points from a known toy cloud given a condition code. Then I give it conditions it did not train on and ask whether the new points land where the true cloud says they should.

Scoring how well it fit the training points is only a debug check. The number I trust is that holdout hit rate, next to a naive random cloud in the same plane.

## Run

```bash
python -m pytest tests/ -q
python scripts/run.py
npm --prefix web install
npm --prefix web run dev
```

`scripts/run.py` prints flow holdout success next to the naive baseline. If torch is blocked, leak tests and an illustrative walkthrough still ship.

## Layout

- `src/` synthetic density, CFM train, eval
- `scripts/run.py`
- `tests/` leak injection
- `web/` case-study page
