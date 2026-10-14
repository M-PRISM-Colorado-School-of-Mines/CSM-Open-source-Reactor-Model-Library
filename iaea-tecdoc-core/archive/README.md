# archive/

Retired drivers kept for reference — do not use for new runs.

- `run_allin.py` — single all-in (f=0) eigenvalue run. Fully superseded by the
  central driver: `python model/core.py --insertion 100 --particles 50000
  --batches 150 --inactive 50`. Archived 2026-07-06 during the pre-flight fix
  pass; it also predates core.build_model() and re-derived its own
  geometry/settings without tallies.
