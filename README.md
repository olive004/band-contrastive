# Band-Supervised Contrastive Learning for Inverse RNA Circuit Design

**TL;DR**: Positives = samples within a scale-invariant label band (e.g., ±10% in log-space). We benchmark against CCL, RNC, ConR on simulator-only RNA tasks. This repo is reviewer-friendly: zero-credential install, config-driven, CPU-only sanity tests, and deterministic seeds.

## Quickstart (reviewer-friendly)

```bash
# 1) (Optional) create a fresh venv
python -m venv .venv && source .venv/bin/activate  # on Windows: .venv\Scripts\activate

# 2) Install package in editable mode
pip install -e .

# 3) Run a tiny CPU sanity test (finishes in ~10–30s on CPU)
pytest -q

# 4) Train a minimal toy run (vector CVAE + Band-SupCon on dummy data)
python scripts/train.py +experiment=exp/vector_cvae_band seed=0 device=cpu

# 5) Generate designs for a target y*
python scripts/generate.py --checkpoint=outputs/dummy.ckpt --target_y=0.75 --n=8

# 6) Evaluate hit-rate at ±10% (dummy eval)
python scripts/eval.py +checkpoint=outputs/dummy.ckpt eval=hitrate
```

> Note: This skeleton ships with **dummy dataset & simulator mocks** so reviewers can run everything without external tools. Swap in your real data + simulators by editing `configs/data/*` and `bandcon/sims/*`.

## Reproducibility
- All experiments are defined in `configs/exp/*`. Each paper table row should correspond to a single config file.
- Deterministic seeding via `bandcon/utils/hydra_utils.py`.
- A CPU-only CI workflow (`.github/workflows/ci.yaml`) runs lint + tests.

## Structure
See `docs/` and the package `bandcon/` for modules. Stubs include docstrings and TODOs guiding implementation.
