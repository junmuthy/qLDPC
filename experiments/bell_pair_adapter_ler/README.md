# Generic CSS Bell-Pair Adapter LER

This experiment measures the decoded logical Pauli-product measurement failure rate of the
Bell-pair adapter between two independently prepared binary stabilizer CSS code blocks.

The reusable construction accepts any two compatible `CSSCode` objects, either logical `XX` or
logical `ZZ`, and an explicit logical index for each block. Surface-3/Steane remains the default
command-line preset, but it is not a dependency of the circuit builder.

## Circuit and measured quantity

The intrinsic experiment has the following phase boundary:

```text
ideal independent block preparation and syndrome rounds
                              ↓
noisy κ/bridge initialization, merged rounds, and detachment
                              ↓
                 ideal diagnostic closure
```

The retained observable is `observable 0`, the final merged-round product of the χ-check outcomes.
With both inputs prepared in the `+1` eigenstate of the measured logical Pauli, the reported rate is

```text
Pr(decoded logical PPM result is wrong | clean encoded inputs).
```

It is obtained directly from decoded observable failures. No code-memory LER is subtracted.

## Noise scopes

| Scope | Preparation | Adapter initialization, surgery, detachment | Terminal closure |
|---|---:|---:|---:|
| `intrinsic` | Ideal | Noisy | Ideal |
| `operational` | Ideal | Noisy | Noisy |
| `end-to-end` | Noisy | Noisy | Noisy |

Errors and idling on original code-block data during the adapter window are part of the adapter
error rate. Bell-pair infidelity is applied only to Bell pairs prepared during merged rounds.

## Generic Python API

```python
from qldpc import codes
from qldpc.objects import Pauli

from experiments.bell_pair_adapter_ler.run_experiment import build_experiment_for_codes

experiment = build_experiment_for_codes(
    left_code=codes.IcebergCode(4),
    right_code=codes.SteaneCode(),
    left_logical_index=1,
    right_logical_index=0,
    basis=Pauli.X,
    preparation_rounds=3,
    surgery_rounds=5,
    noise_scope="intrinsic",
    physical_error_rate=0.001,
)
```

The inputs must be binary, non-subsystem CSS codes. The two arguments always describe separate
physical blocks, even when they reference the same `CSSCode` object.

## Surface-3/Steane preset

Build the default `XX`/`ZZ`, three-/five-surgery-round matrix without sampling:

```bash
PYTHONPATH=src NUMBA_CACHE_DIR=/tmp/qldpc-numba-cache \
  ../.venv/bin/python experiments/bell_pair_adapter_ler/run_experiment.py \
  --preparation-rounds 3 --noise-scope intrinsic --build-only
```

Run a short pilot:

```bash
PYTHONPATH=src NUMBA_CACHE_DIR=/tmp/qldpc-numba-cache \
  ../.venv/bin/python experiments/bell_pair_adapter_ler/run_experiment.py \
  --preparation-rounds 3 --rounds 3 5 --noise-scope intrinsic \
  --max-shots 10000 --max-errors 50 --num-workers 4
```

Artifacts are written under `outputs/`:

- `circuits/*.stim`: phase-aware noisy Stim circuits.
- `dems/*.dem`: detector error models.
- `manifest.json`: code parameters, logical indices, phase lengths, and noise scope.
- `sinter_progress.csv`: resumable Sinter data.
- `summary.csv`: decoded LER with a two-sided 95% Wilson interval.

Artifact names include preparation rounds, surgery rounds, and noise scope, so new intrinsic runs do
not overwrite the earlier direct-merge pilot files.
