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

## Distance metadata

Exact code distance is not needed to build the circuit or detector error model, decode samples, or
estimate the LER. By default, manifests therefore record cached or analytically known distances and
use `null` for unknown distances:

```json
{
  "joint_code_parameters": [136, 18, null],
  "joint_code_distance_status": "not_computed"
}
```

Pass `compute_exact_distances=True` to the Python builders, or
`--compute-exact-distances` to the CLI, only when exact distances are explicitly required. This can
be exponentially more expensive than building and sampling the experiment for large codes.

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

## Bell-pair fidelity sweep

The fidelity sweep holds the Surface-3/Steane adapter, physical error rate, and phase lengths fixed
while varying the independent Werner-state Bell-pair infidelity `p_B = 1 - F_B`. The default grid
is logarithmically spaced from `p_B = 10^-5` through `3e-2` and includes the ideal `p_B = 0`
reference. Four surgery rounds consume 20 Bell pairs: five crossing checks in each round.

Run the 20,000-shot pilot:

```bash
PYTHONPATH=src:. NUMBA_CACHE_DIR=/tmp/qldpc-numba-cache \
  MPLCONFIGDIR=/tmp/qldpc-matplotlib \
  ../.venv/bin/python experiments/bell_pair_adapter_ler/run_fidelity_sweep.py \
  --max-shots 20000 --max-errors 400 --num-workers 12
```

Re-running the same command resumes its own `sinter_progress.csv`. The optional `--existing-data`
argument accepts additional Sinter files with exactly matching task identities. Older result files
whose metadata schema differs are deliberately excluded instead of being silently mixed into the
summary.

The sweep writes the standard circuit manifest and Sinter files plus:

- `run_config.json`: the grid, sampling limits, and reused input data.
- `fidelity_response.csv`: absolute LER, ideal-pair baseline, excess LER, and total Bell exposure.
- `fidelity_requirements.csv`: monotone point-estimate and conservative 95%-interval brackets for
  10% and 25% allowable LER overhead.
- `adapter_ler_vs_bell_fidelity.{pdf,png}`: absolute LER with Wilson 95% intervals.
- `adapter_excess_ler_vs_bell_fidelity.{pdf,png}`: excess over the ideal-pair baseline.

The scalar-fidelity model assumes equal residual `X`, `Y`, and `Z` probabilities on one Bell half.
For a particular distillation protocol, fidelity alone is sufficient only when its output is well
approximated by this Werner model. Biased or correlated distillation outputs require their complete
residual error model for a quantitative adapter prediction.

## Noise restricted to adapter hardware

The `adapter_noise_targets` option selects **where** physical noise is applied during
adapter initialization, surgery rounds, and detachment. It is independent of `noise_scope`,
which selects **when** preparation and diagnostic closure receive noise.

- `full_surgery` is the existing default: all eligible operations in the adapter window
  receive noise. Default circuits and Sinter task metadata retain their previous identities.
- `adapter_hardware` selects added gadget/bridge data, every check ancilla whose merged
  stabilizer touches that data (including extended original block checks), and auxiliary
  surgery ancillas, including the extra Bell halves. Selection uses CSS check matrices and
  circuit IDs; it does not depend on a surface, Steane, or HGP layout.

For `adapter_hardware`, one-qubit gates, resets and measurements on the selected hardware
are noisy. A two-qubit gate is noisy whenever **either** operand belongs to that hardware;
the full two-qubit depolarizing channel acts on both operands, including block data.
Block-only operations are ideal, but they still propagate faults introduced by the adapter.
Optional idling noise applies only to selected hardware. The explicit Bell source channel
is preserved independently of the physical gate-noise selection.

Use `noise_scope="intrinsic"` for ideal incoming blocks and ideal terminal closure.
Using `operational` or `end-to-end` additionally enables the corresponding phase noise;
those combinations no longer measure errors solely due to adapter hardware.

`reduce_detectors=True` (CLI `--reduce-detectors`) removes only detector events absent
from all nonzero-probability fault terms in the complete circuit's detector error model.
Stim first verifies that the detectors are deterministic without noise. All fault-reachable
detectors are retained, including distant block syndromes and the input/output boundaries.
Logical-only errors and observable 0, the last-round PPM result, are retained. The manifest
records the original-to-reduced detector mapping as `retained_detector_ids` (new index maps
to the original index), hardware qubits, and full/reduced sizes.

This optimization reduces decoder input rows; it does **not** remove block qubits, measurements,
or ideal gates. Removing inactive detector events is safe even when the individual stabilizer
measurements used to form them are random. Physical deletion of the blocks would require a
separate equivalence argument. The quantum circuit and its logical boundary conditions remain
the reference model. The statistical model is preserved by row removal, but a heuristic
decoder's tie-breaking or iteration behavior should also be checked on paired samples.

For example, a resumable four-round Surface/Steane run is:

```bash
PYTHONPATH=src:. NUMBA_CACHE_DIR=/tmp/qldpc-numba-cache \
  MPLCONFIGDIR=/tmp/qldpc-matplotlib \
  ../.venv/bin/python -m experiments.bell_pair_adapter_ler.run_experiment \
  --preparation-rounds 3 --rounds 4 --physical-error-rate 0.001 \
  --bell-pair-infidelity 0 --noise-scope intrinsic \
  --adapter-noise-targets adapter_hardware --reduce-detectors \
  --max-shots 10000 --max-errors 200 --num-workers 12 \
  --output-dir experiments/bell_pair_adapter_ler/outputs/hardware_prep3_surg4_bell0_10k
```

The same flags are available in `run_fidelity_sweep.py`, and both options are accepted by
`build_experiment_for_codes` for arbitrary binary stabilizer CSS inputs. New modes use distinct
artifact names and default output directories. Writing into a directory whose manifest describes
a different experiment raises an error. Analyze fidelity sweeps for each noise policy separately.

A small comparison pilot and paired timing benchmark can be run with:

```bash
PYTHONPATH=src:. NUMBA_CACHE_DIR=/tmp/qldpc-numba-cache \
  MPLCONFIGDIR=/tmp/qldpc-matplotlib \
  ../.venv/bin/python -m experiments.bell_pair_adapter_ler.run_hardware_comparison \
  --shots 1000 --rounds 4 --preparation-rounds 3 \
  --output-dir experiments/bell_pair_adapter_ler/outputs/adapter_hardware_prep3_surg4_pilot_1000
```

This pilot writes a manifest, configuration (including seed and package versions), summary,
`timing.csv`, and `adapter_hardware_comparison.{pdf,png}`. It requires an empty output directory
and is not resumable; use the Sinter runner above for larger collections. Hardware rows with
and without detector reduction use **the same shots** and must not be pooled as independent
data. The full-surgery noise baseline uses independent shots. The timing table records paired
prediction disagreements; reduced sampling time measures conversion of reused measurement data,
not an independent sampling speedup. Small pilot failure counts are for validation and rough
comparison, not precision LER estimates.

## Matched hardware-only round sweeps and timing

The `adapter_hardware_prep3_round_sweep_1_5_comparison` output directory compares the
new hardware-only model against both archived intrinsic round sweeps. Both use three ideal
preparation rounds, one through five adapter rounds, physical noise `p=0.001`, ideal closure,
no idling, and logical `XX`/`ZZ`. The two Bell-source settings are infidelity `0` and `0.001`
(fidelity `1` and `0.999`). Production limits are 10,000 shots or 200 failures per point;
Sinter batching can overshoot the failure limit.

To reproduce or resume the new production data, run this for each `INFIDELITY`/`SLUG` pair
(`0`/`bell0` and `0.001`/`bell0p001`), replacing the two placeholders:

```bash
PYTHONPATH=src:. NUMBA_CACHE_DIR=/tmp/qldpc-numba-cache \
  MPLCONFIGDIR=/tmp/qldpc-matplotlib OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  ../.venv/bin/python -m experiments.bell_pair_adapter_ler.run_experiment \
  --preparation-rounds 3 --rounds 1 2 3 4 5 --physical-error-rate 0.001 \
  --bell-pair-infidelity INFIDELITY --noise-scope intrinsic \
  --adapter-noise-targets adapter_hardware --reduce-detectors \
  --max-shots 10000 --max-errors 200 --num-workers 4 \
  --output-dir experiments/bell_pair_adapter_ler/outputs/adapter_hardware_prep3_round_sweep_1_5_comparison/SLUG
```

The archived runs used Sinter's automatic DEM analysis, which attempts graphlike decomposition.
The new hardware modes pass an explicit undecomposed DEM. Although the noise distribution can
be equivalent, decomposition suggestions change the error-column representation seen by the
current BP+OSD wrapper. To avoid confounding representation and noise policy, the comparison
also includes a fresh full-surgery reference with an explicit undecomposed DEM, saved under
`full_surgery_exact`. The archived data remain a separately labeled curve.

After hardware production sampling finishes, collect that reference and build the comparison:

```bash
PYTHONPATH=src:. NUMBA_CACHE_DIR=/tmp/qldpc-numba-cache \
  MPLCONFIGDIR=/tmp/qldpc-matplotlib OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  ../.venv/bin/python -m experiments.bell_pair_adapter_ler.compare_hardware_round_sweep \
  --sample-matched-baseline --num-workers 8
```

Once no production sampling is running, run the serial timing benchmark:

```bash
PYTHONPATH=src:. NUMBA_CACHE_DIR=/tmp/qldpc-numba-cache \
  MPLCONFIGDIR=/tmp/qldpc-matplotlib OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  ../.venv/bin/python -m experiments.bell_pair_adapter_ler.compare_hardware_round_sweep \
  --benchmark --benchmark-shots 256 --benchmark-repeats 3
```

Omit `--benchmark` to regenerate plots from existing data. The benchmark checkpoints each
completed basis/round/Bell setting and can resume with the same configuration. It verifies that
fresh full-surgery circuits match the archived circuits to floating serialization precision.
Policy execution order rotates between
repetitions. Timings report median and min–max range, not statistical confidence intervals.

`comparison.csv` records LER estimates, 95% Wilson intervals, and source summaries. The old
and new Sinter worker times, normalized per 10,000 shots, are included for reference but are
not a controlled speed benchmark. `timing_samples.csv` and `timing_summary.csv` record fresh
single-process BP+OSD measurements for full surgery, hardware noise with all detectors, and
hardware noise with reduced detectors. Circuit building, decoder compilation, measurement
sampling, conversion, wall-clock decoding, and CPU decoding are separated. Build/compile
values are per-condition setup costs, repeated in each batch row, not additional work per batch.
The reduced model reuses the full hardware model's sampled measurement records and sampling
time; no quantum-sampling speedup is claimed. Timing-shot failures are not pooled into LER data.

Changing the noise policy changes the measured experiment. Only the two hardware variants
isolate the effect of detector reduction. A smaller noise model does not guarantee a smaller
observed failure rate with the default heuristic BP+OSD decoder. Nonmonotonic round dependence
or worse hardware-only values should be investigated, not treated as a demonstrated hardware
benefit or evidence that detector reduction preserves the original full-noise LER.

## HGP simplex–surface hardware comparison

`run_hgp_hardware_comparison.py` matches the archived `m=3` HGP simplex–surface experiment:
HGP `[[98,18,4]]` on the left, unrotated distance-4 surface on the right, logical indices zero,
four ideal preparation rounds and four adapter rounds, `p=0.001`, perfect Bell pairs, no idling,
and ideal terminal closure. Both `XX` and `ZZ` are sampled with hardware-only noise and
inactive detector reduction. Exact distance searches remain disabled.

```bash
PYTHONPATH=src:. NUMBA_CACHE_DIR=/tmp/qldpc-numba-cache \
  MPLCONFIGDIR=/tmp/qldpc-matplotlib OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  ../.venv/bin/python -m experiments.bell_pair_adapter_ler.run_hgp_hardware_comparison \
  --max-shots 10000 --num-workers 8
```

This resumes its own Sinter checkpoints and stops at 10,000 shots per basis, without an
earlier failure-count limit. Its default output directory is
`outputs/hgp_simplex3_surface_unrotated_d4_hardware_prep4_surg4_bell0_10k/`, separate from
the archived full-surgery data. `--build-only` performs construction and baseline validation
without collecting shots; `--analyze-only` regenerates the comparison from saved summaries.

The runner verifies the canonical full-surgery circuits against the saved archive and checks
that Sinter's legacy automatic detector model agrees with the explicit undecomposed model.
For this HGP case the check succeeds in both bases; a new expensive full-surgery reference
is therefore unnecessary. This audit requires the archived circuit files, which are local
generated artifacts, and is saved in `baseline_audit.json`. A mismatch raises an error.

The comparison CSV and PDF/PNG report independent LER estimates with 95% Wilson intervals.
Historical and new worker times, normalized per 10,000 shots, are included as workload
records, not as a controlled runtime benchmark. `invocations.jsonl` records wall-clock time
for each completed invocation (including build/analysis; a resumed invocation measures only
that invocation, not all previous work). The original data and its checkpoints are not modified.
