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

| Scope         | Preparation | Adapter initialization, surgery, detachment | Terminal closure |
|---------------|------------:|--------------------------------------------:|-----------------:|
| `intrinsic`   | Ideal       | Noisy                                       | Ideal            |
| `operational` | Ideal       | Noisy                                       | Noisy            |
| `end-to-end`  | Noisy       | Noisy                                       | Noisy            |

Errors and idling on original code-block data during the adapter window are part of the adapter
error rate. Bell-pair infidelity is applied only to Bell pairs prepared during merged rounds.

## Generic Python API

```python
from qldpc import codes
from qldpc.objects import Pauli

from experiments.bell_pair_adapter_ler.common.run_experiment import build_experiment_for_codes

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

## PPM-basis-first version for future hardware tests

Use `schedule_policy="ppm_basis_first"` in `build_experiment_for_codes`,
`build_experiment`, `build_experiment_matrix`, or `build_fidelity_sweep_matrix`.
The versioned identity is `ppm_basis_first_local_v1`:

| Logical PPM | Local check-family order | Bell-assisted check-family order |
|-------------|--------------------------|----------------------------------|
| `XX`        | `X→Z`                    | `X→Z`                            |
| `ZZ`        | `Z→X`                    | `X→Z`                            |

Only merged-round **local family order** changes. Independent preparation,
within-family gate layers, adapter geometry, Bell partners, detachment, and
terminal closure are unchanged. This is not the exact-dual construction, nor a
claim that the rule is optimal. Existing builders still default to the legacy
schedule and retain their old checkpoint identities.

For any pair of compatible binary stabilizer CSS codes:

```python
experiment = build_experiment_for_codes(
    left_code=left_code,
    right_code=right_code,
    basis=Pauli.Z,
    preparation_rounds=4,
    surgery_rounds=4,
    physical_error_rate=0.001,
    bell_pair_infidelity=0.0,
    noise_scope="intrinsic",
    adapter_noise_targets="adapter_hardware",
    reduce_detectors=True,
    schedule_policy="ppm_basis_first",
)
```

At the lower-level circuit API, use `JointPPMSchedule.ppm_basis_first(basis)` as
the merged-round syndrome strategy's schedule. The generic builder rejects
combining a named nonlegacy policy with a custom `surgery_schedule`.

The named policy uses an explicitly undecomposed detector error model (DEM),
recorded in the manifest and passed unchanged to BP+OSD. Its names and task
identities are distinct from legacy runs, **even for XX**, whose circuit is
unchanged. Old shots are not automatically pooled. The original full-surgery
default retains Sinter's automatic DEM analysis; comparing to that archive changes
decoder representation as well as any chosen noise policy.

The dedicated runner below defaults to intrinsic **hardware-only** noise and
inactive-detector reduction. The code blocks and their ideal operations remain
in the circuit; this is not removal of the blocks or an equivalent acceleration
of full-surgery noise. Exact-distance searches and idling noise stay disabled.

Start with an HGP-simplex `m=3` / unrotated surface `d=4` pilot:

```bash
PYTHONPATH=src:. NUMBA_CACHE_DIR=/tmp/qldpc-numba-cache \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  ../.venv/bin/python -m experiments.bell_pair_adapter_ler.pilots.run_ppm_basis_first \
  --code-pair hgp-simplex-surface --simplex-m 3 --surface-distance 4 \
  --preparation-rounds 4 --rounds 4 --max-shots 1000 --num-workers 4 \
  --output-dir experiments/bell_pair_adapter_ler/outputs/hgp_m3_surface_d4_ppm_first_pilot
```

Repeat the same command to resume, or increase `--max-shots` to extend the
**cumulative per-condition** cap. Changing the codes, rounds, logical indices,
noise, circuit, or DEM requires a new output directory. Every invocation is
logged; resumable batches are saved in `sinter_progress.csv`. Ctrl-C saves any
already-flushed batches; an in-flight batch may be lost. No old checkpoints are
imported automatically.

For a larger construction, use `--simplex-m 4 --surface-distance 8` and a new
output directory. Start with `--build-only`, then a small shot cap before
committing to hours of decoding. The surface distance defaults to `2**(m-1)`
if omitted for the HGP preset. `--code-pair surface-steane` selects the original
rotated Surface-3/Steane pair. Both bases are sampled unless `--bases X` or
`--bases Z` is specified. Explicit logical indices are also supported.

Timing is recorded in `invocations.jsonl`, separately from task identity:

- Code construction wall time and each circuit/DEM build wall time.
- Sampling/decoding collection wall time, including worker startup and decoder
  compilation; this is not pure decoder time or per-condition elapsed time.
- Newly collected shots and worker seconds for this invocation only, plus
  cumulative counts. `summary.csv` contains cumulative per-condition worker
  seconds, LER, Wilson intervals, and explicit local/Bell orders.
- Worker count, thread settings, Python/platform, and dependency versions.

For a timing pilot, add `--profile --max-batch-size 128 --max-batch-seconds 5`.
Profiling requires the optional `psutil` package and writes newly flushed batch
statistics to `batch_samples.jsonl`, plus one-second parent/descendant RSS samples
to `resource_samples.jsonl`. Each record identifies its invocation. The sampled
sum of process RSS may double-count shared pages and miss short peaks; it is not
unique physical memory. The batch-time target does not interrupt a slow shot.
Profiling and collection limits do not change the circuit/decoder task identity.

Do not interpret summed worker seconds as elapsed time, or divide resumed
cumulative shots by the latest invocation's time. Small pilots can be dominated
by startup and rare expensive decoder cases; they are not reliable speedup or
low-LER estimates. Keep hardware, worker count, shot budget, and noise/decoder
policies matched for future speed comparisons. This scheduling rule by itself
does not promise a speedup.

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
  ../.venv/bin/python -m experiments.bell_pair_adapter_ler.common.run_experiment \
  --preparation-rounds 3 --noise-scope intrinsic --build-only
```

Run a short pilot:

```bash
PYTHONPATH=src NUMBA_CACHE_DIR=/tmp/qldpc-numba-cache \
  ../.venv/bin/python -m experiments.bell_pair_adapter_ler.common.run_experiment \
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
  ../.venv/bin/python -m experiments.bell_pair_adapter_ler.bell_fidelity.run_fidelity_sweep \
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
  ../.venv/bin/python -m experiments.bell_pair_adapter_ler.common.run_experiment \
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
  ../.venv/bin/python -m experiments.bell_pair_adapter_ler.comparisons.run_hardware_comparison \
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
  ../.venv/bin/python -m experiments.bell_pair_adapter_ler.common.run_experiment \
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
  ../.venv/bin/python -m experiments.bell_pair_adapter_ler.comparisons.compare_hardware_round_sweep \
  --sample-matched-baseline --num-workers 8
```

Once no production sampling is running, run the serial timing benchmark:

```bash
PYTHONPATH=src:. NUMBA_CACHE_DIR=/tmp/qldpc-numba-cache \
  MPLCONFIGDIR=/tmp/qldpc-matplotlib OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  ../.venv/bin/python -m experiments.bell_pair_adapter_ler.comparisons.compare_hardware_round_sweep \
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
  ../.venv/bin/python -m experiments.bell_pair_adapter_ler.comparisons.run_hgp_hardware_comparison \
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

## Exact X/Z-dual hardware schedule

`run_hgp_dual_schedule.py` is a separate opt-in hardware-only experiment. It derives ZZ from
the complete reference XX circuit, preserving gate order, Bell-check order, ideal preparation
and closure, measurement-record order, and detector/observable definitions. It does not insert
extra physical Hadamard gates or change the default surgery schedule.

The generic `CSSXZDuality` helper validates a supplied data-column permutation against every
opposite-basis stabilizer row and matches check ancillas and Bell partners. The HGP/surface
experiment supplies its explicit block-transposition mapping and additionally validates the
chosen logical support and the adapter-only noise mask. It transports basis-dependent resets,
measurements, controlled gates, and Pauli noise together. Unsupported operations or invalid
maps raise errors. This does not assume that arbitrary CSS codes are self-dual: the helper
requires an explicit verified correspondence. Annotation coordinates follow the transported
reference layout rather than the legacy ZZ row order.

Before sampling, the runner requires the ordered dual ZZ detector-error model to equal the
archived hardware XX model exactly, including probabilities and detector/error-column order.
This is a symmetry control, not an optimization claim. With these noise and decoder settings,
XX and dual ZZ have the same predicted decoded error statistics; independent shot samples need
not yield identical counts. The hardware-only test does not explain the archived full-surgery
ZZ excess, since block-only faults remain disabled.

```bash
PYTHONPATH=src:. NUMBA_CACHE_DIR=/tmp/qldpc-numba-cache MPLBACKEND=Agg \
  MPLCONFIGDIR=/tmp/qldpc-matplotlib OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  ../.venv/bin/python -m experiments.bell_pair_adapter_ler.schedules.run_hgp_dual_schedule \
  --max-shots 20000 --pilot-shots 1000 --num-workers 8
```

Only the new ZZ task is sampled. The 1,000-shot pilot is resumed to 20,000 **total** shots,
not added to a separate 20,000-shot estimate. Existing hardware XX/ZZ 20,000-shot results
are read-only references. No full-surgery experiment or exact-distance search is built.
`--build-only` performs validation without production sampling; `--analyze-only` regenerates
the comparison. The default directory is
`outputs/hgp_simplex3_surface_unrotated_d4_hardware_xz_dual_prep4_surg4_bell0_20k/`.

New schedule metadata is canonically ordered and has its own task identity. The saved-artifact
round-trip identity is tested to avoid restarting from zero due to metadata ordering. A
directory containing another experiment is rejected. `schedule_audit.json` records the full
qubit map, reference hash, logical/mask checks, DEM equality, and task ID. The comparison
PDF/PNG shows archived XX, archived ZZ, and new dual ZZ with 95% Wilson intervals;
`comparison.csv` also records normalized worker time. These historical timings are not a
controlled speed benchmark. `invocations.jsonl` records completed pilot/production stage wall
times and cumulative invocation time; pilot and final cumulative estimates are correlated.

## One-factor HGP hardware schedule tests

`run_hgp_schedule_ablations.py` compares three independent changes to each legacy basis:

- A, `bell_order`: Bell-assisted Z checks before X checks, with partner ownership and
  within-family/within-check ordering fixed.
- B, `local_order`: local Z interaction layers before X layers; Bell extraction unchanged.
- C, `local_layers`: local layers mapped from the *opposite legacy basis*, with the target's
  legacy X-then-Z family order restored. Bell extraction is untouched. This replaces local
  coupling order and parallel grouping together and can change depth.

These are **not cumulative changes**. Four ideal preparation rounds, adapter initialization,
four noisy adapter rounds, detachment, and ideal terminal readout retain the legacy protocol.
The blocks are HGP simplex [[98,18,4]] and unrotated surface distance 4, logical index zero,
hardware-only noise at p=0.001, Bell F=1, and no idling noise. No exact distances are searched.

`JointPPMSchedule` exposes independent family-order and validated local-layer controls.
Defaults reproduce the archived hardware circuits/DEMs exactly. Replacement layers must
preserve every local check-data gate and have no within-layer qubit conflicts. Bell partner
allocation depends on check identity, not extraction position. Preparation does not receive
the surgery scheduling controls. The experiment runner checks full gate/noise inventories,
ideal preparation, physical detector/observable identities (including their canonical order),
and noiseless correctness. Only measurement references are regenerated; records are not
transported wholesale from the opposite basis. The DEM and inactive-detector selection are
recomputed with the legacy policy, so their sizes can legitimately change.

```bash
PYTHONPATH=src:. NUMBA_CACHE_DIR=/tmp/qldpc-numba-cache MPLBACKEND=Agg \
  MPLCONFIGDIR=/tmp/qldpc-matplotlib OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  ../.venv/bin/python -m experiments.bell_pair_adapter_ler.schedules.run_hgp_schedule_ablations \
  --max-shots 20000 --pilot-shots 1000 --num-workers 8
```

The default new directory is `outputs/hgp_simplex3_surface_unrotated_d4_hardware_schedule_ablations_20k/`.
Each of six tasks has a separate `Z/` or `X/` policy subdirectory and checkpoint. The pilot
is included in the 20,000-shot total. ZZ runs precede XX. No error-count stopping rule truncates
the planned shot total. Nested task metadata is recursively canonicalized for restart safety.
An exactly equal ordered DEM triggers a stop before sampling so reuse cannot be silently
misreported as new independent data. `--build-only` validates; `--analyze-only` plots saved results.

`comparison.csv` and the PDF/PNG show both legacy baselines, the six one-factor variants, and
the previous combined-control exact-dual ZZ reference. `differences.csv` reports pointwise
Newcombe/Wilson 95% intervals on differences from the corresponding legacy baseline, two-sided
Fisher exact tests, and Holm adjustment over the six planned comparisons (valid with shared
baselines). Intermediate pilot plots are descriptive, not repeated significance stopping tests.
No equivalence or improvement is inferred merely from overlapping pointwise error bars.
Worker times and stage wall times are historical workload records, not controlled benchmarks.

## Surface–Steane local-order-only round sweep

`run_surface_local_order.py` tests the lighter PPM-basis-first local rule without
changing either adapter's geometry or logical supports. XX remains at legacy local
X-then-Z and is reused without new sampling. Only ZZ changes to local Z-then-X.
Bell-family order, Bell partners, within-check gates, the local gate layers themselves,
preparation, detachment, and terminal readout are unchanged. This is not an exact-dual
construction and does not require choosing equivalent logical representatives.

The ten new ZZ configurations span 1–5 adapter rounds and Bell fidelities 1 and 0.999.
All use rotated Surface-3 (left) and Steane (right), logical index zero, three ideal
preparation rounds, hardware-only p=0.001, no idling noise, ideal closure, existing
BP+OSD settings, and exact undecomposed DEM analysis with inactive-detector reduction.
The 20 legacy XX/ZZ hardware configurations in
`outputs/adapter_hardware_prep3_round_sweep_1_5_comparison/{bell0,bell0p001}/` are
read-only references; no HGP or full-surgery data are pooled into this study.

Before sampling, all 20 default **canonical serialized circuits** and ordered DEMs
must match their archives. The serialization qualification matters: Stim text rounds
floating-point gate arguments, so an in-memory gate argument need not have the same
last binary bit as its text round trip. Circuit text must match exactly and ordered
DEM equality is required independently, including error probabilities. The complete
archive files are hashed and rechecked before every analysis.

Each new configuration must preserve the full physical gate/noise inventory, ideal
preparation, canonical detector/observable meanings, Bell partner and gate sequences,
and complete tick depth. Additional checks require exactly local Z layers followed
by the original X layers, and an unchanged quantum detachment/readout suffix. Each
Bell fidelity is audited with its actual full noisy circuit. Noiseless sampling,
decoder smoke tests, and nested-metadata task-identity round trips are also tested.

```bash
PYTHONPATH=src:. NUMBA_CACHE_DIR=/tmp/qldpc-numba-cache MPLBACKEND=Agg \
  MPLCONFIGDIR=/tmp/qldpc-matplotlib OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  ../.venv/bin/python -m experiments.bell_pair_adapter_ler.schedules.run_surface_local_order \
  --max-shots 20000 --pilot-shots 1000 --num-workers 8
```

The new directory is `outputs/surface_steane_hardware_local_order_prep3_rounds1_5_20k/`.
Ten unique task identities share a resumable `sinter_progress.csv`. First all ten reach
1,000 pilot shots each, then the same tasks continue to 20,000 total shots each; pilot
counts must not be added again. XX is never passed to the sampler. `--build-only`
validates without sampling; `--analyze-only` regenerates the comparison without new shots.

The PDF/PNG plots LER versus rounds, separately for the two Bell fidelities, with three
curves: unchanged/reused legacy XX, legacy ZZ, and new local-order-only ZZ. Pointwise
95% Wilson intervals and upper-limit markers for zero failures are used.
`comparison.csv` includes actual shot/failure counts, depth, model sizes, and worker
time; `differences.csv` compares only new ZZ against the matching legacy ZZ using
Newcombe/Wilson intervals, nominal Fisher tests, and Holm adjustment across ten tests.
Legacy points stopped at 10,000 shots or about 200 failures, unlike the new fixed-shot
run. These nominal intervals/tests do not claim stopping-rule-adjusted coverage or
formal equivalence. Historical worker times are workload records, not controlled
benchmarks. All old results and their checkpoints remain unchanged.

## Reusable HGP exact-dual ZZ pilot and timing comparison

`run_hgp_xz_dual.py` generalizes the earlier m=3 exact-dual experiment without
changing it or the PPM-basis-first runner. Its default target is simplex HGP
`m=4` / unrotated surface `d=8`, logical index zero in each block, four ideal
independent preparation rounds and eight adapter rounds. It samples **only new
ZZ shots**. Existing matching-basis XX and ZZ results are read-only references;
they are never imported into the new checkpoint.

`hgp_xz_dual.py` derives candidate sector-transposition permutations from the
actual HGP seed dimensions and surface distance. It validates logical supports,
all opposite-basis check rows, Bell-partner ownership, physical block/adapter
partitions, and the hardware-only noise mask with `CSSXZDuality`. This preset
does not assume arbitrary CSS codes or arbitrary logical selections admit this
map: incompatible selections fail instead of choosing a different logical.
Exact distance searches remain disabled.

The entire XX circuit is transported into ZZ, including its ideal boundaries,
measurement-record references, controlled gates, and Pauli noise. No physical
Hadamards are inserted. Local extraction and Bell extraction both become `Z→X`,
and local gate layers are transported from XX. Metadata explicitly labels the
transported boundary and detector-order conventions; it does not inherit the
source XX's order labels. The ordered dual ZZ DEM must equal the source XX DEM
exactly, including error probabilities and column order. Both full and reduced
circuits must pass noiseless sampling. The source XX and native matching-basis
ZZ circuits/DEMs must also match the requested reference artifacts, and reference
checkpoint task identities are verified before collection.

The default pilot is **256 then 1,000 cumulative ZZ shots**. It stops at 1,000;
there is no implicit 20,000-shot production stage. Both stages use eight workers,
a batch-size cap of 32, a target batch/flush interval of five seconds, and
process/batch profiling. A slow shot or batch can exceed the interval; callback
records can aggregate multiple internal batches.

```bash
env PYTHONPATH=src:. NUMBA_CACHE_DIR=/tmp/qldpc-numba-cache \
  MPLCONFIGDIR=/tmp/qldpc-mpl MPLBACKEND=Agg \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  ../.venv/bin/python -m experiments.bell_pair_adapter_ler.schedules.run_hgp_xz_dual \
  --simplex-m 4 --surface-distance 8 --preparation-rounds 4 --rounds 8 \
  --physical-error-rate 0.001 --bell-pair-infidelity 0 \
  --pilot-shots 256 --max-shots 1000 --num-workers 8 \
  --reference-dir experiments/bell_pair_adapter_ler/outputs/ppm_basis_first_hgp_m4_d8_prep4_surg8_10000/samples \
  --output-dir experiments/bell_pair_adapter_ler/outputs/exact_xz_dual_hgp_m4_d8_prep4_surg8_pilot_1000
```

Add `--build-only` to validate and save artifacts without sampling, or
`--analyze-only` to rebuild/validate the same configuration and regenerate plots
from saved counts without sampling. On resume use the same command/directory;
`--max-shots` is cumulative, not additional. Worker counts and profiling controls
may change without changing the task identity. Changing the experiment metadata
or saved circuit/DEM is rejected. A POSIX advisory lock prevents concurrent
writers; the `.collection.lock` file remaining on disk does not mean a job is
active. Foreign checkpoints, including the source XX checkpoint, are rejected.

Artifacts include:

- `schedule_audit.json`: complete validated map, circuit/DEM and reference hashes,
  noiseless checks, and stable task identity.
- `sinter_progress.csv`: flushed, resumable ZZ counts. Interruptions preserve
  completed batches but can lose unfinished work. `summary.csv` updates at stage
  completion or handled interruption, not after every batch.
- `invocations.jsonl`, `batch_samples.jsonl`, `resource_samples.jsonl`: stage
  before/after counts, timestamps, work and wall time, and sampled process-tree RSS.
- `comparison.csv` and `hgp_xz_dual_comparison.pdf/.png`: the two archived
  matching-basis estimates plus the independent dual ZZ pilot, with counts and
  pointwise 95% Wilson intervals. With only a few failures, wide overlapping
  pilot intervals do not establish equivalence.
- `stage_timings.csv`: new/incremental shot counts, wall time, throughput,
  worker time per shot, and peak summed RSS for each new invocation.
- `timing_comparison.csv`, `timing_notes.json`, and `hgp_xz_dual_timing.pdf/.png`:
  comparison of profiled worker seconds per shot at the current pilot's worker
  count. For the m=4 eight-worker comparison, only the archived eight-worker
  9,000-new-shot-per-basis stage is used; earlier four-worker pilot shots are
  excluded from timing, but remain part of the 10,000-shot LER estimates.

Timing ratios are descriptive sampled work ratios, not controlled wall-time
speedups. The archived eight workers shared XX and ZZ, whereas this pilot runs
ZZ alone. The archived combined collection wall time is labelled as shared and
must not be added across basis rows or interpreted as ZZ-only elapsed time.
Worker seconds include sampling/decoding, not pure decoder CPU time; different
random faults, shot counts, startup/tail overhead, and laptop load can affect
timing. Summed RSS can double-count shared pages. Timing plots intentionally
have no statistical error bars inferred from aggregated batch records.

## Matching-basis hardware-only Bell-fidelity sweeps

`run_ppm_fidelity_sweep.py` reuses the generic CSS circuit builder and the current
matching-basis code presets. It does not invoke the exact-dual runner or search
for code distances. Default settings are HGP simplex m=3 / unrotated surface d=4,
four ideal preparation rounds, four adapter rounds, physical p=0.001, intrinsic
adapter-hardware noise, inactive-detector reduction, and no idling noise.
XX uses local X→Z, ZZ uses local Z→X, and both use Bell X→Z.

The nine default Bell source fidelities are 1, 0.99999, 0.99997, 0.9999, 0.9997,
0.999, 0.997, 0.99, and 0.97. Noise is an independent equal-X/Y/Z Pauli channel
on one Bell half with total probability 1-F. Physical gate noise stays fixed,
including at F=1. Both PPM bases give 18 conditions, sharing eight workers total.
This HGP construction consumes seven Bell pairs per round, or 28 per PPM.

Start the fixed-budget 1,000-shot-per-condition pilot:

```bash
env PYTHONPATH=src:. NUMBA_CACHE_DIR=/tmp/qldpc-numba-cache \
  MPLCONFIGDIR=/tmp/qldpc-mpl MPLBACKEND=Agg \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  ../.venv/bin/python -m experiments.bell_pair_adapter_ler.bell_fidelity.run_ppm_fidelity_sweep \
  --simplex-m 3 --surface-distance 4 --preparation-rounds 4 --rounds 4 \
  --physical-error-rate 0.001 --num-workers 8 --max-shots 1000 \
  --output-dir experiments/bell_pair_adapter_ler/outputs/ppm_basis_first_hgp_m3_d4_prep4_surg4_bell_fidelity
```

After reviewing pilot timing, repeat with `--max-shots 20000` to extend the same
checkpoints, retaining pilot samples. There is no early failure-count stop and
no automatic extension beyond the requested cumulative cap. `--build-only`
validates without sampling. `--analyze-only` regenerates a completed target's
report without sampling. The ideal F=1 point and at least one nonideal point
are required; use `--bell-pair-fidelities ...` or `--bell-pair-infidelities ...`
to specify another grid. A changed grid requires a new output directory.

Before collection, the runner checks that only explicit Bell-source probabilities
change: other physical operations, their noise, and the hardware mask stay fixed.
Each circuit passes a 32-shot noiseless detector/observable check. The source
channel can insert noise-only moments in the existing builder; the comparison
normalizes consecutive empty TICK markers after removing that channel. This is
safe here because idling noise is disabled; actual saved circuits are unmodified,
and this check makes no claim about physical adapter latency.

The output directory is protected by an advisory writer lock. Resuming verifies
the full manifest/grid, circuit/DEM contents, checkpoint strong IDs, dependency
versions and decoder source hashes. There is no import of archived shot counts.
Completed batches are checkpointed, with a 128-shot maximum and a target five-second
batch interval; slow in-flight batches are not forcibly interrupted. Interruptions
retain flushed samples. `summary.csv` updates on completion or handled interruption;
use `sinter_progress.csv` for live counts.

`invocations.jsonl`, `batch_samples.jsonl`, and `resource_samples.jsonl` record
incremental work, shared collection wall time, and process-tree RSS. Final plots,
summary counts, fidelity/excess tables, and timing are saved under `analysis_1000/`
and later `analysis_20000/`, so the pilot report is not overwritten by production.
Plots use fidelity labels with logarithmic spacing in infidelity. Zero-failure
points display upper endpoints of pointwise 95% Wilson intervals, not zero LER
on a logarithmic axis. Excess intervals use the existing envelope formed from
the point's and baseline's marginal Wilson bounds; these are not simultaneous
confidence guarantees over the sweep. Engineering requirement brackets are
exploratory and do not establish saturation from overlapping confidence bars.

The original Surface/Steane fidelity archive used full-surgery noise, not this
hardware-only mask. It is not pooled or overlaid as a matched code-size comparison.

## XX physical-noise / distance comparison

`run_physical_noise_comparison.py` compares HGP-simplex/surface m=3/r=4,
m=3/r=8, and m=4/r=8 at physical p=0.0003 and 0.0001, XX only, with one
four-worker pool at a time and a 1,000-shot-per-condition pilot target.
See [PHYSICAL_NOISE_COMPARISON.md](physical_noise/PHYSICAL_NOISE_COMPARISON.md) for controls,
checkpoint/resume commands, validation, timing, and interpretation. It retains
the production decoder and plots exact binomial intervals including upper bounds
for zero failures. Archived p=0.001 controls are read-only and never pooled into
the new samples; no 10k extension or exact-distance search starts automatically.

For the targeted continuation of just the two d=8/eight-round XX low-p points,
see [D8_TWO_NOISE_EXTENSION.md](physical_noise/continuations/D8_TWO_NOISE_EXTENSION.md). The runner adds
200,000 shots to each fixed seed, stopping at 500,000 (p=1e-4) and 300,000
(p=3e-4), with eight shared workers and a separate resumable checkpoint.

For 20,000 additional shots at p=0.001 for both eight-round XX distances, see
[HIGH_P_EIGHT_ROUND_EXTENSION.md](physical_noise/continuations/HIGH_P_EIGHT_ROUND_EXTENSION.md). The fixed-seed
continuation targets 40,000 total shots for d=4 and 30,000 for d=8, preserves the
latest low-p references, and reuses the same checkpoint/unequal-target safeguards.

The single-point continuation in [D8_SINGLE_POINT_EXTENSION.md](physical_noise/continuations/D8_SINGLE_POINT_EXTENSION.md)
adds 500,000 shots only to the d=8/eight-round XX p=1e-4 point, reaching one million
total. All other points retain their latest counts, including the extended high-p data.

`plot_d8_low_p_mean_history.py` reconstructs the cumulative XX LER at d=8,
eight rounds, p=1e-4 from the original checkpoint and its continuations. It verifies
and excludes imported seed rows, snapshots only complete live CSV lines, and writes
timestamped PDF/PNG/CSV reports under `outputs/ppm_basis_first_hgp_d8_xx_p1e4_surg8_mean_history/`.
This is saved-batch order, not per-shot or wall-clock history. The pointwise
one-sigma jackknife band is not a simultaneous confidence band, neighboring
cumulative estimates are correlated, and zero-failure prefixes have no band.
The script neither locks nor modifies the running collector.

For four fresh XX/eight-round conditions at the geometric midpoints of the
existing physical-noise intervals, see [LOG_MIDPOINT_SWEEP.md](physical_noise/LOG_MIDPOINT_SWEEP.md).
The runner uses one eight-worker pool, includes its 1,000-shot timing pilot in
the 100,000-shot-per-condition targets, and freezes the latest six reference points.

`run_d4_midpoint_extension.py` selectively adds 100,000 shots to only the d4,
eight-round, lower-midpoint XX point. The fixed 100k seed gives a resumable 200k
total, with eight workers and a separate checkpoint; the other nine comparison
points remain read-only. The continuation writes an updated data-only comparison.
