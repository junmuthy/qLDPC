# Command migration map

Run modules from the repository root with `PYTHONPATH=src:.`.
Every new module below is relative to `experiments.bell_pair_adapter_ler`.
For option-taking runners, use
`python -m experiments.bell_pair_adapter_ler.<module> --help`.
Helper-only modules have no CLI; import them from their new package instead.
One historical exception is `comparisons.plot_surface_all_datasets_bell1`: it
accepts no options and runs its plotting workflow immediately, even with `--help`.
Consult its source rather than invoking it just to inspect usage.

| Previous top-level file              | New module                                                   |
|--------------------------------------|--------------------------------------------------------------|
| `adapter_structure_audit.py`         | `fault_analysis.adapter_structure_audit`                     |
| `analyze_fidelity_sweep.py`          | `bell_fidelity.analyze_fidelity_sweep`                       |
| `collection_profile.py`              | `common.collection_profile`                                  |
| `compare_hardware_round_sweep.py`    | `comparisons.compare_hardware_round_sweep`                   |
| `dual_timing.py`                     | `schedules.dual_timing`                                      |
| `effective_distance_audit.py`        | `fault_analysis.effective_distance_audit`                    |
| `explain_distance_witnesses.py`      | `fault_analysis.explain_distance_witnesses`                  |
| `five_fault_search.py`               | `fault_analysis.five_fault_search`                           |
| `hgp_xz_dual.py`                     | `schedules.hgp_xz_dual`                                      |
| `paired_decoder_check.py`            | `fault_analysis.paired_decoder_check`                        |
| `plot_d8_low_p_mean_history.py`      | `physical_noise.plot_d8_low_p_mean_history`                  |
| `plot_hgp_failure_history.py`        | `comparisons.plot_hgp_failure_history`                       |
| `plot_hgp_round_comparison.py`       | `comparisons.plot_hgp_round_comparison`                      |
| `plot_surface_all_datasets_bell1.py` | `comparisons.plot_surface_all_datasets_bell1`                |
| `run_d4_midpoint_extension.py`       | `physical_noise.continuations.run_d4_midpoint_extension`     |
| `run_d4_r8_control.py`               | `physical_noise.run_d4_r8_control`                           |
| `run_d8_noise_extension.py`          | `physical_noise.continuations.run_d8_noise_extension`        |
| `run_d8_single_point_extension.py`   | `physical_noise.continuations.run_d8_single_point_extension` |
| `run_experiment.py`                  | `common.run_experiment`                                      |
| `run_fidelity_sweep.py`              | `bell_fidelity.run_fidelity_sweep`                           |
| `run_hardware_comparison.py`         | `comparisons.run_hardware_comparison`                        |
| `run_hgp_dual_schedule.py`           | `schedules.run_hgp_dual_schedule`                            |
| `run_hgp_hardware_comparison.py`     | `comparisons.run_hgp_hardware_comparison`                    |
| `run_hgp_schedule_ablations.py`      | `schedules.run_hgp_schedule_ablations`                       |
| `run_hgp_xz_dual.py`                 | `schedules.run_hgp_xz_dual`                                  |
| `run_high_p_extension.py`            | `physical_noise.continuations.run_high_p_extension`          |
| `run_log_midpoint_sweep.py`          | `physical_noise.run_log_midpoint_sweep`                      |
| `run_low_p_extension.py`             | `physical_noise.continuations.run_low_p_extension`           |
| `run_physical_noise_comparison.py`   | `physical_noise.run_physical_noise_comparison`               |
| `run_ppm_basis_first.py`             | `pilots.run_ppm_basis_first`                                 |
| `run_ppm_fidelity_sweep.py`          | `bell_fidelity.run_ppm_fidelity_sweep`                       |
| `run_surface_local_order.py`         | `schedules.run_surface_local_order`                          |
| `single_fault_audit.py`              | `fault_analysis.single_fault_audit`                          |
| `summarize_large_ppm_pilot.py`       | `pilots.summarize_large_ppm_pilot`                           |
| `validate_large_ppm_pilot.py`        | `pilots.validate_large_ppm_pilot`                            |
| `verify_single_fault_failures.py`    | `fault_analysis.verify_single_fault_failures`                |

The old top-level `run_experiment`, `run_ppm_basis_first`, `run_ppm_fidelity_sweep`,
`run_fidelity_sweep`, and `run_physical_noise_comparison` modules remain thin
compatibility launchers. Their imports alias the real module, rather than copying
functions or duplicating implementation. Use the new paths for new commands.

Outputs and checkpoint directories have not moved. Do not edit archived commands
or provenance files; use this map when repeating a historical command.
