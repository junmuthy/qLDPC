"""Stable artifact roots and relocated source-file lookup.

Moving a runner must never move its default checkpoints or change provenance keys.
"""

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PACKAGE_ROOT / "outputs"
SOURCE_LOCATIONS = {
    "adapter_structure_audit.py": "fault_analysis/adapter_structure_audit.py",
    "analyze_fidelity_sweep.py": "bell_fidelity/analyze_fidelity_sweep.py",
    "collection_profile.py": "common/collection_profile.py",
    "compare_hardware_round_sweep.py": "comparisons/compare_hardware_round_sweep.py",
    "dual_timing.py": "schedules/dual_timing.py",
    "effective_distance_audit.py": "fault_analysis/effective_distance_audit.py",
    "explain_distance_witnesses.py": "fault_analysis/explain_distance_witnesses.py",
    "five_fault_search.py": "fault_analysis/five_fault_search.py",
    "hgp_xz_dual.py": "schedules/hgp_xz_dual.py",
    "paired_decoder_check.py": "fault_analysis/paired_decoder_check.py",
    "plot_d8_low_p_mean_history.py": "physical_noise/plot_d8_low_p_mean_history.py",
    "plot_hgp_failure_history.py": "comparisons/plot_hgp_failure_history.py",
    "plot_hgp_round_comparison.py": "comparisons/plot_hgp_round_comparison.py",
    "plot_surface_all_datasets_bell1.py": "comparisons/plot_surface_all_datasets_bell1.py",
    "run_d4_midpoint_extension.py": "physical_noise/continuations/run_d4_midpoint_extension.py",
    "run_d4_r8_control.py": "physical_noise/run_d4_r8_control.py",
    "run_d8_noise_extension.py": "physical_noise/continuations/run_d8_noise_extension.py",
    "run_d8_single_point_extension.py": "physical_noise/continuations/run_d8_single_point_extension.py",
    "run_experiment.py": "common/run_experiment.py",
    "run_fidelity_sweep.py": "bell_fidelity/run_fidelity_sweep.py",
    "run_hardware_comparison.py": "comparisons/run_hardware_comparison.py",
    "run_hgp_dual_schedule.py": "schedules/run_hgp_dual_schedule.py",
    "run_hgp_hardware_comparison.py": "comparisons/run_hgp_hardware_comparison.py",
    "run_hgp_schedule_ablations.py": "schedules/run_hgp_schedule_ablations.py",
    "run_hgp_xz_dual.py": "schedules/run_hgp_xz_dual.py",
    "run_high_p_extension.py": "physical_noise/continuations/run_high_p_extension.py",
    "run_log_midpoint_sweep.py": "physical_noise/run_log_midpoint_sweep.py",
    "run_low_p_extension.py": "physical_noise/continuations/run_low_p_extension.py",
    "run_physical_noise_comparison.py": "physical_noise/run_physical_noise_comparison.py",
    "run_ppm_basis_first.py": "pilots/run_ppm_basis_first.py",
    "run_ppm_fidelity_sweep.py": "bell_fidelity/run_ppm_fidelity_sweep.py",
    "run_surface_local_order.py": "schedules/run_surface_local_order.py",
    "single_fault_audit.py": "fault_analysis/single_fault_audit.py",
    "summarize_large_ppm_pilot.py": "pilots/summarize_large_ppm_pilot.py",
    "validate_large_ppm_pilot.py": "pilots/validate_large_ppm_pilot.py",
    "verify_single_fault_failures.py": "fault_analysis/verify_single_fault_failures.py",
}


def source_path(filename: str) -> Path:
    """Resolve an original source basename without importing or running the module."""
    return PACKAGE_ROOT / SOURCE_LOCATIONS[filename]
