"""Compatibility launcher; use the physical_noise.run_physical_noise_comparison module for new commands."""

import importlib
import runpy
import sys

_TARGET = "experiments.bell_pair_adapter_ler.physical_noise.run_physical_noise_comparison"

if __name__ == "__main__":
    runpy.run_module(_TARGET, run_name="__main__")
else:
    # Preserve module identity (including monkeypatches) for existing imports.
    sys.modules[__name__] = importlib.import_module(_TARGET)
