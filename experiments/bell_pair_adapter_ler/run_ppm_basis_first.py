"""Compatibility launcher; use the pilots.run_ppm_basis_first module for new commands."""

import importlib
import runpy
import sys

_TARGET = "experiments.bell_pair_adapter_ler.pilots.run_ppm_basis_first"

if __name__ == "__main__":
    runpy.run_module(_TARGET, run_name="__main__")
else:
    # Preserve module identity (including monkeypatches) for existing imports.
    sys.modules[__name__] = importlib.import_module(_TARGET)
