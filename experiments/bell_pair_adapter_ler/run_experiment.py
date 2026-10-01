"""Compatibility launcher; use the common.run_experiment module for new commands."""

import importlib
import runpy
import sys

_TARGET = "experiments.bell_pair_adapter_ler.common.run_experiment"

if __name__ == "__main__":
    runpy.run_module(_TARGET, run_name="__main__")
else:
    # Preserve module identity (including monkeypatches) for existing imports.
    sys.modules[__name__] = importlib.import_module(_TARGET)
