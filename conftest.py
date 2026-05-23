"""Root conftest.py — runs before any test collection.

Adds each module's directory to sys.path so tests can do
``from src.stats import ...`` regardless of how pytest was invoked.
"""

import sys
from pathlib import Path

_REPO = Path(__file__).parent

for _module in ["05_measurement", "04_cohort_retention", "03_growth_analysis"]:
    _p = str(_REPO / _module)
    if _p not in sys.path:
        sys.path.insert(0, _p)
