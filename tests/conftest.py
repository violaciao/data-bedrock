"""Shared pytest fixtures — load synthetic CSV data once per session."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).parents[1]
DATA_DIR = REPO_ROOT / "data" / "synthetic"

# Make each module's src/ importable without package-relative imports.
for module_dir in ["05_measurement", "04_cohort_retention", "03_growth_analysis"]:
    p = str(REPO_ROOT / module_dir)
    if p not in sys.path:
        sys.path.insert(0, p)


def _ensure_synthetic_data() -> None:
    """Generate synthetic data if the CSV files don't exist yet."""
    if not (DATA_DIR / "users.csv").exists():
        script = DATA_DIR / "generate_synthetic_data.py"
        subprocess.run(["python", str(script)], check=True)


@pytest.fixture(scope="session")
def synthetic_users() -> pd.DataFrame:
    _ensure_synthetic_data()
    return pd.read_csv(DATA_DIR / "users.csv", parse_dates=["signup_at", "churn_at"])


@pytest.fixture(scope="session")
def synthetic_events() -> pd.DataFrame:
    _ensure_synthetic_data()
    return pd.read_csv(DATA_DIR / "events.csv", parse_dates=["occurred_at"])


@pytest.fixture(scope="session")
def synthetic_orders() -> pd.DataFrame:
    _ensure_synthetic_data()
    return pd.read_csv(DATA_DIR / "orders.csv", parse_dates=["billed_at", "period_start", "period_end"])


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)
