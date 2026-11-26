"""Reading and writing the generated survey files.

``generate`` writes three CSVs: the stay population as cell counts (what a property-management
system extract would give), the survey responses (one row per respondent, with the 0-10 score),
and the synthetic truth used only to score the method.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .synth import Population

POPULATION_FILE = "population.csv"
RESPONSES_FILE = "responses.csv"
TRUTH_FILE = "truth.csv"

POPULATION_COLUMNS = ["quarter", "property", "tier", "channel", "purpose", "stays"]
RESPONSE_COLUMNS = ["quarter", "property", "tier", "channel", "purpose", "score"]


def save_dataset(population: Population, responses: pd.DataFrame, out_dir: Path) -> list[Path]:
    """Write the three CSVs into ``out_dir`` and return their paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / POPULATION_FILE, out_dir / RESPONSES_FILE, out_dir / TRUTH_FILE]
    population.margin_table().to_csv(paths[0], index=False)
    responses.to_csv(paths[1], index=False)
    population.truth_table().round(4).to_csv(paths[2], index=False)
    return paths


def _read(path: Path, columns: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `surveyrake generate` first")
    df = pd.read_csv(path)
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns {missing}")
    for col in columns:
        if col not in ("stays", "score"):
            df[col] = df[col].astype(str)
    return df


def load_population(data_dir: Path) -> pd.DataFrame:
    return _read(data_dir / POPULATION_FILE, POPULATION_COLUMNS)


def load_responses(data_dir: Path) -> pd.DataFrame:
    return _read(data_dir / RESPONSES_FILE, RESPONSE_COLUMNS)


def load_truth(data_dir: Path) -> pd.DataFrame | None:
    path = data_dir / TRUTH_FILE
    return pd.read_csv(path) if path.exists() else None
