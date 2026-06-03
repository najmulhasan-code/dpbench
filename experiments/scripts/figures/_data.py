"""Shared CSV-loading helpers for figure modules."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


_HERE = Path(__file__).resolve()
PER_ANALYSIS = _HERE.parent.parent.parent / "results" / "aggregated" / "per_analysis"
MASTER_CSV = _HERE.parent.parent.parent / "results" / "aggregated" / "master.csv"


def load_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def to_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def to_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except ValueError:
        return None


def find_row(rows: list[dict[str, Any]], **filters: Any) -> dict[str, Any] | None:
    for row in rows:
        if all(str(row.get(k)) == str(v) for k, v in filters.items()):
            return row
    return None
