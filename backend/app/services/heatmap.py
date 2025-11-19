from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, TypedDict

class HeatmapRecord(TypedDict):
    plate_x: float
    plate_z: float
    expected_value_diff: float


HEATMAP_FILE_NAME = "heatmap_output.json"


def _dataset_path() -> Path:
    # Try multiple paths for deployment flexibility
    paths = [
        Path(__file__).resolve().parents[2] / "expected-value-calculations" / HEATMAP_FILE_NAME,
        Path.cwd() / "expected-value-calculations" / HEATMAP_FILE_NAME,
        Path.cwd() / "backend" / "expected-value-calculations" / HEATMAP_FILE_NAME,
    ]
    for path in paths:
        if path.exists():
            return path
    return paths[0]  # Return first path even if missing (will be handled in _load_dataset)


@lru_cache(maxsize=1)
def _load_dataset() -> Dict[str, List[HeatmapRecord]]:
    path = _dataset_path()
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def get_heatmap_for_batter(batter_id: int) -> List[HeatmapRecord]:
    dataset = _load_dataset()
    return dataset.get(str(batter_id), [])

