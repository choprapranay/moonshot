from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, TypedDict

from fastapi import HTTPException


class HeatmapRecord(TypedDict):
    plate_x: float
    plate_z: float
    expected_value_diff: float


HEATMAP_FILE_NAME = "heatmap_output.json"


def _dataset_path() -> Path:
    backend_root = Path(__file__).resolve().parents[2]
    dataset_path = backend_root / HEATMAP_FILE_NAME
    if not dataset_path.exists():
        raise HTTPException(status_code=500, detail="Heatmap dataset not found")
    return dataset_path


@lru_cache(maxsize=1)
def _load_dataset() -> Dict[str, List[HeatmapRecord]]:
    path = _dataset_path()
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def get_heatmap_for_batter(batter_id: int) -> List[HeatmapRecord]:
    dataset = _load_dataset()
    return dataset.get(str(batter_id), [])

