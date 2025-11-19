from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

import torch
from torch.serialization import add_safe_globals
from sklearn.preprocessing import LabelEncoder


MODEL_FILENAME = "neuralnet/batter_outcome_model.pth"


def _model_path() -> Path:
    # Try multiple paths for deployment flexibility
    paths = [
        Path(__file__).resolve().parents[2] / MODEL_FILENAME,
        Path.cwd() / MODEL_FILENAME,
        Path.cwd() / "backend" / MODEL_FILENAME,
    ]
    for path in paths:
        if path.exists():
            return path
    return paths[0]  # Return first path even if missing (will be handled in _load_batter_encoder)


@lru_cache(maxsize=1)
def _load_batter_encoder():
    model_path = _model_path()
    if not model_path.exists():
        return None
    add_safe_globals([LabelEncoder])
    payload = torch.load(model_path, map_location="cpu", weights_only=False)
    return payload.get("batter_encoder")


def encode_batter_id(raw_batter_id: int) -> Optional[int]:
    encoder = _load_batter_encoder()
    if encoder is None:
        return None
    try:
        encoded = encoder.transform([raw_batter_id])
        return int(encoded[0])
    except ValueError:
        return None

