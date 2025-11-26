from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

import torch
from torch.serialization import add_safe_globals
from sklearn.preprocessing import LabelEncoder
from fastapi import HTTPException


MODEL_FILENAME = "neuralnet/batter_outcome_model.pth"


def _model_path() -> Path:
    backend_root = Path(__file__).resolve().parents[3]
    model_path = backend_root / MODEL_FILENAME
    if not model_path.exists():
        raise HTTPException(status_code=500, detail="Batter encoder model file not found")
    return model_path


@lru_cache(maxsize=1)
def _load_batter_encoder():
    model_path = _model_path()
    add_safe_globals([LabelEncoder])
    payload = torch.load(model_path, map_location="cpu", weights_only=False)
    encoder = payload.get("batter_encoder")
    if encoder is None:
        raise HTTPException(status_code=500, detail="Batter encoder missing from model payload")
    return encoder


def encode_batter_id(raw_batter_id: int) -> Optional[int]:
    encoder = _load_batter_encoder()
    try:
        encoded = encoder.transform([raw_batter_id])
        return int(encoded[0])
    except ValueError:
        return None

