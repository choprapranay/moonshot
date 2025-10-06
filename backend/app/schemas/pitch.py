from pydantic import BaseModel
from typing import Optional
from uuid import UUID
from decimal import Decimal

class PitchCreate(BaseModel):
    game_id: UUID
    batter_id: UUID
    pitch_type: str
    speed: Optional[Decimal] = None
    description: Optional[str] = None

class PitchUpdate(BaseModel):
    pitch_type: Optional[str] = None
    speed: Optional[Decimal] = None
    description: Optional[str] = None

class PitchResponse(BaseModel):
    id: UUID
    game_id: UUID
    batter_id: UUID
    pitch_type: str
    speed: Optional[Decimal]
    description: Optional[str]
