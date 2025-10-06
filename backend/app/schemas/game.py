from pydantic import BaseModel
from datetime import date
from typing import Optional
from uuid import UUID

class GameCreate(BaseModel):
    game_date: date
    home_team: str
    away_team: str

class GameUpdate(BaseModel):
    game_date: Optional[date] = None
    home_team: Optional[str] = None
    away_team: Optional[str] = None

class GameResponse(BaseModel):
    id: UUID
    game_date: date
    home_team: Optional[str]
    away_team: Optional[str]

class TmpItem(BaseModel):
    name: str
