from pydantic import BaseModel
from datetime import date
from typing import Optional, List
from datetime import datetime



class GameData(BaseModel):
   game_date: datetime
   home_team: str
   away_team: str

class PitchData(BaseModel):
   pitch_type: str
   speed: Optional[float]
   description: Optional[str]
   player_name: Optional[str]
   diagram_index: Optional[int]

class FetchGameDataAndSaveResponse(BaseModel):
   game_id: str
   gameData: GameData
   pitches: List[PitchData]