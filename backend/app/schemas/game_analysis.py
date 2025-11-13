from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class TeamInfo(BaseModel):
    code: str = Field(..., description="Team abbreviation (e.g., CWS)")
    name: str = Field(..., description="Display name for the team")


class PlayerStats(BaseModel):
    pitches_seen: int
    swing_percentage: float
    take_percentage: float
    whiff_percentage: float
    contact_percentage: float
    average_velocity: Optional[float]
    pitcher_handedness: Optional[str]


class PlayerSummary(BaseModel):
    player_id: int
    player_name: str
    team: str
    headshot_url: Optional[str]
    stats: PlayerStats
    impact_zone_delta: Optional[float]


class GameAnalysisResponse(BaseModel):
    game_id: int
    game_date: str
    teams: List[TeamInfo]
    players: List[PlayerSummary]


class HeatmapPoint(BaseModel):
    plate_x: float
    plate_z: float
    expected_value_diff: float


class SwingPoint(BaseModel):
    plate_x: float
    plate_z: float
    pitch_type: Optional[str]
    description: Optional[str]


class PlayerHeatmapResponse(BaseModel):
    heatmap: List[HeatmapPoint]
    swings: List[SwingPoint]

