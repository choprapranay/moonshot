from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class Game:
    """Domain entity representing a baseball game"""
    game_pk: int
    game_date: str
    home_team: str
    away_team: str
    game_id: Optional[str] = None


@dataclass
class Pitch:
    """Domain entity representing a pitch"""
    pitch_type: str
    speed: Optional[float]
    description: Optional[str]
    plate_x: Optional[float] = None
    plate_z: Optional[float] = None
    release_speed: Optional[float] = None


@dataclass
class Player:
    """Domain entity representing a player"""
    player_id: int
    player_name: str
    team: Optional[str] = None
    diagram_index: Optional[int] = None


@dataclass
class PlayerStats:
    """Domain entity representing player statistics"""
    pitches_seen: int
    swing_percentage: float
    take_percentage: float
    whiff_percentage: float
    contact_percentage: float
    average_velocity: Optional[float]
    batter_handedness: Optional[str] = None

