from __future__ import annotations

from typing import List
import json
import math

import pandas as pd
from fastapi import HTTPException

from pybaseball import statcast_single_game

from app.api.schemas.game_analysis import (
    GameAnalysisResponse,
    PlayerStats as PlayerStatsSchema,
    PlayerSummary,
    TeamInfo,
)
from app.domain.services import (
    DataNormalizer,
    PlayerStatsCalculator,
    SwingDetector,
)
from app.domain.entities import PlayerStats as PlayerStatsEntity


PLAYER_HEADSHOT_URL = (
    "https://img.mlbstatic.com/mlb-photos/image/upload/w_120,h_120,c_fill/v1/people/{player_id}/headshot/silo/current"
)


def _fetch_game_dataframe(game_pk: int) -> pd.DataFrame:
    """Infrastructure concern: fetch data from external source"""
    df = statcast_single_game(game_pk)
    if df is None or df.empty:
        raise HTTPException(status_code=404, detail="No data found for this game ID")
    return df


def _convert_domain_stats_to_schema(stats: PlayerStatsEntity) -> PlayerStatsSchema:
    """Convert domain entity to API schema"""
    return PlayerStatsSchema(
        pitches_seen=stats.pitches_seen,
        swing_percentage=stats.swing_percentage,
        take_percentage=stats.take_percentage,
        whiff_percentage=stats.whiff_percentage,
        contact_percentage=stats.contact_percentage,
        average_velocity=stats.average_velocity,
        batter_handedness=stats.batter_handedness,
    )


def _build_player_summary(group_key, group: pd.DataFrame) -> PlayerSummary:
    """Build player summary using domain services"""
    player_id, player_name, team_code = group_key
    
    # Use domain service to calculate stats
    domain_stats = PlayerStatsCalculator.calculate_stats(group)
    stats = _convert_domain_stats_to_schema(domain_stats)

    impact_delta = None  # Placeholder for future implementation

    headshot_url = None
    if pd.notnull(player_id):
        headshot_url = PLAYER_HEADSHOT_URL.format(player_id=int(player_id))

    # Handle None/NaN player_id for int conversion
    player_id_int = int(player_id) if pd.notnull(player_id) else 0

    return PlayerSummary(
        player_id=player_id_int,
        player_name=str(player_name),
        team=str(team_code) if team_code else "",
        headshot_url=headshot_url,
        stats=stats,
        impact_zone_delta=impact_delta,
    )


class BuildGameAnalysisUseCase:
    """Use case for building game analysis with player statistics"""
    
    def execute(self, game_pk: int) -> GameAnalysisResponse:
        """Build game analysis using domain services"""
        df = _fetch_game_dataframe(game_pk)
        # Use domain service to normalize data
        normalized_df = DataNormalizer.normalize_dataframe(df)

        first_row = normalized_df.iloc[0]
        home_team = first_row.get("home_team")
        away_team = first_row.get("away_team")

        teams: List[TeamInfo] = []
        if home_team:
            teams.append(TeamInfo(code=str(home_team), name=str(home_team)))
        if away_team:
            teams.append(TeamInfo(code=str(away_team), name=str(away_team)))

        player_rows = normalized_df[
            normalized_df["batter_id"].notnull() & normalized_df["player_name"].notnull()
        ]

        grouped = player_rows.groupby(["batter_id", "player_name", "team_at_bat"])

        player_summaries = [
            _build_player_summary(group_key, group)
            for group_key, group in grouped
        ]

        player_summaries.sort(key=lambda player: player.stats.pitches_seen, reverse=True)

        game_id = int(first_row.get("game_pk")) if first_row.get("game_pk") is not None else game_pk
        game_date = str(pd.to_datetime(first_row.get("game_date")).date()) if first_row.get("game_date") else ""

        return GameAnalysisResponse(
            game_id=game_id,
            game_date=game_date,
            teams=teams,
            players=player_summaries,
        )


class BuildPitchRecordsUseCase:
    """Use case for building pitch records from game data"""
    
    def execute(self, game_pk: int) -> List[dict]:
        """Build pitch records using domain services"""
        df = _fetch_game_dataframe(game_pk)
        # Use domain service to normalize data
        normalized_df = DataNormalizer.normalize_dataframe(df)
        records = json.loads(normalized_df.to_json(orient="records"))
        return records


class GetBatterSwingsUseCase:
    """Use case for getting batter swing data for heatmap visualization"""
    
    def execute(self, game_pk: int, batter_id: int) -> List[dict]:
        """Get batter swings using domain services"""
        df = _fetch_game_dataframe(game_pk)
        # Use domain service to normalize data
        normalized_df = DataNormalizer.normalize_dataframe(df)
        batter_rows = normalized_df[
            (normalized_df["batter_id"] == batter_id)
            & normalized_df["plate_x"].notnull()
            & normalized_df["plate_z"].notnull()
        ]

        if batter_rows.empty:
            return []

        # Use domain service to detect swings
        swing_dicts = []
        for _, row in batter_rows.iterrows():
            try:
                description = str(row.get("description", "")) if pd.notnull(row.get("description")) else ""
                pitch_type = str(row.get("type", "")) if pd.notnull(row.get("type")) else ""
                
                # Use domain service to determine if this is a swing
                if SwingDetector.is_swing_for_heatmap(description, pitch_type):
                    plate_x = row["plate_x"]
                    plate_z = row["plate_z"]
                    
                    # Ensure plate_x and plate_z are valid floats
                    if pd.notnull(plate_x) and pd.notnull(plate_z):
                        try:
                            plate_x_float = float(plate_x)
                            plate_z_float = float(plate_z)
                            
                            # Check for infinity values
                            if not (math.isinf(plate_x_float) or math.isinf(plate_z_float)):
                                swing_dicts.append(
                                    {
                                        "plate_x": plate_x_float,
                                        "plate_z": plate_z_float,
                                        "pitch_type": str(row.get("pitch_type")) if pd.notnull(row.get("pitch_type")) else None,
                                        "description": str(row.get("description")) if pd.notnull(row.get("description")) else None,
                                    }
                                )
                        except (ValueError, TypeError):
                            # Skip invalid coordinate values
                            continue
            except Exception:
                # Skip rows that cause errors
                continue

        return swing_dicts
