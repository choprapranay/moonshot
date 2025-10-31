from __future__ import annotations

from typing import List
import json

import pandas as pd
from fastapi import HTTPException

from pybaseball import statcast_single_game

from app.schemas.game_analysis import (
    GameAnalysisResponse,
    PlayerStats,
    PlayerSummary,
    TeamInfo,
)


PLAYER_HEADSHOT_URL = (
    "https://img.mlbstatic.com/mlb-photos/image/upload/w_120,h_120,c_fill/v1/people/{player_id}/headshot/silo/current"
)


def _fetch_game_dataframe(game_pk: int) -> pd.DataFrame:
    df = statcast_single_game(game_pk)
    if df is None or df.empty:
        raise HTTPException(status_code=404, detail="No data found for this game ID")
    return df


def _normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    clean_df = (
        df.replace([float("inf"), float("-inf")], pd.NA)
        .where(pd.notnull(df), None)
    )
    clean_df = clean_df.where(pd.notnull(clean_df), None)
    clean_df = clean_df.copy()

    # Determine batter team based on inning context
    clean_df["team_at_bat"] = clean_df.apply(
        lambda row: _resolve_team_for_plate_appearance(
            row.get("inning_topbot"), row.get("home_team"), row.get("away_team")
        ),
        axis=1,
    )

    # ensure batter id numeric
    clean_df["batter_id"] = pd.to_numeric(clean_df.get("batter"), errors="coerce")

    return clean_df


def _resolve_team_for_plate_appearance(
    inning_state: str | None, home_team: str | None, away_team: str | None
) -> str | None:
    if inning_state is None:
        return None
    inning_state = str(inning_state).lower()
    if inning_state == "top":
        return away_team
    if inning_state == "bot" or inning_state == "bottom":
        return home_team
    return None


def _compute_player_stats(group: pd.DataFrame) -> PlayerStats:
    total_pitches = len(group)

    if total_pitches == 0:
        return PlayerStats(
            pitches_seen=0,
            swing_percentage=0.0,
            take_percentage=0.0,
            whiff_percentage=0.0,
            contact_percentage=0.0,
            average_velocity=None,
        )

    description_series = group.get("description", pd.Series(dtype=str)).astype(str)
    pitch_type_series = group.get("type", pd.Series(dtype=str)).astype(str)

    swing_mask = (
        description_series.str.contains("swing|foul|in_play", case=False, na=False)
        | pitch_type_series.str.upper().eq("X")
    )
    whiff_mask = description_series.str.contains("swinging_strike", case=False, na=False)

    swings = int(swing_mask.sum())
    takes = total_pitches - swings
    whiffs = int(whiff_mask.sum())
    contacts = max(swings - whiffs, 0)

    swing_pct = swings / total_pitches if total_pitches else 0.0
    take_pct = takes / total_pitches if total_pitches else 0.0
    whiff_pct = whiffs / swings if swings else 0.0
    contact_pct = contacts / swings if swings else 0.0

    average_velocity = group.get("release_speed").dropna().astype(float).mean()
    avg_velocity_value = float(round(average_velocity, 1)) if pd.notnull(average_velocity) else None

    return PlayerStats(
        pitches_seen=int(total_pitches),
        swing_percentage=round(swing_pct * 100, 1),
        take_percentage=round(take_pct * 100, 1),
        whiff_percentage=round(whiff_pct * 100, 1),
        contact_percentage=round(contact_pct * 100, 1),
        average_velocity=avg_velocity_value,
    )


def _calculate_impact_delta(group: pd.DataFrame, swing_mask: pd.Series) -> float | None:
    if swing_mask is None or swing_mask.empty:
        return None

    swings = group[swing_mask]
    if swings.empty:
        return None

    numeric_plate_x = pd.to_numeric(swings.get("plate_x"), errors="coerce")
    numeric_plate_z = pd.to_numeric(swings.get("plate_z"), errors="coerce")
    high_inside_mask = (numeric_plate_x <= -0.5) & (numeric_plate_z >= 3.0)
    if not high_inside_mask.any():
        return None

    delta_series = pd.to_numeric(swings.get("delta_run_exp"), errors="coerce").fillna(0.0)
    impact_value = float(delta_series[high_inside_mask].sum())
    return round(impact_value, 1)


def _build_player_summary(group_key, group: pd.DataFrame) -> PlayerSummary:
    player_id, player_name, team_code = group_key
    stats = _compute_player_stats(group)

    description_series = group.get("description", pd.Series(dtype=str)).astype(str)
    pitch_type_series = group.get("type", pd.Series(dtype=str)).astype(str)
    swing_mask = (
        description_series.str.contains("swing|foul|in_play", case=False, na=False)
        | pitch_type_series.str.upper().eq("X")
    )

    impact_delta = _calculate_impact_delta(group, swing_mask)

    headshot_url = None
    if pd.notnull(player_id):
        headshot_url = PLAYER_HEADSHOT_URL.format(player_id=int(player_id))

    return PlayerSummary(
        player_id=int(player_id),
        player_name=str(player_name),
        team=str(team_code) if team_code else "",
        headshot_url=headshot_url,
        stats=stats,
        impact_zone_delta=impact_delta,
    )


def build_game_analysis(game_pk: int) -> GameAnalysisResponse:
    df = _fetch_game_dataframe(game_pk)
    normalized_df = _normalize_dataframe(df)

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


def build_pitch_records(game_pk: int) -> List[dict]:
    df = _fetch_game_dataframe(game_pk)
    normalized_df = _normalize_dataframe(df)
    records = json.loads(normalized_df.to_json(orient="records"))
    return records

