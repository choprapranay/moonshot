from __future__ import annotations

from typing import List
import pandas as pd

from app.domain.entities import PlayerStats


class TeamResolver:
    """Domain service for resolving team information"""
    
    @staticmethod
    def resolve_team_for_plate_appearance(
        inning_state: str | None, home_team: str | None, away_team: str | None
    ) -> str | None:
        """Determine which team is at bat based on inning state"""
        if inning_state is None:
            return None
        inning_state = str(inning_state).lower()
        if inning_state == "top":
            return away_team
        if inning_state == "bot" or inning_state == "bottom":
            return home_team
        return None


class SwingDetector:
    """Domain service for detecting swings in pitch data"""
    
    @staticmethod
    def is_swing(description: str, pitch_type: str) -> bool:
        """Determine if a pitch resulted in a swing"""
        description_lower = str(description).lower()
        pitch_type_upper = str(pitch_type).upper()
        
        # Check if description indicates a swing
        swing_keywords = ["swing", "foul", "in_play"]
        has_swing_keyword = any(keyword in description_lower for keyword in swing_keywords)
        
        # Check if pitch type indicates a swing (X = in play)
        is_in_play = pitch_type_upper == "X"
        
        return has_swing_keyword or is_in_play
    
    @staticmethod
    def is_swing_for_heatmap(description: str, pitch_type: str) -> bool:
        """Determine if a pitch resulted in a swing (for heatmap purposes)"""
        description_lower = str(description).lower()
        pitch_type_upper = str(pitch_type).upper()
        
        # Check if it's a called or blocked ball (not a swing)
        if any(keyword in description_lower for keyword in ["called", "blocked_ball"]):
            return False
        
        # Check if pitch type indicates in play
        if pitch_type_upper in ["S", "X"]:
            return True
        
        # Check if description indicates a swing
        swing_keywords = ["swing", "foul", "in_play"]
        return any(keyword in description_lower for keyword in swing_keywords)
    
    @staticmethod
    def is_whiff(description: str) -> bool:
        """Determine if a swing resulted in a whiff (miss)"""
        description_lower = str(description).lower()
        return "swinging_strike" in description_lower


class PlayerStatsCalculator:
    """Domain service for calculating player statistics"""
    
    @staticmethod
    def calculate_stats(pitches: pd.DataFrame) -> PlayerStats:
        """Calculate player statistics from a collection of pitches"""
        total_pitches = len(pitches)
        
        if total_pitches == 0:
            return PlayerStats(
                pitches_seen=0,
                swing_percentage=0.0,
                take_percentage=0.0,
                whiff_percentage=0.0,
                contact_percentage=0.0,
                average_velocity=None,
            )
        
        description_series = pitches.get("description", pd.Series(dtype=str)).astype(str)
        pitch_type_series = pitches.get("type", pd.Series(dtype=str)).astype(str)
        
        # Use domain service to detect swings and whiffs
        swing_mask = pd.Series([
            SwingDetector.is_swing(desc, pitch_type)
            for desc, pitch_type in zip(description_series, pitch_type_series)
        ])
        whiff_mask = pd.Series([
            SwingDetector.is_whiff(desc)
            for desc in description_series
        ])
        
        swings = int(swing_mask.sum())
        takes = total_pitches - swings
        whiffs = int((swing_mask & whiff_mask).sum())
        contacts = max(swings - whiffs, 0)
        
        # Calculate percentages
        swing_pct = swings / total_pitches if total_pitches else 0.0
        take_pct = takes / total_pitches if total_pitches else 0.0
        whiff_pct = whiffs / swings if swings else 0.0
        contact_pct = contacts / swings if swings else 0.0
        
        # Calculate average velocity
        average_velocity = pitches.get("release_speed").dropna().astype(float).mean()
        avg_velocity_value = float(round(average_velocity, 1)) if pd.notnull(average_velocity) else None
        
        # Determine batter handedness
        stand_series = pitches.get("stand", pd.Series(dtype=str))
        batter_handedness = None
        if not stand_series.empty and stand_series.notna().any():
            most_common = stand_series.dropna().mode()
            if not most_common.empty:
                batter_handedness = str(most_common.iloc[0]).upper()
        
        return PlayerStats(
            pitches_seen=int(total_pitches),
            swing_percentage=round(swing_pct * 100, 1),
            take_percentage=round(take_pct * 100, 1),
            whiff_percentage=round(whiff_pct * 100, 1),
            contact_percentage=round(contact_pct * 100, 1),
            average_velocity=avg_velocity_value,
            batter_handedness=batter_handedness,
        )


class DataNormalizer:
    """Domain service for normalizing raw game data"""
    
    @staticmethod
    def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Normalize a dataframe by cleaning data and adding computed fields"""
        clean_df = (
            df.replace([float("inf"), float("-inf")], pd.NA)
            .where(pd.notnull(df), None)
        )
        clean_df = clean_df.where(pd.notnull(clean_df), None)
        clean_df = clean_df.copy()
        
        # Add team_at_bat field using domain service
        clean_df["team_at_bat"] = clean_df.apply(
            lambda row: TeamResolver.resolve_team_for_plate_appearance(
                row.get("inning_topbot"), row.get("home_team"), row.get("away_team")
            ),
            axis=1,
        )
        
        # Convert batter to numeric ID
        clean_df["batter_id"] = pd.to_numeric(clean_df.get("batter"), errors="coerce")
        
        return clean_df

