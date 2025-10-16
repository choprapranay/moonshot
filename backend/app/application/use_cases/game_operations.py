"""
Use cases - extracted business logic from baseball.py
"""
import random
from pybaseball import statcast_single_game
import pandas as pd
from typing import Optional, List, Dict, Any


class GameOperations:
    """
    Business logic for game operations
    Just extracted from baseball.py - minimal changes
    """
    
    def __init__(self, supabase_client):
        self.supabase = supabase_client
    
    def fetch_game_data_and_save(self, game_pk: int) -> Dict[str, Any]:
        """
        This is your EXACT SAME CODE from baseball.py
        Just moved into a method
        """
        # Check if game data already exists in games table
        game_row_raw = self.supabase.table("games").select("id, game_date, home_team, away_team").eq("game_pk", game_pk).maybe_single().execute()
        
        if game_row_raw and game_row_raw.data:
            game_row = game_row_raw.data
        else:
            game_row = None
        
        # This means game data already exists in games table
        if game_row:
            # Fetch pitches data and player data per pitch
            pitches = self.supabase.table("pitches").select("*, players(name, diagram_index)").eq("game_id", game_row["id"]).execute().data
            
            pitches_data = [
                {
                    "pitch_type": pitch["pitch_type"],
                    "speed": float(pitch["speed"]) if pitch["speed"] else None,
                    "description": pitch["description"],
                    "player_name": pitch["players"]["name"],
                    "diagram_index": pitch["players"]["diagram_index"],
                }
                for pitch in pitches
            ]
            
            # Return game data and pitches data
            return {
                "game_id": game_row["id"],
                "gameData": {
                    "game_date": pd.to_datetime(game_row["game_date"]).date(),
                    "home_team": game_row["home_team"],
                    "away_team": game_row["away_team"],
                },
                "pitches": pitches_data,
            }
        
        # If game data does not exist, extract game data and save to games table
        # Call pybaseball api
        df = statcast_single_game(game_pk)
        
        if df is None or df.empty:
            raise ValueError("No data found for this game ID")
        
        # Extract first row
        r = df.iloc[0]
        game_payload = {
            "game_pk": int(r["game_pk"]),
            "game_date": str(pd.to_datetime(r["game_date"]).date()),
            "home_team": r["home_team"],
            "away_team": r["away_team"],
        }
        
        self.supabase.table("games").upsert(game_payload, on_conflict="game_pk").execute()
        game_id = self.supabase.table("games").select("id").eq("game_pk", game_payload["game_pk"]).single().execute().data["id"]
        
        # Extract player data
        names = (
            pd.Series(
                df["player_name"]
                .dropna()
                .astype(str)
                .str.strip()
                .unique()
            )
        )
        
        name_to_player_id = {}
        
        if not names.empty:
            payloads = [
                {
                    "name": name,
                    "diagram_index": random.randint(0, 100),
                }
                for name in names
            ]
            self.supabase.table("players").upsert(payloads, on_conflict="name").execute()
            
            players_rows = self.supabase.table("players").select("id, name").in_("name", names).execute().data
            name_to_player_id = {
                row["name"]: row["id"]
                for row in players_rows
            }
        
        pitches_data = []
        for i in range(len(df)):
            row = df.iloc[i]
            player_name = str(row["player_name"]) if pd.notna(row["player_name"]) else None
            batter_uuid = name_to_player_id.get(player_name)
            
            pitches_data.append({
                "pitch_type": str(row["pitch_type"]) if pd.notna(row["pitch_type"]) else "",
                "speed": float(row["release_speed"]) if pd.notna(row["release_speed"]) else None,
                "description": str(row["description"]) if pd.notna(row["description"]) else None,
                "player_name": player_name,
                "diagram_index": random.randint(0, 100),
            })
        
        pitch_payloads = [
            {
                "game_id": game_id,
                "batter_id": name_to_player_id.get(pitch["player_name"]),
                "pitch_type": pitch["pitch_type"],
                "speed": pitch["speed"],
                "description": pitch["description"],
            }
            for pitch in pitches_data
            if pitch["player_name"] in name_to_player_id
        ]
        self.supabase.table("pitches").upsert(pitch_payloads).execute()
        
        return {
            "game_id": game_id,
            "gameData": {
                "game_date": str(pd.to_datetime(game_payload["game_date"]).date()),
                "home_team": game_payload["home_team"],
                "away_team": game_payload["away_team"],
            },
            "pitches": pitches_data,
        }
    
    def get_all_games(self) -> List[Dict[str, Any]]:
        """Get all games - extracted from games.py"""
        res = self.supabase.table("games").select("id,game_date,home_team,away_team").order("id", desc=True).execute()
        return res.data or []