
import random
from fastapi import APIRouter, HTTPException
from pybaseball import statcast_single_game
import pandas as pd
import json

from pydantic import BaseModel
from typing import List, Optional
from app.db.database import supabase
from datetime import datetime

router = APIRouter()

@router.get("/game-data/{game_pk}")
def get_game_data_with_db(game_pk: int):
   df = statcast_single_game(game_pk)


   return fetch_game_data_and_save(game_pk)

class GameData(BaseModel):
   game_date: datetime
   home_team: str
   away_team: str

class PitchData(BaseModel):
   pitch_type: str
   speed: Optional[float]
   description: Optional[str]
   player_id: str
   diagram_index: Optional[int]

class FetchGameDataAndSaveResponse(BaseModel):
   game_id: str
   gameData: GameData
   pitches: List[PitchData]

def fetch_game_data_and_save(game_pk: int) -> FetchGameDataAndSaveResponse:   
   # extract game data to upload to games table
   # (tmp) endpoint data keys
   # Index(['pitch_type', 'game_date', 'release_speed', 'release_pos_x',
      #  'release_pos_z', 'player_name', 'batter', 'pitcher', 'events',
      #  'description',
      #  ...
      #  'batter_days_until_next_game', 'api_break_z_with_gravity',
      #  'api_break_x_arm', 'api_break_x_batter_in', 'arm_angle', 'attack_angle',
      #  'attack_direction', 'swing_path_tilt',
      #  'intercept_ball_minus_batter_pos_x_inches',
      #  'intercept_ball_minus_batter_pos_y_inches'],
   # check if game data already exists in games table
   game_row_raw = supabase.table("games").select("id, game_date, home_team, away_team").eq("game_pk", game_pk).maybe_single().execute()
   
   if game_row_raw and game_row_raw.data:
      game_row = game_row_raw.data
   else:
      game_row = None

   if game_row:
      pitches = supabase.table("pitches").select("*").eq("game_id", game_row["id"]).execute().data
      pitches_data = [
         PitchData(
            pitch_type=pitch["pitch_type"],
            speed=float(pitch["speed"]),
            description=pitch["description"],
            player_id=pitch["batter_id"],
            diagram_index=int(pitch["diagram_index"]),
         )
         for pitch in pitches
      ]
      return FetchGameDataAndSaveResponse(
         game_id=game_row["id"],
         gameData=GameData(
            game_date=pd.to_datetime(game_row["game_date"]).date(),
            home_team=game_row["home_team"],
            away_team=game_row["away_team"],
         ),
         pitches=pitches_data,
      )
   # call pybaseball api
   df = statcast_single_game(game_pk)
   if df is None or df.empty:
      raise HTTPException(status_code=404, detail="No data found for this game ID")

   # if game data does not exist, extract game data and save to games table
   r = df.iloc[0]
   game_payload = {
      "game_pk": int(r["game_pk"]),
      "game_date": str(pd.to_datetime(r["game_date"]).date()),
      "home_team": r["home_team"],
      "away_team": r["away_team"],
   }
   supabase.table("games").upsert(game_payload, on_conflict="game_pk").execute()
   game_id = supabase.table("games").select("id").eq("game_pk", game_payload["game_pk"]).single().execute().data["id"]

   # extract player data to upload to player table
   name_to_player_id = {}
   for i in range(len(df)):
      row = df.iloc[i]
      player_name = str(row["player_name"])

      if player_name in name_to_player_id:
         continue
      else:
         player_payload = {
            "name": player_name,
         }
         supabase.table("players").upsert(player_payload, on_conflict="name").execute()
         name_to_player_id[player_name] = supabase.table("players").select("id").eq("name", player_name).single().execute().data["id"]
   
   pitches_data = []
   # extract pitches data to upload to pitches table
   for i in range(len(df)):
      row = df.iloc[i]
      player_name = str(row["player_name"]) if pd.notna(row["player_name"]) else None
      batter_uuid = name_to_player_id.get(player_name)
      pitch_payload = {
         "game_id": game_id,
         "batter_id": batter_uuid,
         "pitch_type": str(row["pitch_type"]) if pd.notna(row["pitch_type"]) else "",
         "speed": float(row["release_speed"]) if pd.notna(row["release_speed"]) else None,
         "description": str(row["description"]) if pd.notna(row["description"]) else None,
         "diagram_index": random.randint(0, 100), # TEMPORARY
      }
      supabase.table("pitches").upsert(pitch_payload).execute()
      pitches_data.append(PitchData(
         pitch_type=str(row["pitch_type"]) if pd.notna(row["pitch_type"]) else "",
         speed=float(row["release_speed"]) if pd.notna(row["release_speed"]) else None,
         description=str(row["description"]) if pd.notna(row["description"]) else None,
         player_id=batter_uuid,
         diagram_index=random.randint(0, 100), # TEMPORARY
      ))
   return FetchGameDataAndSaveResponse(
      game_id=game_id,
      gameData=GameData(
         game_date=str(pd.to_datetime(game_payload["game_date"]).date()),
         home_team=game_payload["home_team"],
         away_team=game_payload["away_team"],
      ),
      pitches=pitches_data,
   )
