from fastapi import APIRouter, HTTPException
from pybaseball import statcast_single_game
import pandas as pd
import json
from app.db.database import supabase
from datetime import datetime

router = APIRouter()

@router.get("/game-data/{game_pk}")
def get_game_data_with_db(game_pk: int):
   df = statcast_single_game(game_pk)


   return fetch_game_data_and_save(game_pk)


def fetch_game_data_and_save(game_pk: int):
   df = statcast_single_game(game_pk)
   if df.empty:
      raise HTTPException(status_code=404, detail="No data found for this game ID")
   r = df.iloc[0]
   payload = {
      "game_pk": int(r["game_pk"]),
      "game_date": str(pd.to_datetime(r["game_date"]).date()),
      "home_team": r["home_team"],
      "away_team": r["away_team"],
   }
   supabase.table("games").upsert(payload, on_conflict="game_pk").execute()
   return supabase.table("games").select("*").eq("game_pk", payload["game_pk"]).single().execute().data
