
import random
from fastapi import APIRouter, HTTPException
import pandas as pd
from ..api.games import get_game_data
from app.repositories.games_repository import get_game_by_pk, upsert_game, get_game_id_by_pk
from app.repositories.players_repository import upsert_players, get_player_id_map_by_names
from app.repositories.pitches_repository import get_pitches_with_player_by_game_id, upsert_pitches
from app.schemas.baseball import FetchGameDataAndSaveResponse, GameData, PitchData

router = APIRouter()

@router.get("/game-data/{game_pk}")
def get_game_data_with_db(game_pk: int):
   return fetch_game_data_and_save(game_pk)

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

   game_row = get_game_by_pk(game_pk)

   # this means game data already exists in games table
   if game_row:
      # fetch pitches data and player data per pitch
      pitches = get_pitches_with_player_by_game_id(game_row["id"])

      pitches_data = []
      for pitch in pitches:
         player = pitch.get("players") or {}
         pitches_data.append(
            PitchData(
               pitch_type=pitch.get("pitch_type") or "",
               speed=float(pitch["speed"]) if pitch.get("speed") is not None else None,
               description=pitch.get("description"),
               player_name=player.get("name"),
               diagram_index=player.get("diagram_index"),
            )
         )

      # return game data and pitches data
      return FetchGameDataAndSaveResponse(
         game_id=game_row["id"],
         gameData=GameData(
            game_date=pd.to_datetime(game_row["game_date"]).date(),
            home_team=game_row["home_team"],
            away_team=game_row["away_team"],
         ),
         pitches=pitches_data,
      )
   # if game data does not exist, extract game data and save to games table
   # call pybaseball api
   df = get_game_data(game_pk)

   # remember don't use if not df and use if df is None because dataframe has ambiguous boolean value
   if df is None or df.empty:
      raise HTTPException(status_code=404, detail="No data found for this game ID")

   # if game data does not exist, extract game data and save to games table
   # in a data frame, the columns and rows are reverted

   # import pandas as pd
   # df = pd.DataFrame({
   #     "name": ["Alice", "Bob", "Charlie"],
   #     "age": [25, 30, 35]
   # })
   # print(df)

   # this turns into

   #   name      age
   # 0  Alice     25
   # 1  Bob       30
   # 2  Charlie   35

   # iloc fetches the row, while regular indexing fetches the column
   # so this line fetches the first data row
   r = df.iloc[0]
   game_payload = {
      "game_pk": int(r["game_pk"]),
      "game_date": str(pd.to_datetime(r["game_date"]).date()),
      "home_team": r["home_team"],
      "away_team": r["away_team"],
   }
   
   upsert_game(game_payload)

   game_id = get_game_id_by_pk(game_payload["game_pk"])

   # extract player data to upload to player table
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
      # upsert missing players to player table
      payloads = [
         {
            "name": name,
            "diagram_index": random.randint(0, 100), # TEMPORARY
         }
         for name in names
      ]
      upsert_players(payloads)

      # get the player ids per player name
      name_to_player_id = get_player_id_map_by_names(names)
   
      
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
      }
      pitches_data.append(PitchData(
         pitch_type=str(row["pitch_type"]) if pd.notna(row["pitch_type"]) else "",
         speed=float(row["release_speed"]) if pd.notna(row["release_speed"]) else None,
         description=str(row["description"]) if pd.notna(row["description"]) else None,
         player_name=player_name,
         diagram_index=random.randint(0, 100), # TEMPORARY
      ))
   pitch_payloads = [
      {
         "game_id": game_id,
         "batter_id": name_to_player_id.get(pitch.player_name),
         "pitch_type": pitch.pitch_type,
         "speed": pitch.speed,
         "description": pitch.description,
      }
      for pitch in pitches_data
   ]
   upsert_pitches(pitch_payloads)


   return FetchGameDataAndSaveResponse(
      game_id=game_id,
      gameData=GameData(
         game_date=str(pd.to_datetime(game_payload["game_date"]).date()),
         home_team=game_payload["home_team"],
         away_team=game_payload["away_team"],
      ),
      pitches=pitches_data,
   )
