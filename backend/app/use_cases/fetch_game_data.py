import random
import pandas as pd
from fastapi import HTTPException

from app.domain.interfaces import (
    GameRepositoryInterface,
    PitchRepositoryInterface,
    PlayerRepositoryInterface,
    GameDataServiceInterface,
)
from app.api.schemas.baseball import FetchGameDataAndSaveResponse, GameData, PitchData


class FetchGameDataUseCase:
    def __init__(
        self,
        game_repository: GameRepositoryInterface,
        pitch_repository: PitchRepositoryInterface,
        player_repository: PlayerRepositoryInterface,
        game_data_service: GameDataServiceInterface,
    ):
        self.game_repository = game_repository
        self.pitch_repository = pitch_repository
        self.player_repository = player_repository
        self.game_data_service = game_data_service

    def execute(self, game_pk: int) -> FetchGameDataAndSaveResponse:
        # Path 1: Data exists in DB
        game_row = self.game_repository.get_game_by_pk(game_pk)

        if game_row:
            # Game exists in the database, so fetch pitches linked to this game and their associated players
            pitches = self.pitch_repository.get_pitches_with_player_by_game_id(game_row["id"])

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

            # Return response directly from local DB without contacting external sources
            return FetchGameDataAndSaveResponse(
                game_id=game_row["id"],
                gameData=GameData(
                    game_date=pd.to_datetime(game_row["game_date"]).date(),
                    home_team=game_row["home_team"],
                    away_team=game_row["away_team"],
                ),
                pitches=pitches_data,
            )

        # Path 2: Data does NOT exist in DB, fetch externally
        # Try to get game data from external (e.g., pybaseball) service
        df = self.game_data_service.get_game_data(game_pk)

        if df is None or df.empty:
            # If there is no data at all, raise a 404 error to the client
            raise HTTPException(status_code=404, detail="No data found for this game ID")

        # Extract game-level info from the first row (all rows have the same game-level info)
        r = df.iloc[0]
        game_payload = {
            "game_pk": int(r["game_pk"]),
            "game_date": str(pd.to_datetime(r["game_date"]).date()),
            "home_team": r["home_team"],
            "away_team": r["away_team"],
        }

        # Upsert (insert or update) game in the local DB
        self.game_repository.upsert_game(game_payload)

        # Get local game id for linking
        game_id = self.game_repository.get_game_id_by_pk(game_payload["game_pk"])

        # Get all unique player names for this game, after dropping NAs and trimming whitespace
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
            # Upsert new players (assigning them random diagram indices)
            payloads = [
                {
                    "name": name,
                    "diagram_index": random.randint(0, 100),
                }
                for name in names
            ]
            self.player_repository.upsert_players(payloads)

            # Fetch player_id mapping by name for batters
            name_to_player_id = self.player_repository.get_player_id_map_by_names(names)

        # Build pitches from external source, and also collect the payloads to upsert them
        pitches_data = []
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
                diagram_index=random.randint(0, 100),
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
        # Upsert the new pitches into the local DB
        self.pitch_repository.upsert_pitches(pitch_payloads)

        # Return the response with the newly gathered and saved data
        return FetchGameDataAndSaveResponse(
            game_id=game_id,
            gameData=GameData(
                game_date=str(pd.to_datetime(game_payload["game_date"]).date()),
                home_team=game_payload["home_team"],
                away_team=game_payload["away_team"],
            ),
            pitches=pitches_data,
        )

