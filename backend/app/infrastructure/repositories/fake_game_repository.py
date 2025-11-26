from typing import Any, Dict, Optional

from app.domain.interfaces import GameRepositoryInterface


class FakeGameRepository(GameRepositoryInterface):
    def __init__(self):
        self.games = {}

    def get_game_by_pk(self, game_pk: int) -> Optional[Dict[str, Any]]:
        return self.games.get(game_pk)

    def upsert_game(self, payload: Dict[str, Any]) -> None:
        game_pk = payload["game_pk"]
        self.games[game_pk] = {
            "id": payload.get("id", f"game_{game_pk}"),
            "game_date": payload["game_date"],
            "home_team": payload["home_team"],
            "away_team": payload["away_team"],
        }

    def get_game_id_by_pk(self, game_pk: int) -> str:
        game = self.games.get(game_pk)
        if game:
            return game["id"]
        return f"game_{game_pk}"

