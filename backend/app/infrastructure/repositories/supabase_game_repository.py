from typing import Any, Dict, Optional

from app.infrastructure.db.database import supabase
from app.domain.interfaces import GameRepositoryInterface


class SupabaseGameRepository(GameRepositoryInterface):
    def get_game_by_pk(self, game_pk: int) -> Optional[Dict[str, Any]]:
        resp = (
            supabase
            .table("games")
            .select("id, game_date, home_team, away_team")
            .eq("game_pk", game_pk)
            .maybe_single()
            .execute()
        )
        if resp and getattr(resp, "data", None):
            return resp.data
        return None

    def upsert_game(self, payload: Dict[str, Any]) -> None:
        supabase.table("games").upsert(payload, on_conflict="game_pk").execute()

    def get_game_id_by_pk(self, game_pk: int) -> str:
        return (
            supabase
            .table("games")
            .select("id")
            .eq("game_pk", game_pk)
            .single()
            .execute()
            .data["id"]
        )

