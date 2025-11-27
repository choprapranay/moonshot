from typing import Any, Dict, Iterable, List

from app.infrastructure.db.database import supabase
from app.domain.interfaces import PlayerRepositoryInterface


class SupabasePlayerRepository(PlayerRepositoryInterface):
    def upsert_players(self, payloads: List[Dict[str, Any]]) -> None:
        supabase.table("players").upsert(payloads, on_conflict="name").execute()

    def get_player_id_map_by_names(self, names: Iterable[str]) -> Dict[str, str]:
        rows = (
            supabase
            .table("players")
            .select("id, name")
            .in_("name", list(names))
            .execute()
            .data
        )
        return {row["name"]: row["id"] for row in rows}

