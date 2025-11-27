from typing import Any, Dict, List

from app.infrastructure.db.database import supabase
from app.domain.interfaces import PitchRepositoryInterface


class SupabasePitchRepository(PitchRepositoryInterface):
    def get_pitches_with_player_by_game_id(self, game_id: str) -> List[Dict[str, Any]]:
        return (
            supabase
            .table("pitches")
            .select("*, players(name, diagram_index)")
            .eq("game_id", game_id)
            .execute()
            .data
        )

    def upsert_pitches(self, payloads: List[Dict[str, Any]]) -> None:
        supabase.table("pitches").upsert(payloads).execute()

