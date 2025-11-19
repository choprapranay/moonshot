from typing import Any, Dict, List

from app.db.database import get_supabase_client


def get_pitches_with_player_by_game_id(game_id: str) -> List[Dict[str, Any]]:
   supabase = get_supabase_client()
   return (
      supabase
         .table("pitches")
         .select("*, players(name, diagram_index)")
         .eq("game_id", game_id)
         .execute()
         .data
   )


def upsert_pitches(payloads: List[Dict[str, Any]]) -> None:
   supabase = get_supabase_client()
   supabase.table("pitches").upsert(payloads).execute()


