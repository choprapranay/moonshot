from typing import Any, Dict, Iterable, List

from app.db.database import get_supabase_client


def upsert_players(payloads: List[Dict[str, Any]]) -> None:
   supabase = get_supabase_client()
   supabase.table("players").upsert(payloads, on_conflict="name").execute()


def get_player_id_map_by_names(names: Iterable[str]) -> Dict[str, str]:
   supabase = get_supabase_client()
   rows = (
      supabase
         .table("players")
         .select("id, name")
         .in_("name", list(names))
         .execute()
         .data
   )
   return {row["name"]: row["id"] for row in rows}


