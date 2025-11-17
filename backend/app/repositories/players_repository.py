from typing import Any, Dict, Iterable, List

from app.db.database import supabase


def upsert_players(payloads: List[Dict[str, Any]]) -> None:
   supabase.table("players").upsert(payloads, on_conflict="name").execute()


def get_player_id_map_by_names(names: Iterable[str]) -> Dict[str, str]:
   rows = (
      supabase
         .table("players")
         .select("id, name")
         .in_("name", list(names))
         .execute()
         .data
   )
   return {row["name"]: row["id"] for row in rows}


