from typing import Any, Dict, Optional

from app.db.database import get_supabase_client


def get_game_by_pk(game_pk: int) -> Optional[Dict[str, Any]]:
   supabase = get_supabase_client()
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


def upsert_game(payload: Dict[str, Any]) -> None:
   supabase = get_supabase_client()
   supabase.table("games").upsert(payload, on_conflict="game_pk").execute()


def get_game_id_by_pk(game_pk: int) -> str:
   supabase = get_supabase_client()
   return (
      supabase
         .table("games")
         .select("id")
         .eq("game_pk", game_pk)
         .single()
         .execute()
         .data["id"]
   )


