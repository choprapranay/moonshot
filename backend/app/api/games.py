from fastapi import APIRouter, HTTPException
from datetime import datetime
from app.schemas.game import TmpItem
from app.db.database import supabase

router = APIRouter()

@router.post("/addToTmpData")
def add_to_tmp_data(payload: TmpItem):
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase not configured")
    try:
        res = supabase.table("games").insert({
            "game_date": datetime.now().strftime("%Y-%m-%d"),
            "home_team": payload.name,
            "away_team": payload.name
        }).execute()
        return {"ok": True, "count": getattr(res, "count", None)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/retrieveFromTmpData")
def retrieve_from_tmp_data():
    if not supabase:
        raise HTTPException(status_code=500, detail="Supabase not configured")
    try:
        res = supabase.table("games").select("id,game_date,home_team,away_team").order("id", desc=True).execute()
        return res.data or []
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
