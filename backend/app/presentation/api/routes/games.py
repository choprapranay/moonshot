from fastapi import APIRouter, HTTPException
import pandas as pd
import json
from datetime import datetime

from pydantic import BaseModel
from app.infrastructure.database.database import supabase
from app.application.use_cases.game_operations import GameOperations
from pybaseball import statcast_single_game
from fastapi.responses import JSONResponse

router = APIRouter()

game_ops = GameOperations(supabase)

class TmpItem(BaseModel):
    name: str

@router.get("/game/{game_pk}")
def get_game_data(game_pk: int):
    try:
        df = statcast_single_game(game_pk)

        if df.empty:
            raise HTTPException(status_code=404, detail="No data found for this game ID")

        clean_df = (
            df.replace([float("inf"), float("-inf")], pd.NA)
              .where(pd.notnull(df), pd.NA)
        )

        return JSONResponse(content=json.loads(clean_df.to_json(orient="records")))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@router.post("/addToTmpData")
def add_to_tmp_data(payload: TmpItem):
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
    try:
        res = supabase.table("games").select("id,game_date,home_team,away_team").order("id", desc=True).execute()
        return res.data or []
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
