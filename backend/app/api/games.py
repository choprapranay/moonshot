from fastapi import APIRouter, HTTPException
import pandas as pd
import json
from datetime import datetime

from pydantic import BaseModel
from app.db.database import supabase
from pybaseball import statcast_single_game
from fastapi.responses import JSONResponse

router = APIRouter()

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

        return clean_df

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
