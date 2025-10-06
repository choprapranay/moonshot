from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pybaseball import statcast_single_game
import pandas as pd
import json

router = APIRouter()

@router.get("/game/{game_pk}")
def get_game_data(game_pk: int):
    try:
        df = statcast_single_game(game_pk)

        if df.empty:
            raise HTTPException(status_code=404, detail="No data found for this game ID")

        clean_df = (
            df.replace([float("inf"), float("-inf")], pd.NA)
              .where(pd.notnull(df), None)
        )

        return JSONResponse(content=json.loads(clean_df.to_json(orient="records")))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
