# Imports
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pybaseball import statcast_single_game
import pandas as pd



app = FastAPI()

# Allow frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/game/{game_pk}")
def get_game_data(game_pk: int):
    try:
        df = statcast_single_game(game_pk)

        if df.empty:
            raise HTTPException(status_code=404, detail="No data found for this game ID")

        # Clean DF
        clean_df = (
            df.replace([float("inf"), float("-inf")], pd.NA)
              .where(pd.notnull(df), None)
        )

        return JSONResponse(content=json.loads(clean_df.to_json(orient="records")))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
