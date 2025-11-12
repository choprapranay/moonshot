from fastapi import APIRouter, HTTPException

from app.schemas.game_analysis import GameAnalysisResponse
from app.services.game_analysis import build_game_analysis, build_pitch_records


router = APIRouter()


@router.get("/game/{game_pk}")
def get_game_data(game_pk: int):
    try:
        return build_pitch_records(game_pk)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/game/{game_pk}/analysis", response_model=GameAnalysisResponse)
def get_game_analysis(game_pk: int):
    try:
        return build_game_analysis(game_pk)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
