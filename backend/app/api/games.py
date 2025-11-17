from fastapi import APIRouter, HTTPException

from app.schemas.game_analysis import GameAnalysisResponse, PlayerHeatmapResponse
from app.services.game_analysis import (
    build_game_analysis,
    build_pitch_records,
    get_batter_swings,
)
from app.services.heatmap import get_heatmap_for_batter
from app.services.encoders import encode_batter_id


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


@router.get(
    "/game/{game_pk}/player/{batter_id}/heatmap",
    response_model=PlayerHeatmapResponse,
)
def get_player_heatmap(game_pk: int, batter_id: int):
    try:
        encoded_batter_id = encode_batter_id(batter_id)
        heatmap = []
        if encoded_batter_id is not None:
            heatmap = get_heatmap_for_batter(encoded_batter_id)
        swings = get_batter_swings(game_pk, batter_id)
        return {"heatmap": heatmap, "swings": swings}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
