from fastapi import APIRouter, HTTPException

from app.api.schemas.game_analysis import GameAnalysisResponse, PlayerHeatmapResponse
from app.use_cases.game_analysis import (
    BuildGameAnalysisUseCase,
    BuildPitchRecordsUseCase,
    GetBatterSwingsUseCase,
)
from app.infrastructure.repositories.heatmap_repository import get_heatmap_for_batter
from app.infrastructure.adapters.encoder_adapter import encode_batter_id


router = APIRouter()


@router.get("/game/{game_pk}")
def get_game_data(game_pk: int):
    try:
        use_case = BuildPitchRecordsUseCase()
        return use_case.execute(game_pk)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/game/{game_pk}/analysis", response_model=GameAnalysisResponse)
def get_game_analysis(game_pk: int):
    try:
        use_case = BuildGameAnalysisUseCase()
        return use_case.execute(game_pk)
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
            try:
                heatmap = get_heatmap_for_batter(encoded_batter_id)
            except HTTPException:
                # If heatmap file doesn't exist, just return empty list
                heatmap = []
            except Exception:
                # If any other error with heatmap, return empty list
                heatmap = []
        swings_use_case = GetBatterSwingsUseCase()
        swings = swings_use_case.execute(game_pk, batter_id)
        return {"heatmap": heatmap, "swings": swings}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
