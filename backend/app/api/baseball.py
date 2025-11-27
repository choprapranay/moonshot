from fastapi import APIRouter

from app.use_cases.fetch_game_data import FetchGameDataUseCase
from app.infrastructure.repositories.supabase_game_repository import SupabaseGameRepository
from app.infrastructure.repositories.supabase_pitch_repository import SupabasePitchRepository
from app.infrastructure.repositories.supabase_player_repository import SupabasePlayerRepository
from app.infrastructure.repositories.pybaseball_game_data_service import PybaseballGameDataService

router = APIRouter()

@router.get("/game-data/{game_pk}")
def get_game_data_with_db(game_pk: int):
   game_repo = SupabaseGameRepository()
   pitch_repo = SupabasePitchRepository()
   player_repo = SupabasePlayerRepository()
   game_data_service = PybaseballGameDataService()
   use_case = FetchGameDataUseCase(game_repo, pitch_repo, player_repo, game_data_service)
   return use_case.execute(game_pk)

