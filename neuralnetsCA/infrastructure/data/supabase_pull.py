from backend.infrastructure.supabase_repository import SupabaseRepository #replace with sys
from domain.interfaces import PitchEventRepositoryInterface
import pandas as pd

supabase_repo = SupabaseRepository()

class SupabasePitchEventRepository(PitchEventRepositoryInterface):
    def __init__(self, supabase_repo: SupabaseRepository):
        self.supabase_repo = supabase_repo
    def get_events(self, start_date: str, end_date: str) -> pd.DataFrame:
        return self.supabase_repo.get_pitch_data_rows(
            table="pitch_data",
            start_date=start_date,
            end_date=end_date
        )
