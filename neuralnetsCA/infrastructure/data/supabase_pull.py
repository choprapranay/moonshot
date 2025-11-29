import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

import pandas as pd
from backend.infrastructure.supabase_repository import SupabaseRepository

from domain.interfaces import DataAccessInterface


class SupabaseDataAccess(DataAccessInterface):
    def __init__(self):
        self._repo = SupabaseRepository()
        self._source_name = "supabase"
    
    def fetch_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        try:
            data = self._repo.get_pitch_data_rows(
                table="pitch_data",
                start_date=start_date,
                end_date=end_date
            )
            return data
        except Exception as e:
            print(f"Error fetching data from Supabase: {e}")
            return pd.DataFrame()
    
    def get_source_name(self) -> str:
        return self._source_name