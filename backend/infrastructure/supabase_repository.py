import pandas as pd
from .supabase_client import SupabaseClient

class SupabaseRepository:

    def __init__(self):
        self.db = SupabaseClient().client

    def insert_pitch_data_rows(self, table: str, rows: list[dict]):
        self.db.table(table).insert(rows).execute()

    def get_pitch_data_rows(self, table: str, start_date: str, end_date: str) -> pd.DataFrame:
        res = (
            self.db.table(table)
            .select("*")
            .gte("game_date", start_date)
            .lte("game_date", end_date)
            .execute()
        )
        return pd.DataFrame(res.data)