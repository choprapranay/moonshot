import pandas as pd
from infrastructure.supabase_repository import SupabaseRepository

class DatasetLoaderService:
    def __init__(self):
        self.repo = SupabaseRepository()

    def get_training_dataset(self) -> pd.DataFrame:
        df_2023 = self.repo.fetch_pitch_data("2023-03-30", "2023-10-01")
        df_2024 = self.repo.fetch_pitch_data("2024-03-28", "2024-09-29")

        # Merge into one dataframe
        full_df = pd.concat([df_2023, df_2024], ignore_index=True)

        return full_df