import sys
import os
import time
import pandas as pd
from datetime import datetime, timedelta

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(1, parent_dir)

from backend.infrastructure.supabase_repository import SupabaseRepository

class DatasetLoaderService:
    def __init__(self):
        self.repo = SupabaseRepository()

    def get_training_dataset(self) -> pd.DataFrame:
        date_ranges = [
            ("2023-03-30", "2023-10-01"),
            ("2024-03-28", "2024-09-29")
        ]
        
        dataframes = []
        for start_date_str, end_date_str in date_ranges:
            start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
            end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
            
            current_date = start_date
            while current_date <= end_date:
                date_str = current_date.strftime("%Y-%m-%d")
                print(f"Fetching data for {date_str}...")
                df = self.repo.get_pitch_data_rows('pitch_data_raw', date_str, date_str)
                if not df.empty:
                    dataframes.append(df)
                time.sleep(0.5)  
                current_date += timedelta(days=1)
        
        full_df = pd.concat(dataframes, ignore_index=True) if dataframes else pd.DataFrame()
        
        return full_df