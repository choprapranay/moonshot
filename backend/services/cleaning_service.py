import numpy as np
import pandas as pd
from services.labeling_service import LabelingService

class CleaningService:
    @staticmethod
    def fix_na(df: pd.DataFrame) -> pd.DataFrame:

        # Account for Duplicated Columns (bug fix for loading data)
        df = df.loc[:, ~df.columns.duplicated()].copy()

        # Cleaning NA
        both_na = df['hc_x'].isna() & df['hc_y'].isna()
        both_na_2 = df['launch_speed'].isna() & df['launch_angle'].isna()
        both_na_3 = df['launch_speed_angle'].isna() & df['estimated_ba_using_speedangle'].isna()

        df.loc[both_na, ['hc_x', 'hc_y']] = 0
        df.loc[both_na_2, ['launch_speed', 'launch_angle']] = 0
        df.loc[both_na_3, ['launch_speed_angle', 'estimated_ba_using_speedangle']] = 0

        # Create outcome_text column
        df['outcome_text'] = df['description'].fillna(df['events'])
        # Applying Labeling Service
        df['swing_take'] = df['outcome_text'].apply(LabelingService.swing_type)

        # Replace infinite values with 0 (Supabase/JSON can't handle inf)
        df = df.replace([np.inf, -np.inf], 0)

        # Drop remaining rows with any NaN (clean dataset for ML)
        df = df.dropna()

        # Convert all datetime columns to string for JSON serialization
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = df[col].astype(str)

        return df