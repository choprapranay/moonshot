from typing import List
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from domain.interfaces import DataPreprocessorInterface
from domain.entities import DatasetSplit

def swing_type(outcome_text: str):
    out_text = outcome_text.lower()
    swing_pattern = [r'field_out', r'single', r'double', r'triple',
                     r'home_run', r'grounded_into_double_play', r'force_out', 
                     r'sac_fly', r'field_error', r'fielders_choice', r'fielders_choice_out',
                     r'double_play', r'triple_play', r'swinging_strike', r'foul', r'foul_tip', r'swinging_strike_blocked']
    take_pattern = [r'ball', r'walk', r'hit_by_pitch', r'called_strike', r'called_strike', r'blocked_ball'] 

    for p in swing_pattern:
        if p in out_text:
            return 1
    for p in take_pattern:
        if p in out_text:
            return 0
    return None

class DataPreprocessor(DataPreprocessorInterface):
    def preprocess(self, df: pd.DataFrame) -> DatasetSplit:
        shortened = df[[
            'batter', 'pitch_type', 'description', 'plate_x', 'plate_z',
            'events', 'release_speed', 'release_pos_x', 'launch_speed',
            'launch_angle', 'effective_speed', 'release_spin_rate',
            'release_extension', 'hc_x', 'hc_y', 'vx0', 'vy0', 'vz0',
            'ax', 'ay', 'az', 'sz_top', 'sz_bot',
            'estimated_ba_using_speedangle', 'launch_speed_angle',
        ]].copy()

        both_na = shortened['hc_x'].isna() & shortened['hc_y'].isna()
        both_na_2 = shortened['launch_speed'].isna() & shortened['launch_angle'].isna()
        both_na_3 = shortened['launch_speed_angle'].isna() & shortened['estimated_ba_using_speedangle'].isna()

        shortened['outcome_text'] = shortened['events']
        shortened['outcome_text'] = shortened['description'].where(both_na, shortened['events'])

        shortened.loc[both_na, 'hc_x'] = 0
        shortened.loc[both_na, 'hc_y'] = 0
        shortened.loc[both_na_2, 'launch_speed'] = 0
        shortened.loc[both_na_2, 'launch_angle'] = 0
        shortened.loc[both_na_3, 'launch_speed_angle'] = 0
        shortened.loc[both_na_3, 'estimated_ba_using_speedangle'] = 0

        shortened = shortened.dropna(subset=['outcome_text'])

        shortened['batting_pattern'] = shortened['outcome_text'].apply(swing_type)
        shortened = shortened.dropna(subset=['batting_pattern'])

        base_feature_cols = [
            'release_speed', 'release_pos_x', 'plate_x', 'plate_z',
            'launch_speed', 'launch_angle', 'effective_speed',
            'release_spin_rate', 'release_extension',
            'hc_x', 'hc_y', 'vx0', 'vy0', 'vz0', 'ax', 'ay', 'az',
            'estimated_ba_using_speedangle',
        ]

        shortened = shortened.dropna(subset=base_feature_cols)

        batter_enc = LabelEncoder()
        shortened['batter_id'] = batter_enc.fit_transform(shortened['batter'])

        pitch_enc = LabelEncoder()
        shortened['pitch_type_id'] = pitch_enc.fit_transform(shortened['pitch_type'])

        outcome_enc = LabelEncoder()
        shortened['outcome_id'] = outcome_enc.fit_transform(shortened['outcome_text'])

        batting_pattern_enc = LabelEncoder()
        shortened['batting_pattern_id'] = batting_pattern_enc.fit_transform(shortened['batting_pattern'])

        launch_speed_angle_enc = LabelEncoder()
        shortened['launch_speed_angle_id'] = launch_speed_angle_enc.fit_transform(shortened['launch_speed_angle'])

        scaler = StandardScaler()
        shortened[base_feature_cols] = scaler.fit_transform(shortened[base_feature_cols].astype(float))

        NUM_BATTERS = shortened['batter'].nunique()
        NUM_PITCHES = shortened['pitch_type_id'].nunique()
        NUM_OUTCOMES = shortened['outcome_id'].nunique()
        NUM_BATTER_PATTERNS = shortened['batting_pattern_id'].nunique()

        inf_count = np.isinf(shortened[base_feature_cols]).sum().sum()
        nan_count = np.isnan(shortened[base_feature_cols]).sum().sum()

        if inf_count > 0 or nan_count > 0:
            shortened = shortened.replace([np.inf, -np.inf], np.nan)
            shortened = shortened.dropna(subset=base_feature_cols)
        
        train_df, test_df = train_test_split(shortened, test_size=0.2, random_state=42)

        return self._build_dataset_split_from_dfs(train_df, test_df, base_feature_cols, NUM_BATTERS, NUM_PITCHES, NUM_OUTCOMES)
    def _build_dataset_split_from_dfs(self, train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols, num_batters: int, num_pitch_types: int, num_outcomes: int) -> DatasetSplit:
        batter_ids_train = train_df['batter_id'].to_numpy(dtype=np.int64)
        pitch_type_ids_train = train_df['pitch_type_id'].to_numpy(dtype=np.int64)
        y_train = train_df['outcome_id'].to_numpy(dtype=np.int64)
        features_train = train_df[feature_cols].to_numpy(dtype=np.float32)

        # val
        batter_ids_val = test_df['batter_id'].to_numpy(dtype=np.int64)
        pitch_type_ids_val = test_df['pitch_type_id'].to_numpy(dtype=np.int64)
        y_val = test_df['outcome_id'].to_numpy(dtype=np.int64)
        features_val = test_df[feature_cols].to_numpy(dtype=np.float32)

        return DatasetSplit(
            batter_ids_train=batter_ids_train,
            pitch_type_ids_train=pitch_type_ids_train,
            features_train=features_train,
            y_train=y_train,
            batter_ids_val=batter_ids_val,
            pitch_type_ids_val=pitch_type_ids_val,
            features_val=features_val,
            y_val=y_val,
            num_batters=num_batters,
            num_pitch_types=num_pitch_types,
            num_outcomes=num_outcomes,
        )