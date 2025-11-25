from typing import List, Tuple
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

from domain.interfaces import PreprocessorInterface
from domain.entities import ModelArtifacts


class PreprocessorAdapter(PreprocessorInterface):
    FEATURE_COLUMNS = [
        'release_speed', 'release_pos_x', 'plate_x', 'plate_z', 
        'launch_speed', 'launch_angle', 'effective_speed', 
        'release_spin_rate', 'release_extension', 'hc_x', 'hc_y',
        'vx0', 'vy0', 'vz0', 'ax', 'ay', 'az', 'estimated_ba_using_speedangle'
    ]
    
    MODEL_INPUT_COLUMNS = [
        'release_speed', 'release_pos_x', 'plate_x', 'plate_z', 
        'launch_speed', 'launch_angle', 'effective_speed', 
        'release_spin_rate', 'release_extension', 'batting_pattern_id',
        'launch_speed_angle_id', 'hc_x', 'hc_y', 'vx0', 'vy0', 
        'vz0', 'ax', 'ay', 'az', 'estimated_ba_using_speedangle'
    ]
    
    REQUIRED_COLUMNS = [
        'batter', 'pitch_type', 'description', 'plate_x', 'plate_z', 
        'events', 'release_speed', 'release_pos_x', 'launch_speed', 
        'launch_angle', 'effective_speed', 'release_spin_rate', 
        'release_extension', 'hc_x', 'hc_y', 'vx0', 'vy0', 'vz0', 
        'ax', 'ay', 'az', 'estimated_ba_using_speedangle', 'launch_speed_angle'
    ]
    
    SWING_PATTERNS = [
        'field_out', 'single', 'double', 'triple', 'home_run',
        'grounded_into_double_play', 'force_out', 'sac_fly', 
        'field_error', 'fielders_choice', 'fielders_choice_out',
        'double_play', 'triple_play', 'swinging_strike', 'foul', 
        'foul_tip', 'swinging_strike_blocked'
    ]
    
    TAKE_PATTERNS = [
        'ball', 'walk', 'hit_by_pitch', 'called_strike', 'blocked_ball'
    ]
    
    def __init__(self):
        pass
    
    @staticmethod
    def swing_type(outcome_text: str) -> int | None:
        if outcome_text is None or (isinstance(outcome_text, float) and pd.isna(outcome_text)):
            return None
        
        if not isinstance(outcome_text, str):
            return None
            
        out_text = outcome_text.lower()
        
        for pattern in PreprocessorAdapter.SWING_PATTERNS:
            if pattern in out_text:
                return 1
                
        for pattern in PreprocessorAdapter.TAKE_PATTERNS:
            if pattern in out_text:
                return 0
                
        return None
    
    def get_feature_columns(self) -> List[str]:
        return self.FEATURE_COLUMNS.copy()
    
    def _select_and_validate_columns(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        available_cols = [col for col in self.REQUIRED_COLUMNS if col in raw_data.columns]
        return raw_data[available_cols].copy()
    
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        
        both_na = df['hc_x'].isna() & df['hc_y'].isna()
        df.loc[both_na, 'hc_x'] = 0
        df.loc[both_na, 'hc_y'] = 0
        
        both_na_2 = df['launch_speed'].isna() & df['launch_angle'].isna()
        df.loc[both_na_2, 'launch_speed'] = 0
        df.loc[both_na_2, 'launch_angle'] = 0
        
        both_na_3 = df['launch_speed_angle'].isna() & df['estimated_ba_using_speedangle'].isna()
        df.loc[both_na_3, 'launch_speed_angle'] = 0
        df.loc[both_na_3, 'estimated_ba_using_speedangle'] = 0
        
        return df
    
    def _create_outcome_text(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        both_na = df['hc_x'].isna() & df['hc_y'].isna()
        df['outcome_text'] = df['description'].where(both_na, df['events'])
        return df
    
    def _apply_batting_pattern(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        df['batting_pattern'] = df['outcome_text'].apply(self.swing_type)
        return df
    
    def _remove_inf_nan(self, data: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        df = data.copy()
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=columns)
        return df
    
    def preprocess_for_training(self, raw_data: pd.DataFrame) -> Tuple[pd.DataFrame, ModelArtifacts]:
        df = self._select_and_validate_columns(raw_data)
        
        df = self._create_outcome_text(df)
        df = df.dropna(subset=['outcome_text'])
        
        df = self._handle_missing_values(df)
        
        df = self._apply_batting_pattern(df)
        df = df.dropna(subset=['batting_pattern'])
        
        df = df.dropna(subset=self.FEATURE_COLUMNS)
        
        batter_enc = LabelEncoder()
        df['batter_id'] = batter_enc.fit_transform(df['batter'])
        
        pitch_enc = LabelEncoder()
        df['pitch_type_id'] = pitch_enc.fit_transform(df['pitch_type'])
        
        outcome_enc = LabelEncoder()
        df['outcome_id'] = outcome_enc.fit_transform(df['outcome_text'])
        
        batter_pattern_enc = LabelEncoder()
        df['batting_pattern_id'] = batter_pattern_enc.fit_transform(df['batting_pattern'])
        
        launch_speed_angle_enc = LabelEncoder()
        df['launch_speed_angle_id'] = launch_speed_angle_enc.fit_transform(df['launch_speed_angle'])
        
        scaler = StandardScaler()
        df[self.FEATURE_COLUMNS] = scaler.fit_transform(df[self.FEATURE_COLUMNS].astype(float))
        
        df = self._remove_inf_nan(df, self.FEATURE_COLUMNS)
        
        artifacts = ModelArtifacts(
            batter_encoder=batter_enc,
            pitch_encoder=pitch_enc,
            outcome_encoder=outcome_enc,
            batter_pattern_encoder=batter_pattern_enc,
            launch_speed_angle_encoder=launch_speed_angle_enc,
            scaler=scaler,
            labels=list(outcome_enc.classes_),
            num_batters=df['batter'].nunique(),
            num_pitches=df['pitch_type_id'].nunique(),
            num_outcomes=df['outcome_id'].nunique()
        )
        
        return df, artifacts
    
    def preprocess_for_inference(self, raw_data: pd.DataFrame, artifacts: ModelArtifacts) -> pd.DataFrame:
        df = self._select_and_validate_columns(raw_data)
       
        if 'outcome_text' not in df.columns:
            df = self._create_outcome_text(df)
        
        df = self._handle_missing_values(df)
        
        df = self._apply_batting_pattern(df)
        df = df.dropna(subset=['batting_pattern'])
        
        required = self.FEATURE_COLUMNS + ['batter', 'pitch_type']
        available_required = [col for col in required if col in df.columns]
        df = df.dropna(subset=available_required)
        
       
        known_batters = set(artifacts.batter_encoder.classes_)
        known_pitches = set(artifacts.pitch_encoder.classes_)
        
        df = df[df['batter'].isin(known_batters)]
        df = df[df['pitch_type'].isin(known_pitches)]
        
        if df.empty:
            raise ValueError("No valid samples after filtering for known batters/pitch types")
        
        df['batter_id'] = artifacts.batter_encoder.transform(df['batter'])
        df['pitch_type_id'] = artifacts.pitch_encoder.transform(df['pitch_type'])
        df['batting_pattern_id'] = artifacts.batter_pattern_encoder.transform(df['batting_pattern'])
        
        known_lsa = set(artifacts.launch_speed_angle_encoder.classes_)
        df['launch_speed_angle'] = df['launch_speed_angle'].apply(
            lambda x: x if x in known_lsa else list(known_lsa)[0]
        )
        df['launch_speed_angle_id'] = artifacts.launch_speed_angle_encoder.transform(df['launch_speed_angle'])
        
        df[self.FEATURE_COLUMNS] = artifacts.scaler.transform(df[self.FEATURE_COLUMNS].astype(float))
        
        df = self._remove_inf_nan(df, self.FEATURE_COLUMNS)
        
        return df