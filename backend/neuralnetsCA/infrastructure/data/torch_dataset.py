import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Any

from domain.interfaces import DatasetAdapterInterface


class BatterDataset(Dataset):
    FEATURE_COLUMNS = [
        'release_speed', 'release_pos_x', 'plate_x', 'plate_z', 
        'launch_speed', 'launch_angle', 'effective_speed', 
        'release_spin_rate', 'release_extension', 'batting_pattern_id',
        'launch_speed_angle_id', 'hc_x', 'hc_y', 'vx0', 'vy0', 
        'vz0', 'ax', 'ay', 'az', 'estimated_ba_using_speedangle'
    ]
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)
    
    def __getitem__(self, index: int):
        row = self.df.loc[index]
        
        batter_id = int(row['batter_id']) #Ignore linting error
        pitch_type_id = int(row['pitch_type_id']) #Ignore linting error
        
        features = torch.tensor(row[self.FEATURE_COLUMNS].values.astype(np.float32), dtype=torch.float32)
        
        y = int(row['outcome_id'])
        
        return batter_id, pitch_type_id, features, y
    
    def __len__(self) -> int:
        return len(self.df)


class InferenceDataset(Dataset):
    FEATURE_COLUMNS = [
        'release_speed', 'release_pos_x', 'plate_x', 'plate_z', 
        'launch_speed', 'launch_angle', 'effective_speed', 
        'release_spin_rate', 'release_extension', 'batting_pattern_id',
        'launch_speed_angle_id', 'hc_x', 'hc_y', 'vx0', 'vy0', 
        'vz0', 'ax', 'ay', 'az', 'estimated_ba_using_speedangle'
    ]
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)
    
    def __getitem__(self, index: int):
        row = self.df.loc[index]
        
        batter_id = int(row['batter_id'])
        pitch_type_id = int(row['pitch_type_id'])
        
        features = torch.tensor(row[self.FEATURE_COLUMNS].values.astype(np.float32), dtype=torch.float32)
        
        batter_name = row['batter']
        pitch_type = row['pitch_type']
        actual_outcome = row.get('outcome_text', None)
        
        return batter_id, pitch_type_id, features, batter_name, pitch_type, actual_outcome
    
    def __len__(self) -> int:
        return len(self.df)


class TorchDatasetAdapter(DatasetAdapterInterface):
    def __init__(self):
        pass
    
    def create_dataset(self, df: pd.DataFrame, for_inference: bool = False) -> Dataset:
        if for_inference:
            return InferenceDataset(df)
        return BatterDataset(df)
    
    def create_dataloader(self, dataset: Dataset, batch_size: int, shuffle: bool = True) -> DataLoader:
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)