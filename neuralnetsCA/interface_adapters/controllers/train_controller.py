from typing import Optional
import pandas as pd

from domain.boundaries import TrainModelInputBoundary, TrainModelInputData
from domain.entities import TrainingConfig


class TrainController:
    def __init__(self, train_use_case: TrainModelInputBoundary):
        self._use_case = train_use_case
    
    def train_model(self, dataset: pd.DataFrame, epochs: int = 20, learning_rate: float = 0.0001, batch_size: int = 32, test_split: float = 0.2, save_path: str = 'batter_outcome_model.pth') -> None:
        config = TrainingConfig(
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            test_split=test_split,
            save_path=save_path
        )
        
        if not config.validate():
            raise ValueError("Invalid training configuration")
        
        input_data = TrainModelInputData(config=config, dataset=dataset)
        
        self._use_case.execute(input_data)
    
    def train_with_config(self, dataset: pd.DataFrame, config: TrainingConfig) -> None:
        if not config.validate():
            raise ValueError("Invalid training configuration")
        
        input_data = TrainModelInputData(config=config, dataset=dataset)
        
        self._use_case.execute(input_data)
