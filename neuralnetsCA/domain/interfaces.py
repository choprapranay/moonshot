from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, List, Callable
import pandas as pd
import numpy as np

from .entities import Model, ModelArtifacts, TrainingConfig
class DataAccessInterface(ABC):
    @abstractmethod
    def fetch_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        pass
    
    @abstractmethod
    def get_source_name(self) -> str:
        pass


class PreprocessorInterface(ABC):
    @abstractmethod
    def preprocess_for_training(self, raw_data: pd.DataFrame) -> Tuple[pd.DataFrame, ModelArtifacts]:
        pass
    
    @abstractmethod
    def preprocess_for_inference(self,  raw_data: pd.DataFrame,  artifacts: ModelArtifacts) -> pd.DataFrame:
        pass
    
    @abstractmethod
    def get_feature_columns(self) -> List[str]:
        pass


class ModelInterface(ABC):
    @abstractmethod
    def create_model(self, num_batters: int, num_pitch_types: int, num_outcomes: int, device: str = 'cpu') -> Any:
        pass
    
    @abstractmethod
    def train(self, model: Any, train_data: pd.DataFrame, test_data: pd.DataFrame, config: TrainingConfig, progress_callback: Optional[Callable] = None) -> Tuple[Dict[str, Any], List[Dict[str, float]]]:
        pass
    
    @abstractmethod
    def predict(self, model: Any, input_data: pd.DataFrame, artifacts: ModelArtifacts, device: str = 'cpu') -> List[Dict[str, Any]]:
        pass
    
    @abstractmethod
    def load_model_state(self, model: Any, state_dict: Dict[str, Any]) -> Any:
        pass

class ModelStorageInterface(ABC):
    @abstractmethod
    def save_model(self, model: Model, path: str) -> bool:
        pass
    
    @abstractmethod
    def load_model(self, path: str) -> Optional[Model]:
        pass
    

class DatasetAdapterInterface(ABC):
    @abstractmethod
    def create_dataset(self, df: pd.DataFrame) -> Any:
        pass
    
    @abstractmethod
    def create_dataloader(self,  dataset: Any, batch_size: int,  shuffle: bool = True) -> Any:
        pass