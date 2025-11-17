from abc import ABC, abstractmethod
from typing import Any, Dict
import pandas as pd
import numpy as np

from .entities import PitchEvent, DatasetSplit

class PitchEventRepositoryInterface(ABC):

    @abstractmethod
    def get_events(self, start_date: str, end_date: str) -> pd.DataFrame:
        pass

class DataPreprocessorInterface(ABC):
    @abstractmethod
    def preprocess(self, raw_data: pd.DataFrame) -> DatasetSplit:
        pass

#Sumedh you got these parts
class DecoderInterface(ABC):
    @abstractmethod
    def decode(self, model_output: np.ndarray) -> Dict[str, Any]:
        pass

class ModelInterface(ABC):
    @abstractmethod
    def train_and_save(self, split_data: DatasetSplit, epochs: int, batch_size: int, lr: float):
        pass # Look at my modeltrain.py line 216 to line 260 for reference

    @abstractmethod
    def predict(self, batter_id: int, pitch_type: str, features: np.ndarray) -> Dict[str, Any]:
        pass # Look at my examplemodeltext.py file for reference
