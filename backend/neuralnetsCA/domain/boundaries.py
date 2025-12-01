from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import pandas as pd

from .entities import (
    TrainingConfig,
    TrainingMetrics,
    PredictionOutput,
    DatasetInfo,
    Model
)


@dataclass
class BuildDatasetInputData:
    start_date: str
    end_date: str
    data_source: str  


@dataclass
class TrainModelInputData:
    config: TrainingConfig
    dataset: pd.DataFrame  # Preprocessed dataset


@dataclass
class InferenceInputData:
    model_path: str
    dataset: pd.DataFrame  # Raw or preprocessed data for inference
    use_saved_encoders: bool = True


@dataclass
class BuildDatasetOutputData:
    """Output data from building a dataset."""
    success: bool
    dataset: Optional[pd.DataFrame]  # Preprocessed data for display
    raw_data: Optional[pd.DataFrame]  # Raw data for training
    dataset_info: Optional[DatasetInfo]
    error_message: Optional[str] = None


@dataclass
class TrainModelOutputData:
    success: bool
    model: Optional[Model]
    final_train_accuracy: float
    final_test_accuracy: float
    training_history: List[TrainingMetrics]
    save_path: Optional[str]
    error_message: Optional[str] = None


@dataclass
class InferenceOutputData:
    success: bool
    predictions: List[PredictionOutput]
    num_samples: int
    error_message: Optional[str] = None

class BuildDatasetInputBoundary(ABC):    
    @abstractmethod
    def execute(self, input_data: BuildDatasetInputData) -> None:
        pass


class TrainModelInputBoundary(ABC):
    
    @abstractmethod
    def execute(self, input_data: TrainModelInputData) -> None:
        pass


class InferenceInputBoundary(ABC):
    
    @abstractmethod
    def execute(self, input_data: InferenceInputData) -> None:
        pass


class BuildDatasetOutputBoundary(ABC):    
    @abstractmethod
    def present_dataset(self, output_data: BuildDatasetOutputData) -> None:
        pass


class TrainModelOutputBoundary(ABC):    
    @abstractmethod
    def present_training_progress(self, metrics: TrainingMetrics) -> None:
        pass
    
    @abstractmethod
    def present_training_complete(self, output_data: TrainModelOutputData) -> None:
        pass


class InferenceOutputBoundary(ABC):    
    @abstractmethod
    def present_inference_results(self, output_data: InferenceOutputData) -> None:
        pass