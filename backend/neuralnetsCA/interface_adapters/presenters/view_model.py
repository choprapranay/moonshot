from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
import pandas as pd


@dataclass
class DatasetViewModel:
    success: bool = False
    num_samples: int = 0
    num_batters: int = 0
    num_pitch_types: int = 0
    num_outcomes: int = 0
    outcome_distribution: Dict[str, int] = field(default_factory=dict)
    error_message: Optional[str] = None
    dataset: Optional[pd.DataFrame] = None  # Preprocessed data for display
    raw_data: Optional[pd.DataFrame] = None  # Raw data for training


@dataclass
class TrainingProgressViewModel:
    current_epoch: int = 0
    total_epochs: int = 0
    train_loss: float = 0.0
    train_accuracy: float = 0.0
    test_accuracy: float = 0.0


@dataclass
class TrainingResultViewModel:
    success: bool = False
    final_train_accuracy: float = 0.0
    final_test_accuracy: float = 0.0
    total_epochs: int = 0
    save_path: Optional[str] = None
    error_message: Optional[str] = None
    training_history: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class PredictionViewModel:
    batter_name: str = ""
    pitch_type: str = ""
    predicted_outcome: str = ""
    confidence: float = 0.0
    all_probabilities: Dict[str, float] = field(default_factory=dict)
    actual_outcome: Optional[str] = None


@dataclass
class InferenceResultViewModel:
    success: bool = False
    num_predictions: int = 0
    predictions: List[PredictionViewModel] = field(default_factory=list)
    error_message: Optional[str] = None


class ViewModel:
    def __init__(self):
        self.dataset_view_model = DatasetViewModel()
        self.training_progress_view_model = TrainingProgressViewModel()
        self.training_result_view_model = TrainingResultViewModel()
        self.inference_result_view_model = InferenceResultViewModel()
        
        self._on_dataset_update: Optional[Callable] = None
        self._on_training_progress_update: Optional[Callable] = None
        self._on_training_complete: Optional[Callable] = None
        self._on_inference_complete: Optional[Callable] = None
    
    def set_dataset_update_callback(self, callback: Callable) -> None:
        self._on_dataset_update = callback
    
    def set_training_progress_callback(self, callback: Callable) -> None:
        self._on_training_progress_update = callback
    
    def set_training_complete_callback(self, callback: Callable) -> None:
        self._on_training_complete = callback
    
    def set_inference_complete_callback(self, callback: Callable) -> None:
        self._on_inference_complete = callback
    
    def update_dataset(self, view_model: DatasetViewModel) -> None:
        self.dataset_view_model = view_model
        if self._on_dataset_update:
            self._on_dataset_update(view_model)
    
    def update_training_progress(self, view_model: TrainingProgressViewModel) -> None:
        self.training_progress_view_model = view_model
        if self._on_training_progress_update:
            self._on_training_progress_update(view_model)
    
    def update_training_result(self, view_model: TrainingResultViewModel) -> None:
        self.training_result_view_model = view_model
        if self._on_training_complete:
            self._on_training_complete(view_model)
    
    def update_inference_result(self, view_model: InferenceResultViewModel) -> None:
        self.inference_result_view_model = view_model
        if self._on_inference_complete:
            self._on_inference_complete(view_model)
