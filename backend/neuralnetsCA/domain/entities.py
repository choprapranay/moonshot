from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np


@dataclass
class Batter:
    id: int
    name: str
    encoded_id: Optional[int] = None

    def __hash__(self):
        return hash(self.id)


@dataclass
class PlateAppearance:
    batter_id: int
    pitch_type: str
    
    release_speed: float
    release_pos_x: float
    plate_x: float
    plate_z: float
    effective_speed: float
    release_spin_rate: float
    release_extension: float
    vx0: float
    vy0: float
    vz0: float
    ax: float
    ay: float
    az: float
    
    launch_speed: float = 0.0
    launch_angle: float = 0.0
    hc_x: float = 0.0
    hc_y: float = 0.0
    estimated_ba_using_speedangle: float = 0.0
    launch_speed_angle: float = 0.0
    
    description: Optional[str] = None
    events: Optional[str] = None
    outcome_text: Optional[str] = None
    batting_pattern: Optional[int] = None  
    
    batter_encoded_id: Optional[int] = None
    pitch_type_encoded_id: Optional[int] = None
    outcome_encoded_id: Optional[int] = None
    batting_pattern_encoded_id: Optional[int] = None
    launch_speed_angle_encoded_id: Optional[int] = None


@dataclass
class PredictionInput:
    batter_id: int
    batter_name: str
    pitch_type: str
    pitch_type_id: int
    features: np.ndarray  
    
    raw_plate_appearance: Optional[PlateAppearance] = None


@dataclass
class PredictionOutput:
    batter_name: str
    pitch_type: str
    predicted_outcome: str
    confidence: float
    all_probabilities: Dict[str, float] = field(default_factory=dict)
    actual_outcome: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'batter': self.batter_name,
            'pitch_type': self.pitch_type,
            'predicted_outcome': self.predicted_outcome,
            'confidence': self.confidence,
            'actual_outcome': self.actual_outcome,
            'all_probabilities': self.all_probabilities
        }


@dataclass
class ModelArtifacts:
    batter_encoder: Any  # LabelEncoder
    pitch_encoder: Any  # LabelEncoder
    outcome_encoder: Any  # LabelEncoder
    batter_pattern_encoder: Any  # LabelEncoder
    launch_speed_angle_encoder: Any  # LabelEncoder
    scaler: Any  # StandardScaler
    labels: List[str]
    num_batters: int
    num_pitches: int
    num_outcomes: int
    
    def get_outcome_label(self, encoded_id: int) -> str:
        return self.outcome_encoder.inverse_transform([encoded_id])[0]
    
    def get_batter_name(self, encoded_id: int) -> str:
        return self.batter_encoder.inverse_transform([encoded_id])[0]
    
    def get_pitch_type(self, encoded_id: int) -> str:
        return self.pitch_encoder.inverse_transform([encoded_id])[0]


@dataclass
class Model:
    state_dict: Dict[str, Any]
    artifacts: ModelArtifacts
    model_path: Optional[str] = None
    
    def to_save_dict(self) -> Dict[str, Any]:
        return {
            'model_state_dict': self.state_dict,
            'batter_encoder': self.artifacts.batter_encoder,
            'pitch_encoder': self.artifacts.pitch_encoder,
            'outcome_encoder': self.artifacts.outcome_encoder,
            'batter_pattern_encoder': self.artifacts.batter_pattern_encoder,
            'launch_speed_angle_encoder': self.artifacts.launch_speed_angle_encoder,
            'scaler': self.artifacts.scaler,
            'labels': self.artifacts.labels,
            'num_batters': self.artifacts.num_batters,
            'num_pitches': self.artifacts.num_pitches,
            'num_outcomes': self.artifacts.num_outcomes
        }
    
    @classmethod
    def from_save_dict(cls, save_dict: Dict[str, Any], model_path: Optional[str] = None) -> 'Model':
        artifacts = ModelArtifacts(
            batter_encoder=save_dict['batter_encoder'],
            pitch_encoder=save_dict['pitch_encoder'],
            outcome_encoder=save_dict['outcome_encoder'],
            batter_pattern_encoder=save_dict['batter_pattern_encoder'],
            launch_speed_angle_encoder=save_dict['launch_speed_angle_encoder'],
            scaler=save_dict['scaler'],
            labels=save_dict['labels'],
            num_batters=save_dict['num_batters'],
            num_pitches=save_dict['num_pitches'],
            num_outcomes=save_dict['num_outcomes']
        )
        return cls(
            state_dict=save_dict['model_state_dict'],
            artifacts=artifacts,
            model_path=model_path
        )


@dataclass
class TrainingConfig:
    epochs: int = 20
    learning_rate: float = 0.0001
    batch_size: int = 32
    test_split: float = 0.2
    random_state: int = 42
    save_path: str = 'batter_outcome_model.pth'
    
    def validate(self) -> bool:
        return (
            self.epochs > 0 and
            self.learning_rate > 0 and
            0 < self.test_split < 1 and
            self.batch_size > 0
        )


@dataclass
class TrainingMetrics:
    epoch: int
    train_loss: float
    train_accuracy: float
    test_accuracy: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'epoch': self.epoch,
            'train_loss': self.train_loss,
            'train_accuracy': self.train_accuracy,
            'test_accuracy': self.test_accuracy
        }


@dataclass
class DatasetInfo:
    num_samples: int
    num_batters: int
    num_pitch_types: int
    num_outcomes: int
    feature_columns: List[str]
    outcome_distribution: Dict[str, int] = field(default_factory=dict)