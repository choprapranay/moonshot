from dataclasses import dataclass
from typing import List, Optional
import numpy as np

@dataclass
class PitchEvent:
    batter: int
    pitch_type: str
    description: str
    events: str
    plate_x: float 
    plate_z: float 
    release_speed: float 
    release_pos_x: float 
    launch_speed: float 
    launch_angle: float 
    effective_speed: float 
    release_spin_rate: float 
    release_extension: float 
    hc_x: float 
    hc_y: float 
    vx0: float 
    vy0: float 
    vz0: float 
    ax: float 
    ay: float 
    az: float 
    sz_top: float 
    sz_bot: float 
    estimated_ba_using_speedangle: float 
    launch_speed_angle: float 

@dataclass
class DatasetSplit:
    batter_ids_train: np.ndarray
    pitch_type_ids_train: np.ndarray
    features_train: np.ndarray
    y_train: np.ndarray

    batter_ids_val: np.ndarray
    pitch_type_ids_val: np.ndarray
    features_val: np.ndarray
    y_val: np.ndarray

    num_batters: int
    num_pitch_types: int
    num_outcomes: int