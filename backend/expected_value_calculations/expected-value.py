import json
import os
import sys
import warnings
import pickle
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import pybaseball
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from numpy import ndarray as numpy_ndarray
from numpy import dtype as numpy_dtype
from numpy import dtypes as numpy_dtypes
from numpy.core.multiarray import _reconstruct as numpy_reconstruct
from sklearn.preprocessing import LabelEncoder, StandardScaler
from pybaseball import statcast
from backend.infrastructure.pybaseball_repository import PyBaseballRepository
from backend.infrastructure.model_storage import ModelStorage
from backend.services.dataset_loader_service import DatasetLoaderService
from neuralnets.modeltrain import SuperModel, swing_type
from backend.infrastructure.run_value_repository import RunValueRepository
from backend.expected_value_calculations.ev_calculation import EVCalculator
from backend.expected_value_calculations.heatmap_generator import HeatmapGenerator

try:
    from torch.serialization import add_safe_globals, safe_globals
except ImportError:  
    add_safe_globals = None
    safe_globals = None

# Silence noisy FutureWarnings emitted by third-party dependencies (e.g., pandas, pybaseball).
warnings.filterwarnings("ignore", category=FutureWarning)

def load_model_and_artifacts(model_path = "./neuralnets/batter_outcome_model.pth"):

    if not os.path.exists(model_path):
        print(f"Model not found locally at {model_path}")
        print("Downloading from Supabase...")
        store = ModelStorage()
        store.download_model(
            source_path="moonshot_v1/final_batter_outcome_model.pth",
            dest_path=model_path
        )
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    extra_safe_types = [
        LabelEncoder,
        numpy_reconstruct,
        numpy_ndarray,
        numpy_dtype,
        numpy_dtypes.Float64DType,
        numpy_dtypes.ObjectDType,
    ]

    load_kwargs = {
        "map_location": device,
        "weights_only": True,
    }

    def load_with_safe_globals():
        if safe_globals is not None:
            with safe_globals(extra_safe_types):
                return torch.load(model_path, **load_kwargs)
        if add_safe_globals is not None:
            add_safe_globals(extra_safe_types)
            return torch.load(model_path, **load_kwargs)
        return torch.load(model_path, map_location=device, weights_only=False)

    try:
        checkpoint = load_with_safe_globals()
    except pickle.UnpicklingError:
        warnings.warn(
            f"Falling back to torch.load(weights_only=False) for {model_path}. "
            "Ensure this checkpoint is from a trusted source.",
            RuntimeWarning,
        )
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must be a dictionary with saved artifacts")
    
    batter_enc = checkpoint['batter_encoder']
    pitch_enc = checkpoint['pitch_encoder']
    outcome_enc = checkpoint['outcome_encoder']
    batter_pattern_enc = checkpoint['batter_pattern_encoder']
    launch_speed_angle_enc = checkpoint['launch_speed_angle_encoder']
    scaler = checkpoint['scaler']
    outcome_labels = checkpoint['labels']
    NUM_BATTERS = checkpoint['num_batters']
    NUM_PITCHES = checkpoint['num_pitches']
    NUM_OUTCOMES = checkpoint['num_outcomes']

    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    model = SuperModel(NUM_BATTERS, NUM_PITCHES, 20, NUM_OUTCOMES).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    
    return {
        'model': model,
        'device': device,
        'batter_encoder': batter_enc,
        'pitch_encoder': pitch_enc,
        'outcome_encoder': outcome_enc,
        'batter_pattern_encoder': batter_pattern_enc,
        'launch_speed_angle_encoder': launch_speed_angle_enc,
        'scaler': scaler,
        'outcome_labels': outcome_labels,
        'num_batters': NUM_BATTERS,
        'num_pitches': NUM_PITCHES,
        'num_outcomes': NUM_OUTCOMES
    }

def preprocess_data(dataset, artifacts):
    """Preprocess raw pitch data for model input."""
    shortened_data = dataset[[
        'batter', 'pitch_type', 'description', 'plate_x', 'plate_z',
        'events', 'release_speed', 'release_pos_x', 'launch_speed',
        'launch_angle', 'effective_speed', 'release_spin_rate',
        'release_extension', 'hc_x', 'hc_y', 'vx0', 'vy0', 'vz0',
        'ax', 'ay', 'az', 'sz_top', 'sz_bot',
        'estimated_ba_using_speedangle', 'launch_speed_angle'
    ]]
    
    pruned_data = shortened_data.copy()
    
    both_na = pruned_data['hc_x'].isna() & pruned_data['hc_y'].isna()
    both_na_2 = pruned_data['launch_speed'].isna() & pruned_data['launch_angle'].isna()
    both_na_3 = pruned_data['launch_speed_angle'].isna() & pruned_data['estimated_ba_using_speedangle'].isna()
    
    pruned_data['outcome_text'] = pruned_data['events']
    pruned_data['outcome_text'] = pruned_data['description'].where(both_na, pruned_data['events'])
    
    pruned_data.loc[both_na, 'hc_x'] = 0
    pruned_data.loc[both_na, 'hc_y'] = 0
    pruned_data.loc[both_na_2, 'launch_speed'] = 0
    pruned_data.loc[both_na_2, 'launch_angle'] = 0
    pruned_data.loc[both_na_3, 'launch_speed_angle'] = 0
    pruned_data.loc[both_na_3, 'estimated_ba_using_speedangle'] = 0
    
    pruned_data = pruned_data.dropna(subset=['outcome_text'])
    pruned_data['batting_pattern'] = pruned_data['outcome_text'].apply(swing_type)
    pruned_data = pruned_data.dropna(subset=['batting_pattern'])
    
    feature_cols = [
        'release_speed', 'release_pos_x', 'plate_x', 'plate_z', 'launch_speed',
        'launch_angle', 'effective_speed', 'release_spin_rate', 'release_extension',
        'hc_x', 'hc_y', 'vx0', 'vy0', 'vz0', 'ax', 'ay', 'az', 'estimated_ba_using_speedangle'
    ]
    
    pruned_data = pruned_data.dropna(subset=feature_cols)
    
    batter_enc = artifacts['batter_encoder']
    pitch_enc = artifacts['pitch_encoder']
    outcome_enc = artifacts['outcome_encoder']
    batter_pattern_enc = artifacts['batter_pattern_encoder']
    launch_speed_angle_enc = artifacts['launch_speed_angle_encoder']
    scaler = artifacts['scaler']
    
    known_batters = set(batter_enc.classes_)
    pruned_data = pruned_data[pruned_data['batter'].isin(known_batters)]
    print(f"Rows after batter filtering: {len(pruned_data)}")
    if len(pruned_data) == 0:
        raise ValueError("No known batters found in dataset")
    pruned_data['batter_id'] = batter_enc.transform(pruned_data['batter'])
    
    known_outcomes = set(outcome_enc.classes_)
    pruned_data = pruned_data[pruned_data['outcome_text'].isin(known_outcomes)]
    print(f"Rows after outcome filtering: {len(pruned_data)}")
    if len(pruned_data) == 0:
        raise ValueError("No known outcomes found in dataset")
    pruned_data['outcome_id'] = outcome_enc.transform(pruned_data['outcome_text'])
    
    known_pitches = set(pitch_enc.classes_)
    pruned_data = pruned_data[pruned_data['pitch_type'].isin(known_pitches)]
    print(f"Rows after pitch type filtering: {len(pruned_data)}")
    if len(pruned_data) == 0:
        raise ValueError("No known pitch types found in dataset")
    pruned_data['pitch_type_id'] = pitch_enc.transform(pruned_data['pitch_type'])
    
    known_patterns = set(batter_pattern_enc.classes_)
    pruned_data = pruned_data[pruned_data['batting_pattern'].isin(known_patterns)]
    
    pruned_data['batting_pattern_id'] = batter_pattern_enc.transform(pruned_data['batting_pattern'])
    pruned_data['launch_speed_angle_id'] = launch_speed_angle_enc.transform(pruned_data['launch_speed_angle'])
    
    pruned_data[feature_cols] = scaler.transform(pruned_data[feature_cols].astype(float))
    
    pruned_data = pruned_data.replace([np.inf, -np.inf], np.nan)
    pruned_data = pruned_data.dropna(subset=feature_cols)
    
    return pruned_data

def main():

    repo = PyBaseballRepository()
    dataset = repo.fetch_pitch_data('2023-03-30', '2024-11-02')
    artifacts = load_model_and_artifacts()

    processed_data = preprocess_data(dataset, artifacts)

    run_value_repo = RunValueRepository()
    run_values = run_value_repo.load_run_values()

    ev_calculator = EVCalculator(run_values)
    heatmap_gen = HeatmapGenerator(model=artifacts['model'], device=artifacts['device'], outcome_labels=artifacts['outcome_labels'], ev_calculator=ev_calculator)
    
    heatmap_dict = heatmap_gen.generate(processed_data)

    serialized = {
        int(batter_id): [
            {
                "plate_x": coord[0],
                "plate_z": coord[1],
                "expected_value_diff": ev_diff,
            }
            for coord, ev_diff in batter_data.items()
        ]
        for batter_id, batter_data in heatmap_dict.items()
    }

    output_path = Path(__file__).resolve().parent / "heatmap_output.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(
            serialized,
            f,
            indent=2,
            default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else o,
        )

if __name__ == "__main__":
    main()

