import pybaseball
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from neuralnet.modeltrain import SuperModel, swing_type

RUN_VALUES = {
    'home_run': 1.40,
    'triple': 1.02,
    'double': 0.75,
    'single': 0.46,
    'field_out': -0.27,
    'grounded_into_double_play': -0.82,
    'strikeout': -0.30,
    'walk': 0.31,
    'hit_by_pitch': 0.34,
    'ball': 0.03,
    'called_strike': -0.05,
}

SWING_OUTCOMES = {'home_run', 'triple', 'double', 'single', 'field_out', 
                  'grounded_into_double_play', 'strikeout', 'swinging_strike',
                  'foul', 'force_out', 'sac_fly', 'field_error'}

TAKE_OUTCOMES = {'ball', 'walk', 'hit_by_pitch', 'called_strike'}

def calculate_ev(probability_dict, outcome_set): 
    total_prob = sum(probability_dict.get(outcome, 0) for outcome in outcome_set)
    if total_prob == 0:
        return 0.0
    
    ev = 0
    for outcome in outcome_set:
        if outcome in probability_dict and outcome in RUN_VALUES:
            normalized_prob = probability_dict[outcome] / total_prob
            ev += normalized_prob * RUN_VALUES[outcome]
    return ev


def generate_heatmap_data():

    batter_info = pybaseball.statcast('2025-10-01', '2025-10-15')
    
    shortened_data = batter_info[['batter', 'pitch_type', 'description', 'plate_x', 'plate_z', 
                                   'events', 'release_speed', 'release_pos_x', 'launch_speed', 
                                   'launch_angle', 'effective_speed', 'release_spin_rate', 
                                   'release_extension', 'hc_x', 'hc_y', 'vx0', 'vy0', 'vz0', 
                                   'ax', 'ay', 'az', 'sz_top', 'sz_bot', 
                                   'estimated_ba_using_speedangle', 'launch_speed_angle']]
    
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
    
    feature_cols = ['release_speed', 'release_pos_x', 'plate_x', 'plate_z', 'launch_speed', 
                    'launch_angle', 'effective_speed', 'release_spin_rate', 'release_extension', 
                    'hc_x', 'hc_y', 'vx0', 'vy0', 'vz0', 'ax', 'ay', 'az', 'estimated_ba_using_speedangle']
    
    pruned_data = pruned_data.dropna(subset=feature_cols)

    batter_enc = LabelEncoder() 
    pruned_data['batter_id'] = batter_enc.fit_transform(pruned_data['batter'])
    pitch_enc = LabelEncoder()
    pruned_data['pitch_type_id'] = pitch_enc.fit_transform(pruned_data['pitch_type'])
    outcome_enc = LabelEncoder()
    pruned_data['outcome_id'] = outcome_enc.fit_transform(pruned_data['outcome_text'])
    batter_pattern_enc = LabelEncoder()
    pruned_data['batting_pattern_id'] = batter_pattern_enc.fit_transform(pruned_data['batting_pattern'])
    launch_speed_angle_enc = LabelEncoder()
    pruned_data['launch_speed_angle_id'] = launch_speed_angle_enc.fit_transform(pruned_data['launch_speed_angle'])

    scaler = StandardScaler()
    pruned_data[feature_cols] = scaler.fit_transform(pruned_data[feature_cols].astype(float))

    pruned_data = pruned_data.replace([np.inf, -np.inf], np.nan)
    pruned_data = pruned_data.dropna(subset=feature_cols)
    
    outcome_labels = sorted(pruned_data['outcome_text'].unique())

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_BATTERS = pruned_data['batter_id'].nunique()
    NUM_PITCHES = pruned_data['pitch_type_id'].nunique()
    NUM_OUTCOMES = len(outcome_labels)

    model = SuperModel(NUM_BATTERS, NUM_PITCHES, 20, NUM_OUTCOMES).to(device)
    model.load_state_dict(torch.load('batter_outcome_model.pth', map_location=device))
    model.eval()

    feature_cols_model = ['release_speed', 'release_pos_x', 'plate_x', 'plate_z', 
                        'launch_speed', 'launch_angle', 'effective_speed', 
                        'release_spin_rate', 'release_extension', 'batting_pattern_id', 
                        'launch_speed_angle_id', 'hc_x', 'hc_y', 'vx0', 'vy0', 'vz0', 
                        'ax', 'ay', 'az', 'estimated_ba_using_speedangle']

    heatmap_dict = {}   
    unique_batters = pruned_data['batter_id'].unique()

    for idx, batter_id in enumerate(unique_batters):
        heatmap_dict[batter_id] = {}
        batter_pitches = pruned_data[pruned_data['batter_id'] == batter_id]

        for index, row in batter_pitches.iterrows():
            features = torch.tensor(
                row[feature_cols_model].values.astype(np.float32),
                dtype=torch.float32
            ).unsqueeze(0).to(device)

            batter_tensor = torch.tensor([batter_id], dtype=torch.long).to(device)
            pitch_type_tensor = torch.tensor([int(row['pitch_type_id'])], dtype=torch.long).to(device)
            
            with torch.no_grad():
                logits = model(batter_tensor, pitch_type_tensor, features)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

            prob_dict = {outcome_labels[i]: probs[i] for i in range(len(outcome_labels))}

            ev_swing = calculate_ev(prob_dict, SWING_OUTCOMES)
            ev_take = calculate_ev(prob_dict, TAKE_OUTCOMES)
            ev_diff = ev_swing - ev_take

            coord = (float(row['plate_x']), float(row['plate_z']))
            heatmap_dict[batter_id][coord] = ev_diff
    
    return heatmap_dict


if __name__ == "__main__":
    heatmap_dict = generate_heatmap_data()

