import torch
import pandas as pd
import numpy as np
from typing import Dict, Tuple
from ev_calculation import EVCalculator

class HeatmapGenerator:
   
    
    def __init__(self, model, device: str, outcome_labels: list, ev_calculator: EVCalculator):
        self.model = model
        self.device = device
        self.outcome_labels = outcome_labels
        self.ev_calculator = ev_calculator
        
        self.feature_cols = [
            'release_speed', 'release_pos_x', 'plate_x', 'plate_z',
            'launch_speed', 'launch_angle', 'effective_speed',
            'release_spin_rate', 'release_extension', 'batting_pattern_id',
            'launch_speed_angle_id', 'hc_x', 'hc_y', 'vx0', 'vy0', 'vz0',
            'ax', 'ay', 'az', 'estimated_ba_using_speedangle'
        ]
    

    def generate(self, processed_data: pd.DataFrame) -> Dict[int, Dict[Tuple[float, float], float]]:
        heatmap_dict = {}
        unique_batters = processed_data['batter_id'].unique()
        
        
        for idx, batter_id in enumerate(unique_batters):
            if (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1}/{len(unique_batters)} batters")
            
            heatmap_dict[batter_id] = self._generate_for_batter(
                processed_data,
                batter_id
            )
        
        return heatmap_dict
    
    def _generate_for_batter(self, data: pd.DataFrame, batter_id: int) -> Dict[Tuple[float, float], float]:
   
        batter_pitches = data[data['batter_id'] == batter_id]
        batter_heatmap = {}
        
        for index, row in batter_pitches.iterrows():
          
            features = torch.tensor(
                row[self.feature_cols].values.astype(np.float32),
                dtype=torch.float32
            ).unsqueeze(0).to(self.device)
            
            batter_tensor = torch.tensor([batter_id], dtype=torch.long).to(self.device)
            pitch_type_tensor = torch.tensor(
                [int(row['pitch_type_id'])],
                dtype=torch.long
            ).to(self.device)
            
            with torch.no_grad():
                logits = self.model(batter_tensor, pitch_type_tensor, features)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    
            prob_dict = {
                self.outcome_labels[i]: float(probs[i])
                for i in range(len(self.outcome_labels))
            }
           
            ev_result = self.ev_calculator.calculate_swing_vs_take(prob_dict)
            
            coord = (float(row['plate_x']), float(row['plate_z']))
            batter_heatmap[coord] = ev_result['ev_diff']
        
        return batter_heatmap
