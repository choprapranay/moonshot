import json
import sys
import warnings
from pathlib import Path
from typing import Dict, Tuple

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "neuralnetsCA"))

import torch
import numpy as np

from neuralnetsCA.infrastructure.data.pybaseball_pull import PybaseballDataAccess
from neuralnetsCA.infrastructure.data.preprocess_adapter import PreprocessorAdapter
from neuralnetsCA.infrastructure.data.model_adapter import ModelAdapter
from neuralnetsCA.infrastructure.storage.storage_adapter import ModelStorageAdapter
from backend.infrastructure.run_value_repository import RunValueRepository
from backend.domain.ev_calculation import EVCalculator

warnings.filterwarnings("ignore", category=FutureWarning)


class HeatmapGeneratorCA:
    FEATURE_COLS = [
        'release_speed', 'release_pos_x', 'plate_x', 'plate_z',
        'launch_speed', 'launch_angle', 'effective_speed',
        'release_spin_rate', 'release_extension', 'batting_pattern_id',
        'launch_speed_angle_id', 'hc_x', 'hc_y', 'vx0', 'vy0', 'vz0',
        'ax', 'ay', 'az', 'estimated_ba_using_speedangle'
    ]

    def __init__(self, model, device: str, outcome_labels: list, ev_calculator: EVCalculator):
        self.model = model
        self.device = device
        self.outcome_labels = outcome_labels
        self.ev_calculator = ev_calculator

    def generate(self, processed_data) -> Dict[int, Dict[Tuple[float, float], float]]:
        heatmap_dict = {}
        unique_batters = processed_data['batter_id'].unique()

        for idx, batter_id in enumerate(unique_batters):
            if (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1}/{len(unique_batters)} batters")
            heatmap_dict[batter_id] = self._generate_for_batter(processed_data, batter_id)

        return heatmap_dict

    def _generate_for_batter(self, data, batter_id: int) -> Dict[Tuple[float, float], float]:
        batter_pitches = data[data['batter_id'] == batter_id]
        batter_heatmap = {}

        for _, row in batter_pitches.iterrows():
            features = torch.tensor(
                row[self.FEATURE_COLS].values.astype(np.float32),
                dtype=torch.float32
            ).unsqueeze(0).to(self.device)

            batter_tensor = torch.tensor([batter_id], dtype=torch.long).to(self.device)
            pitch_type_tensor = torch.tensor([int(row['pitch_type_id'])], dtype=torch.long).to(self.device)

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


def main():
    model_path = str(project_root / "final_batter_outcome_model.pth")

    data_access = PybaseballDataAccess()
    preprocessor = PreprocessorAdapter()
    storage = ModelStorageAdapter()
    model_adapter = ModelAdapter()

    print("Fetching pitch data...")
    raw_data = data_access.fetch_data('2023-03-30', '2024-11-02')
    print(f"Fetched {len(raw_data)} rows")

    print("Loading model...")
    model_entity = storage.load_model(model_path)
    if model_entity is None:
        raise ValueError(f"Failed to load model from {model_path}")

    artifacts = model_entity.artifacts
    print(f"Model loaded: {artifacts.num_batters} batters, {artifacts.num_pitches} pitch types, {artifacts.num_outcomes} outcomes")

    print("Preprocessing data for inference...")
    processed_data = preprocessor.preprocess_for_inference(raw_data, artifacts)
    print(f"Preprocessed {len(processed_data)} rows")

    model = model_adapter.create_model(
        num_batters=artifacts.num_batters,
        num_pitch_types=artifacts.num_pitches,
        num_outcomes=artifacts.num_outcomes
    )
    model = model_adapter.load_model_state(model, model_entity.state_dict)
    model.eval()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    run_value_repo = RunValueRepository()
    run_values = run_value_repo.load_run_values()
    ev_calculator = EVCalculator(run_values)

    heatmap_gen = HeatmapGeneratorCA(
        model=model,
        device=device,
        outcome_labels=artifacts.labels,
        ev_calculator=ev_calculator
    )

    print("Generating heatmaps...")
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

    print(f"Heatmap saved to {output_path}")


if __name__ == "__main__":
    main()

