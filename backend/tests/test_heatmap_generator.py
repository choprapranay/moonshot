
import pytest
import pandas as pd
import torch
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from machine_learning.use_cases.heatmap_generator import HeatmapGenerator
from machine_learning.domain.ev_calculation import EVCalculator

class TestHeatmapGenerator:
    @pytest.fixture
    def mock_model(self):
        model = MagicMock()
        # Mock model to return logits for 3 classes: outcome1, outcome2, outcome3
        # Batch size 1 (since we iterate row by row in _generate_for_batter)
        # Logits that will result in some probabilities after softmax
        model.return_value = torch.tensor([[1.0, 0.0, -1.0]]) 
        return model

    @pytest.fixture
    def mock_ev_calculator(self):
        ev_calc = Mock(spec=EVCalculator)
        return ev_calc

    @pytest.fixture
    def sample_data(self):
        # Create a DataFrame with necessary columns
        data = {
            'batter_id': [101, 101, 102],
            'pitch_type_id': [0, 1, 0],
            'plate_x': [0.5, -0.5, 0.0],
            'plate_z': [2.5, 1.5, 2.0],
            # Feature columns required by HeatmapGenerator
            'release_speed': [95.0, 92.0, 94.0],
            'release_pos_x': [2.0, 2.1, 2.0],
            'launch_speed': [90.0, 88.0, 89.0],
            'launch_angle': [15.0, 12.0, 14.0],
            'effective_speed': [94.0, 91.0, 93.0],
            'release_spin_rate': [2200, 2100, 2150],
            'release_extension': [6.0, 6.1, 6.0],
            'batting_pattern_id': [1, 1, 2],
            'launch_speed_angle_id': [3, 3, 4],
            'hc_x': [100.0, 105.0, 102.0],
            'hc_y': [100.0, 95.0, 98.0],
            'vx0': [5.0, 4.0, 4.5],
            'vy0': [-130.0, -128.0, -129.0],
            'vz0': [-3.0, -4.0, -3.5],
            'ax': [-10.0, -12.0, -11.0],
            'ay': [25.0, 26.0, 25.5],
            'az': [-20.0, -22.0, -21.0],
            'estimated_ba_using_speedangle': [0.300, 0.250, 0.280]
        }
        return pd.DataFrame(data)

    def test_generate_structure_and_ev_assignment(self, mock_model, mock_ev_calculator, sample_data):
        """
        Test that generate loops through data and correctly assigns EV diff values 
        to coordinates for each batter.
        """
        outcome_labels = ['outcome1', 'outcome2', 'outcome3']
        device = 'cpu'
        
        # Setup EV calculator to return distinct values
        # We'll simulate different EV diffs for different calls to verify correct assignment
        # Call 1 (Batter 101, pitch 1): EV diff = 0.5 (Positive - "Blue")
        # Call 2 (Batter 101, pitch 2): EV diff = -0.3 (Negative - "Red")
        # Call 3 (Batter 102, pitch 1): EV diff = 0.1
        mock_ev_calculator.calculate_swing_vs_take.side_effect = [
            {'ev_swing': 0.8, 'ev_take': 0.3, 'ev_diff': 0.5},
            {'ev_swing': 0.1, 'ev_take': 0.4, 'ev_diff': -0.3},
            {'ev_swing': 0.5, 'ev_take': 0.4, 'ev_diff': 0.1},
        ]

        generator = HeatmapGenerator(
            model=mock_model,
            device=device,
            outcome_labels=outcome_labels,
            ev_calculator=mock_ev_calculator
        )

        heatmap_dict = generator.generate(sample_data)

        # Verify structure
        assert 101 in heatmap_dict
        assert 102 in heatmap_dict
        
        # Verify Batter 101 data
        batter_101_map = heatmap_dict[101]
        assert len(batter_101_map) == 2
        
        # Check first pitch for Batter 101 (0.5, 2.5) -> 0.5 EV diff
        # Note: Coordinates are floats, use approx if needed, but exact match expected here from data
        assert (0.5, 2.5) in batter_101_map
        assert batter_101_map[(0.5, 2.5)] == 0.5
        
        # Check second pitch for Batter 101 (-0.5, 1.5) -> -0.3 EV diff
        assert (-0.5, 1.5) in batter_101_map
        assert batter_101_map[(-0.5, 1.5)] == -0.3
        
        # Verify Batter 102 data
        batter_102_map = heatmap_dict[102]
        assert len(batter_102_map) == 1
        assert (0.0, 2.0) in batter_102_map
        assert batter_102_map[(0.0, 2.0)] == 0.1

        # Verify interactions
        assert mock_model.call_count == 3
        assert mock_ev_calculator.calculate_swing_vs_take.call_count == 3
        
        # Verify inputs to ev_calculator contained expected keys
        # The probs come from softmax of [1.0, 0.0, -1.0]
        # softmax([1, 0, -1]) approx [0.665, 0.245, 0.090]
        args, _ = mock_ev_calculator.calculate_swing_vs_take.call_args
        prob_dict = args[0]
        assert set(prob_dict.keys()) == set(outcome_labels)
        assert isinstance(prob_dict['outcome1'], float)

    def test_generate_empty_data(self, mock_model, mock_ev_calculator):
        """Test behavior with empty input dataframe."""
        outcome_labels = ['o1', 'o2']
        generator = HeatmapGenerator(mock_model, 'cpu', outcome_labels, mock_ev_calculator)
        
        empty_df = pd.DataFrame(columns=['batter_id'] + generator.feature_cols + ['pitch_type_id'])
        
        heatmap_dict = generator.generate(empty_df)
        
        assert heatmap_dict == {}

    def test_initialization(self, mock_model, mock_ev_calculator):
        """Test initialization of HeatmapGenerator."""
        outcome_labels = ['outcome1', 'outcome2']
        device = 'cpu'
        generator = HeatmapGenerator(mock_model, device, outcome_labels, mock_ev_calculator)
        
        assert generator.model == mock_model
        assert generator.device == device
        assert generator.outcome_labels == outcome_labels
        assert generator.ev_calculator == mock_ev_calculator
        assert len(generator.feature_cols) == 20

    def test_generate_for_batter_internal(self, mock_model, mock_ev_calculator, sample_data):
        """Test the internal _generate_for_batter method directly."""
        outcome_labels = ['outcome1', 'outcome2', 'outcome3']
        device = 'cpu'
        mock_ev_calculator.calculate_swing_vs_take.return_value = {'ev_diff': 0.5}
        
        generator = HeatmapGenerator(
            model=mock_model,
            device=device,
            outcome_labels=outcome_labels,
            ev_calculator=mock_ev_calculator
        )
        
        result = generator._generate_for_batter(sample_data, 101)
        
        assert len(result) == 2
        assert (0.5, 2.5) in result
        assert (-0.5, 1.5) in result
        assert result[(0.5, 2.5)] == 0.5

    def test_generate_print_logic(self, mock_model, mock_ev_calculator):
        """Test that the print statement is triggered correctly."""
        outcome_labels = ['o1', 'o2']
        # Ensure calculate_swing_vs_take returns a dict with 'ev_diff'
        mock_ev_calculator.calculate_swing_vs_take.return_value = {'ev_diff': 0.0}
        
        generator = HeatmapGenerator(mock_model, 'cpu', outcome_labels, mock_ev_calculator)
        
        # Create data for 100 batters to trigger the print condition
        num_batters = 100
        data = {
            'batter_id': list(range(num_batters)),
            'pitch_type_id': [0] * num_batters,
            'plate_x': [0.0] * num_batters,
            'plate_z': [2.0] * num_batters,
        }
        # Add required feature cols
        for col in generator.feature_cols:
            data[col] = [1.0] * num_batters
            
        df = pd.DataFrame(data)
        
        with patch('builtins.print') as mock_print:
            generator.generate(df)
            mock_print.assert_called_with(f"Processed 100/{num_batters} batters")
