import pytest
import pandas as pd
from unittest.mock import Mock, patch, mock_open
from backend.use_cases.run_value_regression import load_season_safe, main

class TestLoadSeasonSafe: 

    def test_load_season_safe_success(self):

        mock_repo = Mock()

        mock_data = []

        for i in range(6):
            df = pd.DataFrame({"game_pk":[1, 2], "events":['single', 'double'], "bat_score": [0, 1]})
            mock_data.append(df)

        mock_repo.fetch_pitch_data.side_effect = mock_data

        result = load_season_safe(mock_repo, 2023)

        assert len(result) == 12
        assert mock_repo.fetch_pitch_data.call_count == 6

    def test_load_season_safe_failure(self):

        mock_repo = Mock()
        
        df_1 = pd.DataFrame({'game_pk': [1], 'events': ['single']})
        df_2 = pd.DataFrame({'game_pk': [3], 'events': ['double']})

        mock_repo.fetch_pitch_data.side_effect = [
            df_1,
            Exception("Network error"),
            df_2,
            Exception("Timeout"),
            df_2,
            df_2
        ]

        result = load_season_safe(mock_repo, 2024)

        assert len(result) == 4
        assert mock_repo.fetch_pitch_data.call_count == 6

class TestMain:

    @patch('json.dump')                     
    @patch('builtins.print')
    @patch('builtins.open', new_callable=mock_open)
    @patch('backend.use_cases.run_value_regression.load_season_safe')
    @patch('backend.use_cases.run_value_regression.RunValueCalculator')
    @patch('backend.use_cases.run_value_regression.PyBaseballRepository') 

    def test_main_success(self, mock_repo_class, mock_calculator_class, mock_load_season, mock_file, mock_print, mock_json_dump):
        mock_repo = Mock()
        mock_repo_class.return_value = mock_repo

        df_2023 = pd.DataFrame({
            'game_pk': [1, 2],
            'inning': [1, 1],
            'inning_topbot': ['Top', 'Top'],
            'at_bat_number': [1, 2],
            'events': ['single', 'double'],
            'bat_score': [0, 1],
            'outs_when_up': [0, 1],
            'on_1b': [None, 1],
            'on_2b': [None, None],
            'on_3b': [None, None]
        })

        df_2024 = pd.DataFrame({
            'game_pk': [3, 4],
            'inning': [1, 1],
            'inning_topbot': ['Top', 'Top'],
            'at_bat_number': [1, 2],
            'events': ['home_run', 'strikeout'],
            'bat_score': [4, 4],
            'outs_when_up': [0, 1],
            'on_1b': [None, None],
            'on_2b': [None, None],
            'on_3b': [None, None]
        })

        mock_load_season.side_effect = [df_2023, df_2024]

        mock_calculator = Mock()
        mock_calculator.calculate.return_value = {
            'single': 0.456,
            'double': 0.789,
            'home_run': 1.397,
            'strikeout': -0.301
        }
        mock_calculator_class.return_value = mock_calculator
        
        main()
        
        mock_repo_class.assert_called_once()

        assert mock_load_season.call_count == 2
        mock_load_season.assert_any_call(mock_repo, 2023)
        mock_load_season.assert_any_call(mock_repo, 2024)

        mock_calculator_class.assert_called_once()
        mock_calculator.calculate.assert_called_once()

        call_args = mock_calculator.calculate.call_args[0][0]
        assert len(call_args) == 4

        mock_print.assert_called_once()

        mock_file.assert_called_once_with('run_values.json', 'w')
        mock_json_dump.assert_called_once()

        dumped_data = mock_json_dump.call_args[0][0]
        assert dumped_data == {
            'single': 0.456,
            'double': 0.789,
            'home_run': 1.397,
            'strikeout': -0.301
        }

    @patch('backend.use_cases.run_value_regression.PyBaseballRepository')
    @patch('backend.use_cases.run_value_regression.load_season_safe')
    
    def test_main_with_empty_data(self, mock_load_season, mock_repo_class):
        
        mock_repo = Mock()
        mock_repo_class.return_value = mock_repo
        
        empty_df = pd.DataFrame()
        mock_load_season.return_value = empty_df
    
        with pytest.raises(Exception):
            main()