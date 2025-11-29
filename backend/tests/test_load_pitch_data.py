"""
Test suite for load_pitch_data ETL pipeline use case interactor.
"""
import sys
import os
from unittest.mock import MagicMock

sys.modules['supabase'] = MagicMock()
mock_create_client = MagicMock()
sys.modules['supabase'].create_client = mock_create_client
sys.modules['dotenv'] = MagicMock()
sys.modules['dotenv'].load_dotenv = MagicMock()

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(1, parent_dir)

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, call

from backend.app.load_pitch_data import load_pitch_data, batch_insert, BATCH_SIZE, RETRY_LIMIT


@pytest.fixture
def sample_raw_data():
    """Fixture providing sample raw pitch data from Statcast."""
    return pd.DataFrame({
        'game_date': ['2023-04-01', '2023-04-01', '2023-04-01'],
        'batter': [12345, 12346, 12345],
        'pitch_type': ['FF', 'SL', 'FF'],
        'description': ['swinging_strike', 'ball', 'single'],
        'plate_x': [-0.5, 0.3, 0.0],
        'plate_z': [2.0, 3.0, 2.5],
        'events': ['strikeout', None, 'single'],
        'release_speed': [95.0, 88.0, 92.0],
        'release_pos_x': [0.5, 0.4, 0.6],
        'launch_speed': [None, None, 100.0],
        'launch_angle': [None, None, 15.0],
        'effective_speed': [94.0, 87.0, 91.0],
        'release_spin_rate': [2200.0, 1800.0, 2100.0],
        'release_extension': [6.0, 5.5, 6.2],
        'hc_x': [None, None, 100.0],
        'hc_y': [None, None, 50.0],
        'vx0': [-10.0, -8.0, -9.5],
        'vy0': [120.0, 110.0, 118.0],
        'vz0': [-5.0, -4.0, -4.8],
        'ax': [0.0, 0.0, 0.0],
        'ay': [-30.0, -28.0, -29.0],
        'az': [0.0, 0.0, 0.0],
        'sz_top': [3.5, 3.5, 3.5],
        'sz_bot': [1.5, 1.5, 1.5],
        'estimated_ba_using_speedangle': [None, None, 0.3],
        'launch_speed_angle': [None, None, 'barrel'],
    })


@pytest.fixture
def sample_cleaned_data():
    """Fixture providing sample cleaned data."""
    return pd.DataFrame({
        'game_date': ['2023-04-01', '2023-04-01', '2023-04-01'],
        'batter': [12345, 12346, 12345],
        'pitch_type': ['FF', 'SL', 'FF'],
        'description': ['swinging_strike', 'ball', 'single'],
        'plate_x': [-0.5, 0.3, 0.0],
        'plate_z': [2.0, 3.0, 2.5],
        'events': ['strikeout', 'ball', 'single'],
        'release_speed': [95.0, 88.0, 92.0],
        'release_pos_x': [0.5, 0.4, 0.6],
        'launch_speed': [0.0, 0.0, 100.0],
        'launch_angle': [0.0, 0.0, 15.0],
        'effective_speed': [94.0, 87.0, 91.0],
        'release_spin_rate': [2200.0, 1800.0, 2100.0],
        'release_extension': [6.0, 5.5, 6.2],
        'hc_x': [0.0, 0.0, 100.0],
        'hc_y': [0.0, 0.0, 50.0],
        'vx0': [-10.0, -8.0, -9.5],
        'vy0': [120.0, 110.0, 118.0],
        'vz0': [-5.0, -4.0, -4.8],
        'ax': [0.0, 0.0, 0.0],
        'ay': [-30.0, -28.0, -29.0],
        'az': [0.0, 0.0, 0.0],
        'sz_top': [3.5, 3.5, 3.5],
        'sz_bot': [1.5, 1.5, 1.5],
        'estimated_ba_using_speedangle': [0.0, 0.0, 0.3],
        'launch_speed_angle': [0.0, 0.0, 'barrel'],
        'outcome_text': ['swinging_strike', 'ball', 'single'],
        'swing_take': [1, 0, 1],
    })


@pytest.fixture
def mock_supabase_repo():
    """Fixture providing a mock SupabaseRepository."""
    repo = Mock()
    repo.insert_pitch_data_rows = Mock(return_value=None)
    return repo


class TestBatchInsert:
    """Test the batch_insert function."""
    
    def test_batch_insert_single_batch(self, mock_supabase_repo):
        """Test batch insert with data smaller than batch size."""
        rows = [{'id': i} for i in range(100)]
        batch_insert(mock_supabase_repo, "pitch_data_raw", rows)
        
        mock_supabase_repo.insert_pitch_data_rows.assert_called_once()
        call_args = mock_supabase_repo.insert_pitch_data_rows.call_args
        assert call_args[0][0] == "pitch_data_raw"
        assert len(call_args[0][1]) == 100
    
    def test_batch_insert_multiple_batches(self, mock_supabase_repo):
        """Test batch insert with data larger than batch size."""
        rows = [{'id': i} for i in range(BATCH_SIZE * 2 + 50)]
        batch_insert(mock_supabase_repo, "pitch_data_raw", rows)
        
        assert mock_supabase_repo.insert_pitch_data_rows.call_count == 3
        
        calls = mock_supabase_repo.insert_pitch_data_rows.call_args_list
        assert len(calls[0][0][1]) == BATCH_SIZE
        assert len(calls[1][0][1]) == BATCH_SIZE
        assert len(calls[2][0][1]) == 50
    
    def test_batch_insert_exact_batch_size(self, mock_supabase_repo):
        """Test batch insert with exactly one batch size."""
        rows = [{'id': i} for i in range(BATCH_SIZE)]
        batch_insert(mock_supabase_repo, "pitch_data_raw", rows)
        
        mock_supabase_repo.insert_pitch_data_rows.assert_called_once()
        call_args = mock_supabase_repo.insert_pitch_data_rows.call_args
        assert len(call_args[0][1]) == BATCH_SIZE
    
    def test_batch_insert_retry_on_failure(self, mock_supabase_repo):
        """Test that batch insert retries on failure."""
        mock_supabase_repo.insert_pitch_data_rows.side_effect = [
            Exception("Connection error"),
            Exception("Timeout"),
            None
        ]
        
        rows = [{'id': 1}]
        batch_insert(mock_supabase_repo, "pitch_data_raw", rows)
        
        assert mock_supabase_repo.insert_pitch_data_rows.call_count == 3
    
    def test_batch_insert_max_retries_exceeded(self, mock_supabase_repo, capsys):
        """Test that batch insert gives up after max retries."""
        mock_supabase_repo.insert_pitch_data_rows.side_effect = Exception("Persistent error")
        
        rows = [{'id': 1}]
        batch_insert(mock_supabase_repo, "pitch_data_raw", rows)
        
        assert mock_supabase_repo.insert_pitch_data_rows.call_count == RETRY_LIMIT
        
        captured = capsys.readouterr()
        assert "Failed to insert batch after" in captured.out
    
    def test_batch_insert_empty_rows(self, mock_supabase_repo):
        """Test batch insert with empty rows."""
        rows = []
        batch_insert(mock_supabase_repo, "pitch_data_raw", rows)
        
        mock_supabase_repo.insert_pitch_data_rows.assert_not_called()


class TestLoadPitchData:
    """Test the main load_pitch_data function."""
    
    @patch('backend.infrastructure.supabase_client.create_client')
    @patch('backend.app.load_pitch_data.SupabaseRepository')
    @patch('backend.app.load_pitch_data.CleaningService')
    @patch('backend.app.load_pitch_data.PitchDataService')
    def test_load_pitch_data_success(self, mock_pitch_service, mock_cleaning_service, 
                                     mock_repo_class, mock_supabase_client, 
                                     sample_raw_data, sample_cleaned_data, capsys):
        """Test successful ETL pipeline execution."""
        mock_pitch_service.get_pitch_data.return_value = sample_raw_data
        mock_cleaning_service.fix_na.return_value = sample_cleaned_data
        mock_repo = Mock()
        mock_repo.insert_pitch_data_rows = Mock(return_value=None)
        mock_repo_class.return_value = mock_repo
        
        # Execute
        load_pitch_data("2023-04-01", "2023-04-01")
        
        # Verify extraction
        mock_pitch_service.get_pitch_data.assert_called_once_with("2023-04-01", "2023-04-01")
        
        # Verify transformation
        mock_cleaning_service.fix_na.assert_called_once()
        
        # Verify loading
        assert mock_repo.insert_pitch_data_rows.called
        
        # Verify output messages
        captured = capsys.readouterr()
        assert "Extracting" in captured.out
        assert "Uploading" in captured.out
        assert "COMPLETED" in captured.out
    
    @patch('backend.infrastructure.supabase_client.create_client')
    @patch('backend.app.load_pitch_data.SupabaseRepository')
    @patch('backend.app.load_pitch_data.CleaningService')
    @patch('backend.app.load_pitch_data.PitchDataService')
    def test_load_pitch_data_empty_result(self, mock_pitch_service, mock_cleaning_service,
                                          mock_repo_class, mock_supabase_client, capsys):
        """Test ETL pipeline with empty data result."""
        mock_pitch_service.get_pitch_data.return_value = pd.DataFrame()
        mock_cleaning_service.fix_na.return_value = pd.DataFrame()
        mock_repo_class.return_value = Mock()
        
        # Execute
        load_pitch_data("2023-04-01", "2023-04-01")
        
        # Verify no insertion was attempted
        mock_repo = mock_repo_class.return_value
        mock_repo.insert_pitch_data_rows.assert_not_called()
        
        # Verify appropriate message
        captured = capsys.readouterr()
        assert "No valid rows found" in captured.out
    
    @patch('backend.infrastructure.supabase_client.create_client')
    @patch('backend.app.load_pitch_data.SupabaseRepository')
    @patch('backend.app.load_pitch_data.CleaningService')
    @patch('backend.app.load_pitch_data.PitchDataService')
    def test_load_pitch_data_date_range(self, mock_pitch_service, mock_cleaning_service,
                                       mock_repo_class, mock_supabase_client,
                                       sample_raw_data, sample_cleaned_data):
        """Test ETL pipeline with date range."""
        mock_pitch_service.get_pitch_data.return_value = sample_raw_data
        mock_cleaning_service.fix_na.return_value = sample_cleaned_data
        mock_repo = Mock()
        mock_repo.insert_pitch_data_rows = Mock(return_value=None)
        mock_repo_class.return_value = mock_repo
        
        start_date = "2023-04-01"
        end_date = "2023-04-05"
        load_pitch_data(start_date, end_date)
        
        # Verify extraction was called with date range
        mock_pitch_service.get_pitch_data.assert_called_once_with(start_date, end_date)
    
    @patch('backend.infrastructure.supabase_client.create_client')
    @patch('backend.app.load_pitch_data.SupabaseRepository')
    @patch('backend.app.load_pitch_data.CleaningService')
    @patch('backend.app.load_pitch_data.PitchDataService')
    def test_load_pitch_data_large_dataset_batching(self, mock_pitch_service, mock_cleaning_service,
                                                    mock_repo_class, mock_supabase_client):
        """Test that large datasets are properly batched."""
        large_data = pd.DataFrame({
            'game_date': ['2023-04-01'] * (BATCH_SIZE * 3 + 100),
            'batter': range(BATCH_SIZE * 3 + 100),
            'pitch_type': ['FF'] * (BATCH_SIZE * 3 + 100),
            'description': ['ball'] * (BATCH_SIZE * 3 + 100),
            'plate_x': [0.0] * (BATCH_SIZE * 3 + 100),
            'plate_z': [2.5] * (BATCH_SIZE * 3 + 100),
            'events': ['ball'] * (BATCH_SIZE * 3 + 100),
            'release_speed': [95.0] * (BATCH_SIZE * 3 + 100),
            'release_pos_x': [0.5] * (BATCH_SIZE * 3 + 100),
            'launch_speed': [0.0] * (BATCH_SIZE * 3 + 100),
            'launch_angle': [0.0] * (BATCH_SIZE * 3 + 100),
            'effective_speed': [94.0] * (BATCH_SIZE * 3 + 100),
            'release_spin_rate': [2200.0] * (BATCH_SIZE * 3 + 100),
            'release_extension': [6.0] * (BATCH_SIZE * 3 + 100),
            'hc_x': [0.0] * (BATCH_SIZE * 3 + 100),
            'hc_y': [0.0] * (BATCH_SIZE * 3 + 100),
            'vx0': [-10.0] * (BATCH_SIZE * 3 + 100),
            'vy0': [120.0] * (BATCH_SIZE * 3 + 100),
            'vz0': [-5.0] * (BATCH_SIZE * 3 + 100),
            'ax': [0.0] * (BATCH_SIZE * 3 + 100),
            'ay': [-30.0] * (BATCH_SIZE * 3 + 100),
            'az': [0.0] * (BATCH_SIZE * 3 + 100),
            'sz_top': [3.5] * (BATCH_SIZE * 3 + 100),
            'sz_bot': [1.5] * (BATCH_SIZE * 3 + 100),
            'estimated_ba_using_speedangle': [0.0] * (BATCH_SIZE * 3 + 100),
            'launch_speed_angle': [0.0] * (BATCH_SIZE * 3 + 100),
            'outcome_text': ['ball'] * (BATCH_SIZE * 3 + 100),
            'swing_take': [0] * (BATCH_SIZE * 3 + 100),
        })
        
        mock_pitch_service.get_pitch_data.return_value = large_data
        mock_cleaning_service.fix_na.return_value = large_data
        mock_repo = Mock()
        mock_repo.insert_pitch_data_rows = Mock(return_value=None)
        mock_repo_class.return_value = mock_repo
        
        load_pitch_data("2023-04-01", "2023-04-01")
        
        assert mock_repo.insert_pitch_data_rows.call_count == 4
    
    @patch('backend.infrastructure.supabase_client.create_client')
    @patch('backend.app.load_pitch_data.SupabaseRepository')
    @patch('backend.app.load_pitch_data.CleaningService')
    @patch('backend.app.load_pitch_data.PitchDataService')
    def test_load_pitch_data_cleaning_applied(self, mock_pitch_service, mock_cleaning_service,
                                              mock_repo_class, mock_supabase_client,
                                              sample_raw_data, sample_cleaned_data):
        """Test that cleaning service is properly applied."""
        mock_pitch_service.get_pitch_data.return_value = sample_raw_data
        mock_cleaning_service.fix_na.return_value = sample_cleaned_data
        mock_repo = Mock()
        mock_repo.insert_pitch_data_rows = Mock(return_value=None)
        mock_repo_class.return_value = mock_repo
        
        load_pitch_data("2023-04-01", "2023-04-01")
        
        # Verify cleaning was called with raw data
        mock_cleaning_service.fix_na.assert_called_once()
        call_args = mock_cleaning_service.fix_na.call_args[0][0]
        pd.testing.assert_frame_equal(call_args, sample_raw_data)
    
    @patch('backend.infrastructure.supabase_client.create_client')
    @patch('backend.app.load_pitch_data.SupabaseRepository')
    @patch('backend.app.load_pitch_data.CleaningService')
    @patch('backend.app.load_pitch_data.PitchDataService')
    def test_load_pitch_data_repository_initialization(self, mock_pitch_service, mock_cleaning_service,
                                                        mock_repo_class, mock_supabase_client,
                                                        sample_raw_data, sample_cleaned_data):
        """Test that SupabaseRepository is properly initialized."""
        mock_pitch_service.get_pitch_data.return_value = sample_raw_data
        mock_cleaning_service.fix_na.return_value = sample_cleaned_data
        mock_repo = Mock()
        mock_repo.insert_pitch_data_rows = Mock(return_value=None)
        mock_repo_class.return_value = mock_repo
        
        load_pitch_data("2023-04-01", "2023-04-01")
        
        # Verify repository was instantiated
        mock_repo_class.assert_called_once()
    
    @patch('backend.infrastructure.supabase_client.create_client')
    @patch('backend.app.load_pitch_data.SupabaseRepository')
    @patch('backend.app.load_pitch_data.CleaningService')
    @patch('backend.app.load_pitch_data.PitchDataService')
    def test_load_pitch_data_conversion_to_records(self, mock_pitch_service, mock_cleaning_service,
                                                   mock_repo_class, mock_supabase_client,
                                                   sample_cleaned_data):
        """Test that DataFrame is converted to records format."""
        mock_pitch_service.get_pitch_data.return_value = sample_cleaned_data
        mock_cleaning_service.fix_na.return_value = sample_cleaned_data
        mock_repo = Mock()
        mock_repo.insert_pitch_data_rows = Mock(return_value=None)
        mock_repo_class.return_value = mock_repo
        
        load_pitch_data("2023-04-01", "2023-04-01")
        
        # Verify insert was called with records
        assert mock_repo.insert_pitch_data_rows.called
        call_args = mock_repo.insert_pitch_data_rows.call_args
        assert call_args[0][0] == "pitch_data_raw"
        assert isinstance(call_args[0][1], list)
        assert len(call_args[0][1]) == len(sample_cleaned_data)


class TestETLPipelineIntegration:
    """Integration tests for the complete ETL pipeline."""
    
    @patch('backend.infrastructure.supabase_client.create_client')
    @patch('backend.app.load_pitch_data.SupabaseRepository')
    @patch('backend.app.load_pitch_data.CleaningService')
    @patch('backend.app.load_pitch_data.PitchDataService')
    def test_full_etl_pipeline_flow(self, mock_pitch_service, mock_cleaning_service,
                                    mock_repo_class, mock_supabase_client,
                                    sample_raw_data, sample_cleaned_data):
        """Test the complete ETL pipeline flow."""
        mock_pitch_service.get_pitch_data.return_value = sample_raw_data
        mock_cleaning_service.fix_na.return_value = sample_cleaned_data
        mock_repo = Mock()
        mock_repo.insert_pitch_data_rows = Mock(return_value=None)
        mock_repo_class.return_value = mock_repo
        
        # Execute
        load_pitch_data("2023-04-01", "2023-04-01")
        
        # Verify ETL steps in order
        assert mock_pitch_service.get_pitch_data.called
        assert mock_cleaning_service.fix_na.called
        assert mock_repo.insert_pitch_data_rows.called
        
        # Verify data flow
        cleaning_call_arg = mock_cleaning_service.fix_na.call_args[0][0]
        pd.testing.assert_frame_equal(cleaning_call_arg, sample_raw_data)
    
    @patch('backend.infrastructure.supabase_client.create_client')
    @patch('backend.app.load_pitch_data.SupabaseRepository')
    @patch('backend.app.load_pitch_data.CleaningService')
    @patch('backend.app.load_pitch_data.PitchDataService')
    def test_etl_pipeline_error_handling(self, mock_pitch_service, mock_cleaning_service,
                                         mock_repo_class, mock_supabase_client,
                                         sample_raw_data, sample_cleaned_data):
        """Test ETL pipeline error handling."""
        mock_pitch_service.get_pitch_data.side_effect = Exception("API Error")
        mock_cleaning_service.fix_na.return_value = sample_cleaned_data
        mock_repo = Mock()
        mock_repo_class.return_value = mock_repo
        
        # Execute and verify exception is raised
        with pytest.raises(Exception):
            load_pitch_data("2023-04-01", "2023-04-01")
        
        # Verify later steps were not called
        mock_cleaning_service.fix_na.assert_not_called()
        mock_repo.insert_pitch_data_rows.assert_not_called()


class TestConstants:
    """Test that constants are properly defined."""
    
    def test_batch_size_constant(self):
        """Test that BATCH_SIZE constant is defined."""
        from backend.app.load_pitch_data import BATCH_SIZE
        assert BATCH_SIZE == 500
        assert isinstance(BATCH_SIZE, int)
        assert BATCH_SIZE > 0
    
    def test_retry_limit_constant(self):
        """Test that RETRY_LIMIT constant is properly defined."""
        from backend.app.load_pitch_data import RETRY_LIMIT
        assert RETRY_LIMIT == 3
        assert isinstance(RETRY_LIMIT, int)
        assert RETRY_LIMIT > 0

