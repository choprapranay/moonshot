import pytest
from unittest.mock import Mock, MagicMock, patch, call
import pandas as pd
import numpy as np

from application.build_dataset_use_case import BuildDatasetUseCase
from application.train_use_case import TrainModelUseCase
from application.pred_use_case import InferenceUseCase

from domain.boundaries import (
    BuildDatasetInputData,
    BuildDatasetOutputData,
    TrainModelInputData,
    TrainModelOutputData,
    InferenceInputData,
    InferenceOutputData,
    BuildDatasetOutputBoundary,
    TrainModelOutputBoundary,
    InferenceOutputBoundary
)
from domain.interfaces import (
    DataAccessInterface,
    PreprocessorInterface,
    ModelInterface,
    ModelStorageInterface
)
from domain.entities import (
    DatasetInfo,
    TrainingConfig,
    TrainingMetrics,
    Model,
    ModelArtifacts,
    PredictionOutput
)


@pytest.fixture
def sample_raw_data():
    return pd.DataFrame({
        'batter': [1, 2, 3, 4, 5],
        'pitch_type': ['FF', 'SL', 'CH', 'FF', 'SL'],
        'release_speed': [95.0, 85.0, 82.0, 94.0, 84.0],
        'plate_x': [0.1, -0.2, 0.3, -0.1, 0.2],
        'plate_z': [2.5, 2.0, 1.8, 2.3, 2.1],
        'outcome_text': ['single', 'strikeout', 'ball', 'double', 'single']
    })


@pytest.fixture
def sample_preprocessed_data():
    return pd.DataFrame({
        'batter_encoded': [0, 1, 2, 3, 4],
        'pitch_type_encoded': [0, 1, 2, 0, 1],
        'release_speed': [95.0, 85.0, 82.0, 94.0, 84.0],
        'plate_x': [0.1, -0.2, 0.3, -0.1, 0.2],
        'plate_z': [2.5, 2.0, 1.8, 2.3, 2.1],
        'outcome_text': ['single', 'strikeout', 'ball', 'double', 'single'],
        'outcome_encoded': [0, 1, 2, 3, 0]
    })


@pytest.fixture
def sample_artifacts():
    artifacts = Mock(spec=ModelArtifacts)
    artifacts.num_batters = 5
    artifacts.num_pitches = 3
    artifacts.num_outcomes = 4
    artifacts.labels = ['single', 'strikeout', 'ball', 'double']
    artifacts.batter_encoder = Mock()
    artifacts.pitch_encoder = Mock()
    artifacts.outcome_encoder = Mock()
    artifacts.batter_pattern_encoder = Mock()
    artifacts.launch_speed_angle_encoder = Mock()
    artifacts.scaler = Mock()
    artifacts.outcome_encoder.inverse_transform = Mock(side_effect=lambda x: ['single' if x[0] == 0 else 'strikeout'])
    artifacts.batter_encoder.inverse_transform = Mock(return_value=['Player1'])
    artifacts.pitch_encoder.inverse_transform = Mock(return_value=['FF'])
    return artifacts


@pytest.fixture
def sample_training_config():
    return TrainingConfig(
        epochs=5,
        learning_rate=0.001,
        batch_size=32,
        test_split=0.2,
        random_state=42,
        save_path='test_model.pth'
    )


@pytest.fixture
def sample_model(sample_artifacts):
    return Model(
        state_dict={'layer1.weight': np.array([1, 2, 3])},
        artifacts=sample_artifacts,
        model_path='test_model.pth'
    )


@pytest.fixture
def sample_training_history():
    return [
        {'epoch': 1, 'train_loss': 1.5, 'train_accuracy': 0.4, 'test_accuracy': 0.35},
        {'epoch': 2, 'train_loss': 1.2, 'train_accuracy': 0.5, 'test_accuracy': 0.45},
        {'epoch': 3, 'train_loss': 0.9, 'train_accuracy': 0.6, 'test_accuracy': 0.55}
    ]


class TestBuildDatasetUseCase:
    
    @pytest.fixture
    def mock_data_access(self):
        mock = Mock(spec=DataAccessInterface)
        mock.get_source_name.return_value = 'test_source'
        return mock
    
    @pytest.fixture
    def mock_preprocessor(self):
        mock = Mock(spec=PreprocessorInterface)
        mock.get_feature_columns.return_value = ['release_speed', 'plate_x', 'plate_z']
        return mock
    
    @pytest.fixture
    def mock_output_boundary(self):
        return Mock(spec=BuildDatasetOutputBoundary)
    
    @pytest.fixture
    def use_case(self, mock_data_access, mock_preprocessor, mock_output_boundary):
        return BuildDatasetUseCase(
            data_access=mock_data_access,
            preprocessor=mock_preprocessor,
            output_boundary=mock_output_boundary
        )
    
    def test_execute_success(self, use_case, mock_data_access, mock_preprocessor, 
                              mock_output_boundary, sample_raw_data, sample_preprocessed_data, sample_artifacts):
        mock_data_access.fetch_data.return_value = sample_raw_data
        mock_preprocessor.preprocess_for_training.return_value = (sample_preprocessed_data, sample_artifacts)
        
        input_data = BuildDatasetInputData(
            start_date='2023-01-01',
            end_date='2023-12-31',
            data_source='test'
        )
        
        use_case.execute(input_data)
        
        mock_data_access.fetch_data.assert_called_once_with('2023-01-01', '2023-12-31')
        mock_preprocessor.preprocess_for_training.assert_called_once()
        mock_output_boundary.present_dataset.assert_called_once()
        
        call_args = mock_output_boundary.present_dataset.call_args[0][0]
        assert call_args.success is True
        assert call_args.dataset is not None
        assert call_args.raw_data is not None
        assert call_args.dataset_info is not None
        assert call_args.error_message is None
    
    def test_execute_empty_data(self, use_case, mock_data_access, mock_output_boundary):
        mock_data_access.fetch_data.return_value = pd.DataFrame()
        
        input_data = BuildDatasetInputData(
            start_date='2023-01-01',
            end_date='2023-12-31',
            data_source='test'
        )
        
        use_case.execute(input_data)
        
        mock_output_boundary.present_dataset.assert_called_once()
        call_args = mock_output_boundary.present_dataset.call_args[0][0]
        assert call_args.success is False
        assert call_args.error_message == "No data returned from data source"
    
    def test_execute_none_data(self, use_case, mock_data_access, mock_output_boundary):
        mock_data_access.fetch_data.return_value = None
        
        input_data = BuildDatasetInputData(
            start_date='2023-01-01',
            end_date='2023-12-31',
            data_source='test'
        )
        
        use_case.execute(input_data)
        
        call_args = mock_output_boundary.present_dataset.call_args[0][0]
        assert call_args.success is False
        assert "No data returned" in call_args.error_message
    
    def test_execute_fetch_exception(self, use_case, mock_data_access, mock_output_boundary):
        mock_data_access.fetch_data.side_effect = Exception("Network error")
        
        input_data = BuildDatasetInputData(
            start_date='2023-01-01',
            end_date='2023-12-31',
            data_source='test'
        )
        
        use_case.execute(input_data)
        
        call_args = mock_output_boundary.present_dataset.call_args[0][0]
        assert call_args.success is False
        assert "Network error" in call_args.error_message
    
    def test_execute_preprocessing_exception(self, use_case, mock_data_access, 
                                              mock_preprocessor, mock_output_boundary, sample_raw_data):
        mock_data_access.fetch_data.return_value = sample_raw_data
        mock_preprocessor.preprocess_for_training.side_effect = ValueError("Invalid data format")
        
        input_data = BuildDatasetInputData(
            start_date='2023-01-01',
            end_date='2023-12-31',
            data_source='test'
        )
        
        use_case.execute(input_data)
        
        call_args = mock_output_boundary.present_dataset.call_args[0][0]
        assert call_args.success is False
        assert "Invalid data format" in call_args.error_message
    
    def test_dataset_info_populated_correctly(self, use_case, mock_data_access, 
                                               mock_preprocessor, mock_output_boundary, 
                                               sample_raw_data, sample_preprocessed_data, sample_artifacts):
        mock_data_access.fetch_data.return_value = sample_raw_data
        mock_preprocessor.preprocess_for_training.return_value = (sample_preprocessed_data, sample_artifacts)
        mock_preprocessor.get_feature_columns.return_value = ['release_speed', 'plate_x', 'plate_z']
        
        input_data = BuildDatasetInputData(
            start_date='2023-01-01',
            end_date='2023-12-31',
            data_source='test'
        )
        
        use_case.execute(input_data)
        
        call_args = mock_output_boundary.present_dataset.call_args[0][0]
        dataset_info = call_args.dataset_info
        
        assert dataset_info.num_samples == len(sample_preprocessed_data)
        assert dataset_info.num_batters == sample_artifacts.num_batters
        assert dataset_info.num_pitch_types == sample_artifacts.num_pitches
        assert dataset_info.num_outcomes == sample_artifacts.num_outcomes
        assert dataset_info.feature_columns == ['release_speed', 'plate_x', 'plate_z']


class TestTrainModelUseCase:
    
    @pytest.fixture
    def mock_model_adapter(self):
        return Mock(spec=ModelInterface)
    
    @pytest.fixture
    def mock_preprocessor(self):
        return Mock(spec=PreprocessorInterface)
    
    @pytest.fixture
    def mock_storage(self):
        return Mock(spec=ModelStorageInterface)
    
    @pytest.fixture
    def mock_output_boundary(self):
        return Mock(spec=TrainModelOutputBoundary)
    
    @pytest.fixture
    def use_case(self, mock_model_adapter, mock_preprocessor, mock_storage, mock_output_boundary):
        return TrainModelUseCase(
            model_adapter=mock_model_adapter,
            preprocessor=mock_preprocessor,
            storage=mock_storage,
            output_boundary=mock_output_boundary
        )
    
    def test_execute_success(self, use_case, mock_model_adapter, mock_preprocessor, 
                              mock_storage, mock_output_boundary, sample_raw_data,
                              sample_preprocessed_data, sample_artifacts, 
                              sample_training_config, sample_training_history):
        mock_preprocessor.preprocess_for_training.return_value = (sample_preprocessed_data, sample_artifacts)
        
        mock_model = Mock()
        mock_model_adapter.create_model.return_value = mock_model
        mock_model_adapter.train.return_value = ({'layer1.weight': [1, 2, 3]}, sample_training_history)
        
        mock_storage.save_model.return_value = True
        
        input_data = TrainModelInputData(
            config=sample_training_config,
            dataset=sample_raw_data
        )
        
        use_case.execute(input_data)
        
        mock_preprocessor.preprocess_for_training.assert_called_once()
        mock_model_adapter.create_model.assert_called_once()
        mock_model_adapter.train.assert_called_once()
        mock_storage.save_model.assert_called_once()
        mock_output_boundary.present_training_complete.assert_called_once()
        
        call_args = mock_output_boundary.present_training_complete.call_args[0][0]
        assert call_args.success is True
        assert call_args.model is not None
        assert call_args.final_train_accuracy == 0.6
        assert call_args.final_test_accuracy == 0.55
        assert len(call_args.training_history) == 3
        assert call_args.save_path == 'test_model.pth'
    
    def test_execute_invalid_config(self, use_case, mock_output_boundary, sample_raw_data):
        invalid_config = TrainingConfig(
            epochs=-1,
            learning_rate=0.001,
            batch_size=32,
            test_split=0.2,
            random_state=42
        )
        
        input_data = TrainModelInputData(
            config=invalid_config,
            dataset=sample_raw_data
        )
        
        use_case.execute(input_data)
        
        call_args = mock_output_boundary.present_training_complete.call_args[0][0]
        assert call_args.success is False
        assert call_args.error_message == "Invalid training configuration"
    
    def test_execute_invalid_test_split(self, use_case, mock_output_boundary, sample_raw_data):
        invalid_config = TrainingConfig(
            epochs=5,
            learning_rate=0.001,
            batch_size=32,
            test_split=1.5,
            random_state=42
        )
        
        input_data = TrainModelInputData(
            config=invalid_config,
            dataset=sample_raw_data
        )
        
        use_case.execute(input_data)
        
        call_args = mock_output_boundary.present_training_complete.call_args[0][0]
        assert call_args.success is False
        assert "Invalid training configuration" in call_args.error_message
    
    def test_execute_preprocessing_exception(self, use_case, mock_preprocessor, 
                                              mock_output_boundary, sample_raw_data, sample_training_config):
        mock_preprocessor.preprocess_for_training.side_effect = ValueError("Preprocessing failed")
        
        input_data = TrainModelInputData(
            config=sample_training_config,
            dataset=sample_raw_data
        )
        
        use_case.execute(input_data)
        
        call_args = mock_output_boundary.present_training_complete.call_args[0][0]
        assert call_args.success is False
        assert "Preprocessing failed" in call_args.error_message
    
    def test_execute_training_exception(self, use_case, mock_model_adapter, mock_preprocessor,
                                         mock_output_boundary, sample_raw_data, 
                                         sample_preprocessed_data, sample_artifacts, sample_training_config):
        mock_preprocessor.preprocess_for_training.return_value = (sample_preprocessed_data, sample_artifacts)
        mock_model_adapter.create_model.return_value = Mock()
        mock_model_adapter.train.side_effect = RuntimeError("CUDA out of memory")
        
        input_data = TrainModelInputData(
            config=sample_training_config,
            dataset=sample_raw_data
        )
        
        use_case.execute(input_data)
        
        call_args = mock_output_boundary.present_training_complete.call_args[0][0]
        assert call_args.success is False
        assert "CUDA out of memory" in call_args.error_message
    
    def test_execute_save_failure(self, use_case, mock_model_adapter, mock_preprocessor,
                                   mock_storage, mock_output_boundary, sample_raw_data,
                                   sample_preprocessed_data, sample_artifacts, 
                                   sample_training_config, sample_training_history):
        mock_preprocessor.preprocess_for_training.return_value = (sample_preprocessed_data, sample_artifacts)
        mock_model_adapter.create_model.return_value = Mock()
        mock_model_adapter.train.return_value = ({'layer1.weight': [1, 2, 3]}, sample_training_history)
        mock_storage.save_model.return_value = False
        
        input_data = TrainModelInputData(
            config=sample_training_config,
            dataset=sample_raw_data
        )
        
        use_case.execute(input_data)
        
        call_args = mock_output_boundary.present_training_complete.call_args[0][0]
        assert call_args.success is False
        assert call_args.save_path is None
    
    def test_progress_callback_called(self, use_case, mock_model_adapter, mock_preprocessor,
                                       mock_storage, mock_output_boundary, sample_raw_data,
                                       sample_preprocessed_data, sample_artifacts, sample_training_config):
        mock_preprocessor.preprocess_for_training.return_value = (sample_preprocessed_data, sample_artifacts)
        mock_model_adapter.create_model.return_value = Mock()
        
        captured_callback = None
        def capture_train_call(*args, **kwargs):
            nonlocal captured_callback
            captured_callback = kwargs.get('progress_callback')
            if captured_callback:
                captured_callback({
                    'epoch': 1,
                    'train_loss': 1.0,
                    'train_accuracy': 0.5,
                    'test_accuracy': 0.45
                })
            return {'layer1.weight': [1, 2, 3]}, [
                {'epoch': 1, 'train_loss': 1.0, 'train_accuracy': 0.5, 'test_accuracy': 0.45}
            ]
        
        mock_model_adapter.train.side_effect = capture_train_call
        mock_storage.save_model.return_value = True
        
        input_data = TrainModelInputData(
            config=sample_training_config,
            dataset=sample_raw_data
        )
        
        use_case.execute(input_data)
        
        mock_output_boundary.present_training_progress.assert_called_once()
        progress_args = mock_output_boundary.present_training_progress.call_args[0][0]
        assert isinstance(progress_args, TrainingMetrics)
        assert progress_args.epoch == 1
        assert progress_args.train_accuracy == 0.5
    
    def test_execute_empty_training_history(self, use_case, mock_model_adapter, mock_preprocessor,
                                             mock_storage, mock_output_boundary, sample_raw_data,
                                             sample_preprocessed_data, sample_artifacts, sample_training_config):
        mock_preprocessor.preprocess_for_training.return_value = (sample_preprocessed_data, sample_artifacts)
        mock_model_adapter.create_model.return_value = Mock()
        mock_model_adapter.train.return_value = ({'layer1.weight': [1, 2, 3]}, [])
        mock_storage.save_model.return_value = True
        
        input_data = TrainModelInputData(
            config=sample_training_config,
            dataset=sample_raw_data
        )
        
        use_case.execute(input_data)
        
        call_args = mock_output_boundary.present_training_complete.call_args[0][0]
        assert call_args.success is True
        assert call_args.final_train_accuracy == 0.0
        assert call_args.final_test_accuracy == 0.0


class TestInferenceUseCase:
    
    @pytest.fixture
    def mock_model_adapter(self):
        return Mock(spec=ModelInterface)
    
    @pytest.fixture
    def mock_preprocessor(self):
        return Mock(spec=PreprocessorInterface)
    
    @pytest.fixture
    def mock_storage(self):
        return Mock(spec=ModelStorageInterface)
    
    @pytest.fixture
    def mock_output_boundary(self):
        return Mock(spec=InferenceOutputBoundary)
    
    @pytest.fixture
    def use_case(self, mock_model_adapter, mock_preprocessor, mock_storage, mock_output_boundary):
        return InferenceUseCase(
            model_adapter=mock_model_adapter,
            preprocessor=mock_preprocessor,
            storage=mock_storage,
            output_boundary=mock_output_boundary
        )
    
    def test_execute_success(self, use_case, mock_model_adapter, mock_preprocessor,
                              mock_storage, mock_output_boundary, sample_raw_data,
                              sample_preprocessed_data, sample_model, sample_artifacts):
        mock_storage.load_model.return_value = sample_model
        mock_preprocessor.preprocess_for_inference.return_value = sample_preprocessed_data
        
        mock_trained_model = Mock()
        mock_model_adapter.create_model.return_value = mock_trained_model
        mock_model_adapter.load_model_state.return_value = mock_trained_model
        mock_model_adapter.predict.return_value = [
            {
                'batter': 'Player1',
                'pitch_type': 'FF',
                'predicted_outcome': 'single',
                'confidence': 0.85,
                'all_probabilities': {'single': 0.85, 'strikeout': 0.10, 'ball': 0.05},
                'actual_outcome': 'single'
            },
            {
                'batter': 'Player2',
                'pitch_type': 'SL',
                'predicted_outcome': 'strikeout',
                'confidence': 0.72,
                'all_probabilities': {'single': 0.15, 'strikeout': 0.72, 'ball': 0.13},
                'actual_outcome': 'strikeout'
            }
        ]
        
        input_data = InferenceInputData(
            model_path='test_model.pth',
            dataset=sample_raw_data,
            use_saved_encoders=True
        )
        
        use_case.execute(input_data)
        
        mock_storage.load_model.assert_called_once_with('test_model.pth')
        mock_preprocessor.preprocess_for_inference.assert_called_once()
        mock_model_adapter.create_model.assert_called_once()
        mock_model_adapter.load_model_state.assert_called_once()
        mock_model_adapter.predict.assert_called_once()
        mock_output_boundary.present_inference_results.assert_called_once()
        
        call_args = mock_output_boundary.present_inference_results.call_args[0][0]
        assert call_args.success is True
        assert call_args.num_samples == 2
        assert len(call_args.predictions) == 2
        assert call_args.predictions[0].batter_name == 'Player1'
        assert call_args.predictions[0].confidence == 0.85
        assert 'single' in call_args.predictions[0].all_probabilities
    
    def test_execute_model_load_failure(self, use_case, mock_storage, mock_output_boundary, sample_raw_data):
        mock_storage.load_model.return_value = None
        
        input_data = InferenceInputData(
            model_path='nonexistent_model.pth',
            dataset=sample_raw_data,
            use_saved_encoders=True
        )
        
        use_case.execute(input_data)
        
        call_args = mock_output_boundary.present_inference_results.call_args[0][0]
        assert call_args.success is False
        assert "Failed to load model" in call_args.error_message
        assert call_args.num_samples == 0
        assert len(call_args.predictions) == 0
    
    def test_execute_preprocessing_exception(self, use_case, mock_preprocessor, mock_storage,
                                              mock_output_boundary, sample_raw_data, sample_model):
        mock_storage.load_model.return_value = sample_model
        mock_preprocessor.preprocess_for_inference.side_effect = ValueError("Unknown batter in inference data")
        
        input_data = InferenceInputData(
            model_path='test_model.pth',
            dataset=sample_raw_data,
            use_saved_encoders=True
        )
        
        use_case.execute(input_data)
        
        call_args = mock_output_boundary.present_inference_results.call_args[0][0]
        assert call_args.success is False
        assert "Preprocessing error" in call_args.error_message
        assert "Unknown batter" in call_args.error_message
    
    def test_execute_prediction_exception(self, use_case, mock_model_adapter, mock_preprocessor,
                                           mock_storage, mock_output_boundary, sample_raw_data,
                                           sample_preprocessed_data, sample_model):
        mock_storage.load_model.return_value = sample_model
        mock_preprocessor.preprocess_for_inference.return_value = sample_preprocessed_data
        mock_model_adapter.create_model.return_value = Mock()
        mock_model_adapter.load_model_state.return_value = Mock()
        mock_model_adapter.predict.side_effect = RuntimeError("Model inference failed")
        
        input_data = InferenceInputData(
            model_path='test_model.pth',
            dataset=sample_raw_data,
            use_saved_encoders=True
        )
        
        use_case.execute(input_data)
        
        call_args = mock_output_boundary.present_inference_results.call_args[0][0]
        assert call_args.success is False
        assert "Model inference failed" in call_args.error_message
    
    def test_execute_empty_predictions(self, use_case, mock_model_adapter, mock_preprocessor,
                                        mock_storage, mock_output_boundary, sample_raw_data,
                                        sample_preprocessed_data, sample_model):
        mock_storage.load_model.return_value = sample_model
        mock_preprocessor.preprocess_for_inference.return_value = sample_preprocessed_data
        mock_model_adapter.create_model.return_value = Mock()
        mock_model_adapter.load_model_state.return_value = Mock()
        mock_model_adapter.predict.return_value = []
        
        input_data = InferenceInputData(
            model_path='test_model.pth',
            dataset=sample_raw_data,
            use_saved_encoders=True
        )
        
        use_case.execute(input_data)
        
        call_args = mock_output_boundary.present_inference_results.call_args[0][0]
        assert call_args.success is True
        assert call_args.num_samples == 0
        assert len(call_args.predictions) == 0
    
    def test_execute_prediction_without_all_probabilities(self, use_case, mock_model_adapter,
                                                           mock_preprocessor, mock_storage,
                                                           mock_output_boundary, sample_raw_data,
                                                           sample_preprocessed_data, sample_model):
        mock_storage.load_model.return_value = sample_model
        mock_preprocessor.preprocess_for_inference.return_value = sample_preprocessed_data
        mock_model_adapter.create_model.return_value = Mock()
        mock_model_adapter.load_model_state.return_value = Mock()
        mock_model_adapter.predict.return_value = [
            {
                'batter': 'Player1',
                'pitch_type': 'FF',
                'predicted_outcome': 'single',
                'confidence': 0.85
            }
        ]
        
        input_data = InferenceInputData(
            model_path='test_model.pth',
            dataset=sample_raw_data,
            use_saved_encoders=True
        )
        
        use_case.execute(input_data)
        
        call_args = mock_output_boundary.present_inference_results.call_args[0][0]
        assert call_args.success is True
        assert call_args.predictions[0].all_probabilities == {}
    
    def test_execute_prediction_with_actual_outcome(self, use_case, mock_model_adapter,
                                                     mock_preprocessor, mock_storage,
                                                     mock_output_boundary, sample_raw_data,
                                                     sample_preprocessed_data, sample_model):
        mock_storage.load_model.return_value = sample_model
        mock_preprocessor.preprocess_for_inference.return_value = sample_preprocessed_data
        mock_model_adapter.create_model.return_value = Mock()
        mock_model_adapter.load_model_state.return_value = Mock()
        mock_model_adapter.predict.return_value = [
            {
                'batter': 'Player1',
                'pitch_type': 'FF',
                'predicted_outcome': 'single',
                'confidence': 0.85,
                'all_probabilities': {'single': 0.85, 'strikeout': 0.15},
                'actual_outcome': 'double'
            }
        ]
        
        input_data = InferenceInputData(
            model_path='test_model.pth',
            dataset=sample_raw_data,
            use_saved_encoders=True
        )
        
        use_case.execute(input_data)
        
        call_args = mock_output_boundary.present_inference_results.call_args[0][0]
        assert call_args.predictions[0].actual_outcome == 'double'
        assert call_args.predictions[0].predicted_outcome == 'single'
    
    def test_execute_storage_exception(self, use_case, mock_storage, mock_output_boundary, sample_raw_data):
        mock_storage.load_model.side_effect = IOError("Disk read error")
        
        input_data = InferenceInputData(
            model_path='test_model.pth',
            dataset=sample_raw_data,
            use_saved_encoders=True
        )
        
        use_case.execute(input_data)
        
        call_args = mock_output_boundary.present_inference_results.call_args[0][0]
        assert call_args.success is False
        assert "Disk read error" in call_args.error_message


class TestTrainingConfig:
    
    def test_valid_config(self):
        config = TrainingConfig(
            epochs=10,
            learning_rate=0.001,
            batch_size=32,
            test_split=0.2,
            random_state=42
        )
        assert config.validate() is True
    
    def test_invalid_epochs(self):
        config = TrainingConfig(epochs=0, learning_rate=0.001, batch_size=32, test_split=0.2)
        assert config.validate() is False
        
        config = TrainingConfig(epochs=-5, learning_rate=0.001, batch_size=32, test_split=0.2)
        assert config.validate() is False
    
    def test_invalid_learning_rate(self):
        config = TrainingConfig(epochs=10, learning_rate=0, batch_size=32, test_split=0.2)
        assert config.validate() is False
        
        config = TrainingConfig(epochs=10, learning_rate=-0.001, batch_size=32, test_split=0.2)
        assert config.validate() is False
    
    def test_invalid_test_split(self):
        config = TrainingConfig(epochs=10, learning_rate=0.001, batch_size=32, test_split=0)
        assert config.validate() is False
        
        config = TrainingConfig(epochs=10, learning_rate=0.001, batch_size=32, test_split=1)
        assert config.validate() is False
        
        config = TrainingConfig(epochs=10, learning_rate=0.001, batch_size=32, test_split=1.5)
        assert config.validate() is False
    
    def test_invalid_batch_size(self):
        config = TrainingConfig(epochs=10, learning_rate=0.001, batch_size=0, test_split=0.2)
        assert config.validate() is False
        
        config = TrainingConfig(epochs=10, learning_rate=0.001, batch_size=-32, test_split=0.2)
        assert config.validate() is False


class TestTrainingMetrics:
    
    def test_to_dict(self):
        metrics = TrainingMetrics(
            epoch=5,
            train_loss=0.5,
            train_accuracy=0.85,
            test_accuracy=0.80
        )
        
        result = metrics.to_dict()
        
        assert result['epoch'] == 5
        assert result['train_loss'] == 0.5
        assert result['train_accuracy'] == 0.85
        assert result['test_accuracy'] == 0.80


class TestPredictionOutput:
    
    def test_to_dict(self):
        prediction = PredictionOutput(
            batter_name='Player1',
            pitch_type='FF',
            predicted_outcome='single',
            confidence=0.85,
            all_probabilities={'single': 0.85, 'strikeout': 0.15},
            actual_outcome='single'
        )
        
        result = prediction.to_dict()
        
        assert result['batter'] == 'Player1'
        assert result['pitch_type'] == 'FF'
        assert result['predicted_outcome'] == 'single'
        assert result['confidence'] == 0.85
        assert result['actual_outcome'] == 'single'
        assert result['all_probabilities'] == {'single': 0.85, 'strikeout': 0.15}
    
    def test_default_all_probabilities(self):
        prediction = PredictionOutput(
            batter_name='Player1',
            pitch_type='FF',
            predicted_outcome='single',
            confidence=0.85
        )
        
        assert prediction.all_probabilities == {}


class TestModel:
    
    def test_to_save_dict(self, sample_artifacts):
        model = Model(
            state_dict={'layer1.weight': [1, 2, 3]},
            artifacts=sample_artifacts,
            model_path='test_model.pth'
        )
        
        result = model.to_save_dict()
        
        assert 'model_state_dict' in result
        assert result['model_state_dict'] == {'layer1.weight': [1, 2, 3]}
        assert 'batter_encoder' in result
        assert 'num_batters' in result
        assert result['num_batters'] == 5
    
    def test_from_save_dict(self, sample_artifacts):
        save_dict = {
            'model_state_dict': {'layer1.weight': [1, 2, 3]},
            'batter_encoder': sample_artifacts.batter_encoder,
            'pitch_encoder': sample_artifacts.pitch_encoder,
            'outcome_encoder': sample_artifacts.outcome_encoder,
            'batter_pattern_encoder': sample_artifacts.batter_pattern_encoder,
            'launch_speed_angle_encoder': sample_artifacts.launch_speed_angle_encoder,
            'scaler': sample_artifacts.scaler,
            'labels': sample_artifacts.labels,
            'num_batters': 5,
            'num_pitches': 3,
            'num_outcomes': 4
        }
        
        model = Model.from_save_dict(save_dict, model_path='loaded_model.pth')
        
        assert model.state_dict == {'layer1.weight': [1, 2, 3]}
        assert model.model_path == 'loaded_model.pth'
        assert model.artifacts.num_batters == 5
        assert model.artifacts.num_pitches == 3
        assert model.artifacts.num_outcomes == 4


class TestDatasetInfo:
    
    def test_creation(self):
        info = DatasetInfo(
            num_samples=1000,
            num_batters=50,
            num_pitch_types=10,
            num_outcomes=15,
            feature_columns=['release_speed', 'plate_x', 'plate_z'],
            outcome_distribution={'single': 200, 'strikeout': 300, 'ball': 500}
        )
        
        assert info.num_samples == 1000
        assert info.num_batters == 50
        assert info.num_pitch_types == 10
        assert info.num_outcomes == 15
        assert len(info.feature_columns) == 3
        assert info.outcome_distribution['single'] == 200
    
    def test_default_outcome_distribution(self):
        info = DatasetInfo(
            num_samples=100,
            num_batters=5,
            num_pitch_types=3,
            num_outcomes=4,
            feature_columns=['col1', 'col2']
        )
        
        assert info.outcome_distribution == {}


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
