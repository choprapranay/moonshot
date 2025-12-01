from typing import List
from sklearn.model_selection import train_test_split

from domain.boundaries import (
    TrainModelInputBoundary,
    TrainModelOutputBoundary,
    TrainModelInputData,
    TrainModelOutputData
)
from domain.interfaces import (
    ModelInterface,
    ModelStorageInterface,
    PreprocessorInterface
)
from domain.entities import (
    TrainingConfig,
    TrainingMetrics,
    Model,
    ModelArtifacts
)


class TrainModelUseCase(TrainModelInputBoundary):
    def __init__(self, model_adapter: ModelInterface, preprocessor: PreprocessorInterface, storage: ModelStorageInterface, output_boundary: TrainModelOutputBoundary):
        self._model_adapter = model_adapter
        self._preprocessor = preprocessor
        self._storage = storage
        self._output_boundary = output_boundary
    
    def execute(self, input_data: TrainModelInputData) -> None:
        try:
            config = input_data.config
            if not config.validate():
                self._output_boundary.present_training_complete(TrainModelOutputData(
                    success=False,
                    model=None,
                    final_train_accuracy=0.0,
                    final_test_accuracy=0.0,
                    training_history=[],
                    save_path=None,
                    error_message="Invalid training configuration"
                ))
                return
            
            preprocessed_data, artifacts = self._preprocessor.preprocess_for_training(input_data.dataset)
            
            train_df, test_df = train_test_split(preprocessed_data, test_size=config.test_split, random_state=config.random_state)
            
            model = self._model_adapter.create_model(num_batters=artifacts.num_batters, num_pitch_types=artifacts.num_pitches, num_outcomes=artifacts.num_outcomes)
            
            def progress_callback(metrics_dict: dict):
                metrics = TrainingMetrics(epoch=metrics_dict['epoch'], train_loss=metrics_dict['train_loss'], train_accuracy=metrics_dict['train_accuracy'], test_accuracy=metrics_dict['test_accuracy'])
                self._output_boundary.present_training_progress(metrics)
            
            state_dict, history = self._model_adapter.train(model=model, train_data=train_df, test_data=test_df, config=config, progress_callback=progress_callback)
            
            training_history = [
                TrainingMetrics(epoch=int(h['epoch']), train_loss=float(h['train_loss']), train_accuracy=float(h['train_accuracy']), test_accuracy=float(h['test_accuracy']))
                for h in history
            ]
            
            trained_model = Model(state_dict=state_dict, artifacts=artifacts, model_path=config.save_path)
            
            save_success = self._storage.save_model(trained_model, config.save_path)
            
            final_train_acc = training_history[-1].train_accuracy if training_history else 0.0
            final_test_acc = training_history[-1].test_accuracy if training_history else 0.0
            
            self._output_boundary.present_training_complete(TrainModelOutputData(
                success=save_success,
                model=trained_model,
                final_train_accuracy=final_train_acc,
                final_test_accuracy=final_test_acc,
                training_history=training_history,
                save_path=config.save_path if save_success else None
            ))
            
        except Exception as e:
            self._output_boundary.present_training_complete(TrainModelOutputData(
                success=False,
                model=None,
                final_train_accuracy=0.0,
                final_test_accuracy=0.0,
                training_history=[],
                save_path=None,
                error_message=str(e)
            ))