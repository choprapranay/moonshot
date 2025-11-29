from typing import List

from domain.boundaries import (
    InferenceInputBoundary,
    InferenceOutputBoundary,
    InferenceInputData,
    InferenceOutputData
)
from domain.interfaces import (
    ModelInterface,
    ModelStorageInterface,
    PreprocessorInterface
)
from domain.entities import PredictionOutput, Model


class InferenceUseCase(InferenceInputBoundary):
    def __init__(self, model_adapter: ModelInterface, preprocessor: PreprocessorInterface, storage: ModelStorageInterface, output_boundary: InferenceOutputBoundary):
        self._model_adapter = model_adapter
        self._preprocessor = preprocessor
        self._storage = storage
        self._output_boundary = output_boundary
    
    def execute(self, input_data: InferenceInputData) -> None:
        try:
            model_entity = self._storage.load_model(input_data.model_path)
            
            if model_entity is None:
                self._output_boundary.present_inference_results(InferenceOutputData(success=False, predictions=[], num_samples=0, error_message=f"Failed to load model from {input_data.model_path}"))
                return
            
            artifacts = model_entity.artifacts
            
            try:
                preprocessed_data = self._preprocessor.preprocess_for_inference(input_data.dataset, artifacts)
            except Exception as e:
                self._output_boundary.present_inference_results(InferenceOutputData(success=False, predictions=[], num_samples=0, error_message=f"Preprocessing error: {str(e)}"))
                return
            
            model = self._model_adapter.create_model(num_batters=artifacts.num_batters, num_pitch_types=artifacts.num_pitches, num_outcomes=artifacts.num_outcomes)
            model = self._model_adapter.load_model_state(model, model_entity.state_dict)
            
            raw_predictions = self._model_adapter.predict(model=model, input_data=preprocessed_data, artifacts=artifacts)
            
            predictions = []
            for pred in raw_predictions:
                predictions.append(PredictionOutput(
                    batter_name=pred['batter'],
                    pitch_type=pred['pitch_type'],
                    predicted_outcome=pred['predicted_outcome'],
                    confidence=pred['confidence'],
                    all_probabilities=pred.get('all_probabilities', {}),
                    actual_outcome=pred.get('actual_outcome')
                ))
            
            self._output_boundary.present_inference_results(InferenceOutputData(success=True, predictions=predictions, num_samples=len(predictions)))
            
        except Exception as e:
            self._output_boundary.present_inference_results(InferenceOutputData(success=False, predictions=[], num_samples=0, error_message=str(e)))