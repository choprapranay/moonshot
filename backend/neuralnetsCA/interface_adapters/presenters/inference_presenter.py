from typing import List

from domain.boundaries import (
    InferenceOutputBoundary,
    InferenceOutputData
)
from interface_adapters.presenters.view_model import (
    ViewModel, 
    InferenceResultViewModel, 
    PredictionViewModel
)


class InferencePresenter(InferenceOutputBoundary):
    def __init__(self, view_model: ViewModel):
        self._view_model = view_model
    
    def present_inference_results(self, output_data: InferenceOutputData) -> None:
        if output_data.success:
            prediction_vms = [
                PredictionViewModel(
                    batter_name=str(pred.batter_name),
                    pitch_type=str(pred.pitch_type),
                    predicted_outcome=str(pred.predicted_outcome),
                    confidence=float(pred.confidence),
                    all_probabilities=pred.all_probabilities,
                    actual_outcome=str(pred.actual_outcome) if pred.actual_outcome else None
                )
                for pred in output_data.predictions
            ]
            
            result_vm = InferenceResultViewModel(
                success=True,
                num_predictions=output_data.num_samples,
                predictions=prediction_vms
            )
        else:
            result_vm = InferenceResultViewModel(
                success=False,
                error_message=output_data.error_message
            )
        
        self._view_model.update_inference_result(result_vm)
