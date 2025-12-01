from typing import Optional
import pandas as pd

from domain.boundaries import (
    InferenceInputBoundary,
    InferenceInputData
)


class InferenceController:
    def __init__(self, inference_use_case: InferenceInputBoundary):
        self._use_case = inference_use_case
    
    def run_inference(self, dataset: pd.DataFrame, model_path: str = 'batter_outcome_model.pth') -> None:
        input_data = InferenceInputData(model_path=model_path, dataset=dataset, use_saved_encoders=True)
        
        self._use_case.execute(input_data)
    
    def run_inference_from_csv(self, csv_path: str, model_path: str = 'batter_outcome_model.pth') -> None:
        dataset = pd.read_csv(csv_path)
        self.run_inference(dataset, model_path)
