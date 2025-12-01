from typing import Optional
import pandas as pd

from domain.boundaries import (
    BuildDatasetInputBoundary,
    BuildDatasetOutputBoundary,
    BuildDatasetInputData,
    BuildDatasetOutputData
)
from domain.interfaces import DataAccessInterface, PreprocessorInterface
from domain.entities import DatasetInfo


class BuildDatasetUseCase(BuildDatasetInputBoundary):
    def __init__(self, data_access: DataAccessInterface, preprocessor: PreprocessorInterface, output_boundary: BuildDatasetOutputBoundary):
        self._data_access = data_access
        self._preprocessor = preprocessor
        self._output_boundary = output_boundary
    
    def execute(self, input_data: BuildDatasetInputData) -> None:
        try:
            raw_data = self._data_access.fetch_data(input_data.start_date, input_data.end_date)
            
            if raw_data is None or raw_data.empty:
                self._output_boundary.present_dataset(BuildDatasetOutputData(success=False, dataset=None, raw_data=None, dataset_info=None, error_message="No data returned from data source"))
                return
            
            
            preprocessed_data, artifacts = self._preprocessor.preprocess_for_training(raw_data)
            
            outcome_distribution = {}
            if 'outcome_text' in preprocessed_data.columns:
                outcome_distribution = preprocessed_data['outcome_text'].value_counts().to_dict()
            
            dataset_info = DatasetInfo(
                num_samples=len(preprocessed_data),
                num_batters=artifacts.num_batters,
                num_pitch_types=artifacts.num_pitches,
                num_outcomes=artifacts.num_outcomes,
                feature_columns=self._preprocessor.get_feature_columns(),
                outcome_distribution=outcome_distribution
            )
            
            self._output_boundary.present_dataset(BuildDatasetOutputData(success=True, dataset=preprocessed_data, raw_data=raw_data.copy(), dataset_info=dataset_info))
            
        except Exception as e:
            self._output_boundary.present_dataset(BuildDatasetOutputData(success=False, dataset=None, raw_data=None, dataset_info=None, error_message=str(e)))