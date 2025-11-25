from typing import Optional
import pandas as pd

from domain.boundaries import (
    BuildDatasetInputBoundary,
    BuildDatasetInputData
)


class DatasetController:
    def __init__(self, build_dataset_use_case: BuildDatasetInputBoundary):
        self._use_case = build_dataset_use_case
    
    def build_dataset(self, start_date: str, end_date: str, data_source: str = 'pybaseball') -> None:
        if data_source not in ['pybaseball', 'supabase']:
            raise ValueError(f"Invalid data source: {data_source}. Must be 'pybaseball' or 'supabase'")
        
        input_data = BuildDatasetInputData(
            start_date=start_date,
            end_date=end_date,
            data_source=data_source
        )
        
        self._use_case.execute(input_data)
